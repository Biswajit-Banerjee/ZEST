"""
Token-level MLM pretraining for ThemeEncoder.

Usage:
    python -m zest.training.pretrain --model base --epochs 8
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from zest.model.theme_encoder import create_theme_encoder, ThemeEncoder
from zest.tokenizer import ZESTTokenizer
from zest.data.pretrain import MLMDataset, mlm_collate_fn
from zest.training.losses import UncertaintyWeightedLoss
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, setup_logging,
    make_run_dir, save_environment, save_model_arch,
)

GRAD_CLIP = 1.0


def _is_oom(err):
    return "out of memory" in str(err).lower()


def train_step(batch, model, device, amp, ga,
               combine_loss=None, token_mlm_weight=0.7, residue_mlm_weight=0.3,
               token_only=False, residue_only=False):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["token_labels"].to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        # Token-level MLM
        logits = model(ids, mask)
        loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     labels.view(-1), ignore_index=-100)

        if token_only or "token_lengths" not in batch:
            scaled = loss_token / ga
        else:
            # Residue-level MLM
            tl = batch["token_lengths"].to(device)
            gi = batch["global_indices"].to(device)
            aa = batch["aa_labels"].to(device)
            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            res_logits = m.forward_residue_mlm(ids, mask, tl, gi)
            ml = min(res_logits.shape[1], aa.shape[1])
            loss_residue = F.cross_entropy(
                res_logits[:, :ml, :].reshape(-1, 20),
                aa[:, :ml].reshape(-1), ignore_index=-100)

            if residue_only:
                # Bypass UncertaintyWeightedLoss — use raw residue loss only
                scaled = loss_residue / ga
                loss_stats = {"w_tok": 0.0, "w_res": 1.0,
                              "pct_tok": 0.0, "pct_res": 1.0}
            else:
                balanced, loss_stats = combine_loss(
                    loss_token, loss_residue,
                    priority_token=token_mlm_weight,
                    priority_residue=residue_mlm_weight)
                scaled = balanced / ga

        scaled.backward()

    with torch.no_grad():
        valid = labels != -100
        tok_acc = torch.tensor(0.0, device=device)
        if valid.any():
            tok_acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean()

    if token_only or "token_lengths" not in batch:
        loss_val = loss_token.detach().item()
    elif residue_only:
        loss_val = loss_residue.detach().item()
    else:
        loss_val = balanced.detach().item()

    metrics = {
        "loss": loss_val,
        "tok_loss": loss_token.detach().item(),
        "tok_acc": tok_acc.item(),
    }

    if not token_only and "token_lengths" in batch:
        with torch.no_grad():
            rp = res_logits[:, :ml, :].argmax(-1)
            rv = aa[:, :ml] != -100
            res_acc = (rp[rv] == aa[:, :ml][rv]).float().mean() if rv.any() else torch.tensor(0.0, device=device)
        metrics.update({
            "res_loss": loss_residue.detach().item(),
            "res_acc": res_acc.item(),
            "w_tok": loss_stats["w_tok"],
            "w_res": loss_stats["w_res"],
            "pct_tok": loss_stats["pct_tok"],
            "pct_res": loss_stats["pct_res"],
        })

    return metrics


@torch.no_grad()
def eval_step(batch, model, device, amp, eval_residue=True):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["token_labels"].to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(ids, mask)
        loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     labels.view(-1), ignore_index=-100)

    valid = labels != -100
    tok_acc = torch.tensor(0.0, device=device)
    if valid.any():
        tok_acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean()

    metrics = {"tok_loss": loss_token.item(), "tok_acc": tok_acc.item()}

    if eval_residue and "token_lengths" in batch:
        tl = batch["token_lengths"].to(device)
        gi = batch["global_indices"].to(device)
        aa = batch["aa_labels"].to(device)
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            res_logits = m.forward_residue_mlm(ids, mask, tl, gi)
        ml = min(res_logits.shape[1], aa.shape[1])
        loss_residue = F.cross_entropy(
            res_logits[:, :ml, :].reshape(-1, 20),
            aa[:, :ml].reshape(-1), ignore_index=-100)
        rp = res_logits[:, :ml, :].argmax(-1)
        rv = aa[:, :ml] != -100
        res_acc = (rp[rv] == aa[:, :ml][rv]).float().mean() if rv.any() else torch.tensor(0.0, device=device)
        metrics.update({"res_loss": loss_residue.item(), "res_acc": res_acc.item()})

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="ThemeEncoder MLM pretraining")
    p.add_argument("--model",            default="base",  choices=["test", "base", "deep"],  help="Model architecture")
    p.add_argument("--epochs",           default=8,       type=int,    help="Number of training epochs")
    p.add_argument("--batch-size",       default=128,     type=int,    help="Batch size per GPU")
    p.add_argument("--grad-accum-steps", default=8,       type=int,    help="Gradient accumulation steps") # 128 * 8 = 1024
    p.add_argument("--lr",               default=5e-4,    type=float,  help="Learning rate")
    p.add_argument("--warmup-ratio",     default=0.05,    type=float,  help="Warmup ratio")
    p.add_argument("--mask-prob",        default=0.3,     type=float,  help="Masking probability") # 30% masking
    p.add_argument("--trie-dropout",     default=0.15,    type=float,  help="Trie level dropout")  # 15% trie level dropout
    p.add_argument("--max-tokens",       default=512,     type=int,    help="Maximum sequence length")
    p.add_argument("--num-workers",      default=8,       type=int,    help="Number of data loading workers")
    p.add_argument("--seed",             default=42,      type=int,    help="Random seed for reproducibility")
    p.add_argument("--val-every",        default=1,       type=int,    help="Run validation every N epochs")
    p.add_argument("--val-shards",       default=None, type=int,       help="Limit validation to first K shards (longest seqs)")
    p.add_argument("--tokenizer",        default="data/tokens/tedlh_zest_32K.json",     help="Tokenizer file path")
    p.add_argument("--train-data",       required=True,                help="Training sequences (FASTA dir/file, parquet, or txt)")
    p.add_argument("--val-data",         required=True,                help="Validation sequences (FASTA dir/file, parquet, or txt)")
    p.add_argument("--shards-per-epoch", default=None, type=int,       help="Parquet shards to load per epoch (streaming mode)")
    p.add_argument("--unique-shards",    action="store_true",          help="Each shard seen exactly once (no repeat across epochs)")
    p.add_argument("--output-dir",       default="outputs/models",     help="Output directory for checkpoints")
    p.add_argument("--run-dir",          default=None,                 help="Run directory (inferred from --checkpoint if given)")
    p.add_argument("--checkpoint",       default=None,                 help="Resume from checkpoint (infers run-dir)")
    p.add_argument("--amp",              action="store_true",          help="Use automatic mixed precision")
    p.add_argument("--keep-all-checkpoints", action="store_true",      help="Keep all checkpoints (default: last 3)")
    p.add_argument("--domain",           default="protein", choices=["protein", "code"], help="Data domain")
    p.add_argument("--ast-language",     default="python",             help="Language for AST encoder (code domain only)")
    p.add_argument("--seq-column",       default="sequence",           help="Column name for sequences in parquet")
    p.add_argument("--tokenizer-type",   default="zest", choices=["zest", "bpe", "char"], help="Tokenizer type (zest=trie, bpe=subword baseline, char=character-level)")
    p.add_argument("--token-mlm-weight",  default=0.7,  type=float, help="Priority weight for token-level MLM")
    p.add_argument("--residue-mlm-weight", default=0.3, type=float, help="Priority weight for residue-level MLM (0 disables)")
    p.add_argument("--token-only",        action="store_true",       help="Token MLM only; skip residue head")
    p.add_argument("--freeze-core",       action="store_true",       help="Freeze encoder core; only train residue head + global_pos_emb")
    p.add_argument("--optim-checkpoint",  default=None,              help="Load optimizer/scheduler from this checkpoint (model from --checkpoint)")
    
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            amp = True

    # Determine run directory (explicit --run-dir takes priority)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.checkpoint:
        run_dir = Path(args.checkpoint).parent.parent
    else:
        run_dir = make_run_dir(args.output_dir, args.model)

    keep_n = 999 if args.keep_all_checkpoints else 3
    ckpt_mgr = CheckpointManager(run_dir, phase="pretrain", keep_last_n=keep_n)
    log = setup_logging("pretrain", str(ckpt_mgr.log_path))

    if args.tokenizer_type == "bpe":
        if args.domain == "code":
            from experiments.code.build_bpe_tokenizer import CodeBPETokenizer
            tokenizer = CodeBPETokenizer(args.tokenizer)
        else:
            from experiments.biology.build_protein_bpe_tokenizer import ProteinBPETokenizer
            tokenizer = ProteinBPETokenizer(args.tokenizer)
        log.info("Using BPE tokenizer: vocab_size=%d", tokenizer.vocab_size)
    elif args.tokenizer_type == "char":
        from experiments.biology.char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        log.info("Using char tokenizer: vocab_size=%d", tokenizer.vocab_size)
    else:
        tokenizer = ZESTTokenizer(args.tokenizer)

    # Build domain-specific encoder (code: role/BPE, protein: None)
    encoder = None
    lang_column = None
    if args.domain == "code":
        if args.tokenizer_type == "bpe":
            from experiments.code.build_bpe_tokenizer import make_multilang_bpe_encoder
            encoder = make_multilang_bpe_encoder(args.tokenizer)
            log.info("Code domain: BPE encoder (comment-stripped)")
        else:
            from experiments.code.build_role_tokenizer import make_multilang_role_encoder
            encoder = make_multilang_role_encoder()
            log.info("Code domain: multi-language role encoder (dropout=%.2f)", args.trie_dropout)
        lang_column = "language"

    residue_mlm = (not args.token_only) and args.residue_mlm_weight > 0
    train_ds = MLMDataset(
        args.train_data, tokenizer,
        is_training=True,
        max_tokens=args.max_tokens, trie_dropout=args.trie_dropout,
        shards_per_epoch=args.shards_per_epoch,
        unique_shards=args.unique_shards, seed=args.seed,
        seq_column=args.seq_column,
        lang_column=lang_column,
        encoder=encoder,
        residue_mlm=residue_mlm,
    )
    val_ds = MLMDataset(
        args.val_data, tokenizer,
        is_training=False,
        max_tokens=args.max_tokens,
        seq_column=args.seq_column,
        lang_column=lang_column,
        encoder=encoder,
        residue_mlm=residue_mlm,
        shards_per_epoch=args.val_shards,
    )

    collate = lambda b: mlm_collate_fn(
        b, tokenizer.pad_id, tokenizer.mask_id, tokenizer.vocab_size,
        mask_prob=args.mask_prob,
    )

    if args.checkpoint:
        model, _ = ThemeEncoder.load_pretrained(args.checkpoint, device="cpu")
        model = model.to(device)
    else:
        model = create_theme_encoder(args.model, tokenizer.vocab_size, args.max_tokens).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s (%.1fM params)", args.model, total_params / 1e6)

    # Freeze core if requested (only residue head + global_pos_emb train)
    if args.freeze_core:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.profile_expansion_head.parameters():
            p.requires_grad = True
        model.global_pos_emb.weight.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Core frozen — training %.1fM / %.1fM params (residue head + pos_emb)",
                 trainable / 1e6, total_params / 1e6)

    bs, ga = args.batch_size, args.grad_accum_steps
    nw = args.num_workers
    pf = 8 if nw > 0 else None

    def make_loader(ds, shuffle=True, drop_last=True, persistent=True):
        pw = nw > 0 and persistent and not ds.streaming
        return DataLoader(
            ds, bs, shuffle=shuffle, num_workers=nw,
            pin_memory=True, collate_fn=collate, drop_last=drop_last,
            prefetch_factor=pf, persistent_workers=pw,
        )

    # Loss + optimizers
    token_only = args.token_only or args.residue_mlm_weight <= 0
    residue_only = args.freeze_core  # bypass UncertaintyWeightedLoss when core frozen
    combine_loss = None
    opt_loss = None
    if not token_only and not residue_only:
        combine_loss = UncertaintyWeightedLoss(num_tasks=2).to(device)
        opt_loss = optim.AdamW(combine_loss.parameters(), lr=1e-3, weight_decay=0.0)

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(train_params, lr=args.lr)

    loader = make_loader(train_ds)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)

    steps_ep = max(1, len(loader) // ga)
    sched = build_cosine_schedule(
        opt, args.epochs * steps_ep,
        int(args.epochs * steps_ep * args.warmup_ratio)
    )

    # Save run metadata
    config = {
        "model": args.model,
        "params_M": round(total_params / 1e6, 2),
        "epochs": args.epochs,
        "batch_size": bs,
        "grad_accum_steps": ga,
        "effective_batch": bs * ga,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "scheduler": "cosine",
        "total_steps": args.epochs * steps_ep,
        "warmup_steps": int(args.epochs * steps_ep * args.warmup_ratio),
        "steps_per_epoch": steps_ep,
        "mask_prob": args.mask_prob,
        "trie_dropout": args.trie_dropout,
        "max_tokens": args.max_tokens,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "amp": amp,
        "shards_per_epoch": args.shards_per_epoch,
        "streaming": train_ds.streaming,
        "total_train_shards": len(train_ds._shard_paths) if train_ds.streaming else None,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "token_only": token_only,
        "token_mlm_weight": args.token_mlm_weight,
        "residue_mlm_weight": args.residue_mlm_weight,
    }
    with open(ckpt_mgr.run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    save_environment(ckpt_mgr.run_dir)
    save_model_arch(ckpt_mgr.run_dir, model)
    log.info("Run directory: %s", ckpt_mgr.run_dir)

    # Resume from checkpoint
    start_epoch = 1
    if args.checkpoint:
        skip_optim = args.freeze_core or args.optim_checkpoint is not None
        restore_opt = None if skip_optim else opt
        restore_sched = None if skip_optim else sched
        ckpt = ckpt_mgr.load(args.checkpoint, model, restore_opt, restore_sched, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1

        # Optionally load optimizer/scheduler from a different checkpoint
        if args.optim_checkpoint:
            ckpt_opt = torch.load(args.optim_checkpoint, map_location=device, weights_only=False)
            opt.load_state_dict(ckpt_opt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_opt:
                sched.load_state_dict(ckpt_opt["scheduler_state_dict"])
            log.info("Optimizer/scheduler restored from %s", args.optim_checkpoint)
            del ckpt_opt

        log.info("Resumed from %s (epoch %d, best_val=%.4f)%s",
                 args.checkpoint, start_epoch - 1, ckpt_mgr.best_val_loss,
                 " [freeze-core: fresh optimizer+scheduler]" if args.freeze_core else "")

    eval_residue = not token_only
    global_step = (start_epoch - 1) * max(1, len(loader) // ga)
    for ep in range(start_epoch, args.epochs + 1):
        # Rotate shards if streaming (must recreate loader for worker visibility)
        if train_ds.streaming:
            train_ds.load_epoch(ep)
            loader = make_loader(train_ds)

        # --- Train ---
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)

        # Accumulators for epoch metrics
        avg_keys = ["loss", "tok_loss", "tok_acc"]
        if not token_only:
            avg_keys += ["res_loss", "res_acc", "pct_tok", "pct_res"]
        sums = {k: 0.0 for k in avg_keys}
        n_steps = 0

        pbar = tqdm(loader, desc=f"Epoch {ep}", leave=False)
        for i, batch in enumerate(pbar):
            try:
                m = train_step(
                    batch, model, device, amp, ga,
                    combine_loss=combine_loss,
                    token_mlm_weight=args.token_mlm_weight,
                    residue_mlm_weight=args.residue_mlm_weight,
                    token_only=token_only,
                    residue_only=residue_only,
                )
            except RuntimeError as e:
                if _is_oom(e):
                    log.warning("OOM at step %d — skipping batch", i)
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    continue
                raise

            if math.isnan(m["tok_loss"]) or math.isnan(m.get("res_loss", 0)):
                opt.zero_grad(set_to_none=True)
                continue

            for k in sums:
                if k in m:
                    sums[k] += m[k]
            n_steps += 1

            if n_steps % 50 == 0:
                d = max(1, n_steps)
                tok_ppl = math.exp(min(sums["tok_loss"] / d, 20.0))
                postfix = {
                    "loss": f"{sums['loss']/d:.4f}",
                    "t_acc": f"{sums['tok_acc']/d:.3f}",
                    "t_ppl": f"{tok_ppl:.1f}",
                    "lr": f"{sched.get_last_lr()[0]:.2e}",
                }
                if not token_only and sums.get("res_acc", 0) > 0:
                    postfix["r_acc"] = f"{sums['res_acc']/d:.3f}"
                    postfix["r_ppl"] = f"{math.exp(min(sums['res_loss']/d, 20.0)):.1f}"
                if not token_only and sums.get("pct_tok", 0) > 0:
                    postfix["split"] = f"{sums['pct_tok']/d:.0%}/{sums['pct_res']/d:.0%}"
                pbar.set_postfix(postfix)

            if (i + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                if opt_loss is not None:
                    opt_loss.step()
                    opt_loss.zero_grad(set_to_none=True)
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

        # Flush remaining accumulated gradients
        if (i + 1) % ga != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if opt_loss is not None:
                opt_loss.step()
                opt_loss.zero_grad(set_to_none=True)
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

        train_elapsed = time.time() - t0
        current_lr = sched.get_last_lr()[0]
        d = max(1, n_steps)
        t_metrics = {k: sums[k] / d for k in sums if sums[k] > 0}
        t_metrics["tok_ppl"] = math.exp(min(t_metrics.get("tok_loss", 0), 20.0))
        if "res_loss" in t_metrics:
            t_metrics["res_ppl"] = math.exp(min(t_metrics["res_loss"], 20.0))

        # --- Validation ---
        v_metrics = None
        val_elapsed = 0.0
        if ep % args.val_every == 0 or ep == args.epochs:
            t1 = time.time()
            model.eval()
            val_list = []
            for batch in val_loader:
                try:
                    vm = eval_step(batch, model, device, amp, eval_residue=eval_residue)
                    val_list.append(vm)
                except RuntimeError as e:
                    if _is_oom(e):
                        torch.cuda.empty_cache()
                        continue
                    raise
            val_elapsed = time.time() - t1
            if val_list:
                v_metrics = {k: np.mean([m[k] for m in val_list]) for k in val_list[0]}
                v_metrics["tok_ppl"] = math.exp(min(v_metrics["tok_loss"], 20.0))
                if "res_loss" in v_metrics:
                    v_metrics["res_ppl"] = math.exp(min(v_metrics["res_loss"], 20.0))
                # Compat: put tok_loss into "loss" for checkpoint best-tracking
                v_metrics["loss"] = v_metrics["tok_loss"]

        # Log
        msg = f"Epoch {ep:2d}/{args.epochs:2d} [{train_elapsed:.0f}s"
        if v_metrics:
            msg += f"+{val_elapsed:.0f}s"
        msg += f"] loss={t_metrics.get('loss',0):.4f}"
        msg += f" t_acc={t_metrics.get('tok_acc',0):.3f} t_ppl={t_metrics.get('tok_ppl',0):.1f}"
        if not token_only and "res_acc" in t_metrics:
            msg += f" | r_acc={t_metrics['res_acc']:.3f} r_ppl={t_metrics.get('res_ppl',0):.1f}"
        if not token_only and "pct_tok" in t_metrics:
            msg += f" | split={t_metrics['pct_tok']:.0%}/{t_metrics.get('pct_res',0):.0%}"
        msg += f" | lr={current_lr:.2e}"
        if v_metrics:
            msg += f" | val_loss={v_metrics['loss']:.4f} val_acc={v_metrics['tok_acc']:.3f} val_ppl={v_metrics['tok_ppl']:.1f}"
            if "res_acc" in v_metrics:
                msg += f" val_r_acc={v_metrics['res_acc']:.3f}"
        log.info(msg)

        # Save checkpoint + metrics
        ckpt_mgr.save(
            model, opt, sched, ep, global_step=global_step,
            train_metrics=t_metrics, val_metrics=v_metrics,
            lr=current_lr, train_elapsed_s=train_elapsed,
            val_elapsed_s=val_elapsed,
        )

    log.info("=" * 50)
    log.info("MLM pretraining complete. Best val loss: %.4f (epoch %d)",
             ckpt_mgr.best_val_loss, ckpt_mgr.best_epoch)
    log.info("Checkpoints: %s", ckpt_mgr.phase_dir)


if __name__ == "__main__":
    main()
    """
    nohup python -m zest.training.pretrain \
    --model deep --val-shards 1\
    --train-data data/protein/uniref90_parquet/train \
    --val-data data/protein/uniref90_parquet/val \
    --batch-size 64 --grad-accum-steps 16 \
    --shards-per-epoch 32 --unique-shards \
    --epochs 16 --lr 5e-4 --warmup-ratio 0.05 \
    --mask-prob 0.30 --trie-dropout 0.15 \
    --max-tokens 1024 --amp \
    --token-mlm-weight 0.7 --residue-mlm-weight 0.3 \
    --val-every 4 --run-dir outputs/deep_dual_mlm > outputs/deep_dual_mlm_pretrain.log 2>&1 &
    """