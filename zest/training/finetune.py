"""
Hierarchical contrastive fine-tuning for ThemeEncoder.

Supports two domains:
  - **protein** — 3-level CATH hierarchy (Architecture / Topology / Superfamily)
  - **code** — single-level algorithm class (POJ-104 clone detection)

Loss functions live in ``zest.training.losses``; encoding helpers and
train/eval step wrappers live in ``zest.training.steps``.

Usage:
    python -m zest.training.finetune --checkpoint path/to/pretrain.pth --epochs 8
    python -m zest.training.finetune --domain code --checkpoint path/to/pretrain.pth
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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from zest.model.theme_encoder import ThemeEncoder
from zest.tokenizer import ZESTTokenizer
from zest.data.finetune import HierarchicalPairDataset, finetune_collate_fn
from zest.training.losses import MemoryBank
from zest.training.steps import train_step, eval_step, get_metric_keys
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, build_llrd_param_groups, setup_logging,
    save_environment, save_model_arch,
)


def _is_oom(err):
    return "out of memory" in str(err).lower()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def _seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical contrastive fine-tuning")
    p.add_argument("--checkpoint",       required=True,                help="Pretrained model checkpoint to fine-tune from")
    p.add_argument("--resume",           default=None,                 help="Resume fine-tuning from a finetune checkpoint")
    p.add_argument("--tokenizer",        default="data/tokens.json",   help="Tokenizer file path")
    p.add_argument("--shards-dir",       default="data/pair_shards/train", help="Training shards directory")
    p.add_argument("--val-shards-dir",   default="data/pair_shards/val",   help="Validation shards directory")
    p.add_argument("--shards-per-epoch", type=int, default=None,       help="Parquet shards to load per epoch (streaming mode)")
    p.add_argument("--run-dir",          default=None,                 help="Run directory (inferred from --checkpoint if given)")

    p.add_argument("--epochs",           type=int,   default=8,        help="Number of training epochs")
    p.add_argument("--batch-size",       type=int,   default=128,      help="Batch size per GPU")
    p.add_argument("--grad-accum-steps", type=int,   default=1,        help="Gradient accumulation steps")
    p.add_argument("--lr",               type=float, default=5e-4,     help="Learning rate")
    p.add_argument("--encoder-lr-scale", type=float, default=0.1,      help="Encoder LR multiplier")
    p.add_argument("--warmup-ratio",     type=float, default=0.05,     help="Warmup ratio")
    p.add_argument("--grad-clip",        type=float, default=1.0,      help="Gradient clipping norm")
    p.add_argument("--weight-decay",     type=float, default=0.01,     help="Weight decay")
    p.add_argument("--llrd-decay",       type=float, default=0.95,     help="Layer-wise LR decay")
    p.add_argument("--max-tokens",       type=int,   default=512,      help="Maximum sequence length")
    p.add_argument("--num-workers",      type=int,   default=6,        help="Number of data loading workers")
    p.add_argument("--bpe-dropout",      type=float, default=0.3,      help="BPE dropout probability")
    p.add_argument("--seed",             type=int,   default=42,       help="Random seed for reproducibility")
    p.add_argument("--val-every",        type=int,   default=1,        help="Run validation every N epochs")
    p.add_argument("--val-max-batches",  type=int,   default=200,      help="Max validation batches per eval (0=all)")
    p.add_argument("--amp",              action="store_true",          help="Use automatic mixed precision")
    p.add_argument("--keep-all-checkpoints", action="store_true",      help="Keep all checkpoints (default: last 3)")

    p.add_argument("--unfreeze-layers",  type=int, default=4,
                   help="Encoder layers to unfreeze from top (-1 = all)")
    p.add_argument("--unfreeze-embeddings", action="store_true")

    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument("--cross-level-weight", type=float, default=0.0,
                   help="Weight for cross-level InfoNCE (all pairs, T+A only)")
    p.add_argument("--smooth-ap-weight", type=float, default=0.0)
    p.add_argument("--smooth-ap-temp",   type=float, default=0.01)
    p.add_argument("--rank-weight",      type=float, default=0.0)
    p.add_argument("--rank-margin",      type=float, default=1.0)
    p.add_argument("--rank-level-weights", type=float, nargs=3, default=[0.2, 0.3, 0.5])

    p.add_argument("--temp-h",           type=float, default=0.05)
    p.add_argument("--temp-t",           type=float, default=0.07)
    p.add_argument("--temp-a",           type=float, default=0.10)
    p.add_argument("--weight-h",         type=float, default=0.50)
    p.add_argument("--weight-t",         type=float, default=0.35)
    p.add_argument("--weight-a",         type=float, default=0.15)

    p.add_argument("--memory-bank-size", type=int,   default=0)
    p.add_argument("--memory-bank-samples", type=int, default=4096)
    p.add_argument("--hard-neg-weight",  type=float, default=1.0)
    p.add_argument("--smooth-ap-max-samples", type=int, default=256)

    p.add_argument("--num-proj-layers",  type=int, default=None)
    p.add_argument("--proj-dim",         type=int, default=None)
    p.add_argument("--proj-ff-mult",     type=int, default=None)
    p.add_argument("--domain",           default="protein", choices=["protein", "code"], help="Data domain")
    p.add_argument("--ast-language",     default="cpp",              help="Language for AST encoder (code domain only)")
    return p.parse_args()


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

    # Determine run directory
    if args.resume:
        run_dir = Path(args.resume).parent.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(args.checkpoint).parent.parent

    keep_n = 999 if args.keep_all_checkpoints else 3
    ckpt_mgr = CheckpointManager(run_dir, phase="finetune", keep_last_n=keep_n)
    log = setup_logging("finetune", str(ckpt_mgr.log_path))

    # Load model
    overrides = {}
    if args.num_proj_layers is not None:
        overrides["num_proj_layers"] = args.num_proj_layers
    if args.proj_dim is not None:
        overrides["proj_dim"] = args.proj_dim
    if args.proj_ff_mult is not None:
        overrides["proj_ff_mult"] = args.proj_ff_mult

    model, _ = ThemeEncoder.load_pretrained(args.checkpoint, device="cpu", **overrides)
    model = model.to(device)

    # Start fully frozen, then selectively unfreeze
    for p in model.parameters():
        p.requires_grad = False
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_layers = m.core.num_layers

    if args.unfreeze_layers == -1 or args.unfreeze_layers >= n_layers:
        for p in m.core.parameters():
            p.requires_grad = True
    elif args.unfreeze_layers > 0:
        for layer in m.core.layers[-args.unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in m.core.final_norm.parameters():
            p.requires_grad = True

    if args.unfreeze_embeddings:
        for p in m.core.token_emb.parameters():
            p.requires_grad = True
        for p in m.core.emb_dropout.parameters():
            p.requires_grad = True

    # Always train these
    for module in [m.aggregator, m.projector]:
        for p in module.parameters():
            p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %.1fM total, %.1fM trainable (%.1f%%)",
             total_params / 1e6, trainable / 1e6, 100 * trainable / total_params)

    # Tokenizer + data
    log.info("Loading tokenizer and training data from %s ...", args.shards_dir)
    tokenizer = ZESTTokenizer(args.tokenizer)

    # Build domain-specific encoder for code
    encoder = None
    encoder_cfg = None
    if args.domain == "code":
        from experiments.code.build_role_tokenizer import make_role_encoder
        encoder = make_role_encoder(args.ast_language)
        encoder_cfg = {"language": args.ast_language, "dropout": args.bpe_dropout}
        log.info("Code domain: role encoder for %s (dropout=%.2f)",
                 args.ast_language, args.bpe_dropout)

    train_ds = HierarchicalPairDataset(
        args.shards_dir, tokenizer, max_tokens=args.max_tokens,
        shards_per_epoch=args.shards_per_epoch, bpe_dropout=args.bpe_dropout,
        encoder=encoder, encoder_cfg=encoder_cfg,
    )
    val_ds = None
    collate = lambda b: finetune_collate_fn(b, tokenizer.pad_id)

    bs, ga = args.batch_size, args.grad_accum_steps
    nw = args.num_workers
    pf = 6 if nw > 0 else None
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)

    def make_loader(ds, shuffle=True, drop_last=True):
        return DataLoader(
            ds, bs, shuffle=shuffle,
            num_workers=nw,
            pin_memory=True, collate_fn=collate, drop_last=drop_last,
            worker_init_fn=_seed_worker, generator=loader_gen,
            prefetch_factor=pf if nw > 0 else None,
        )

    # Optimizer
    encoder_lr = args.lr * args.encoder_lr_scale
    if args.llrd_decay < 1.0 and args.unfreeze_layers != 0:
        groups = build_llrd_param_groups(model, encoder_lr, args.lr, args.llrd_decay)
    else:
        groups = [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith("core")],
             "lr": args.lr},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith("core")],
             "lr": encoder_lr},
        ]
    opt = optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    # Memory bank
    proj_dim = m.proj_dim
    memory_bank = MemoryBank(args.memory_bank_size, proj_dim, device) if args.memory_bank_size > 0 else None

    loader = make_loader(train_ds)
    val_loader = None

    steps_ep = max(1, len(loader) // ga)
    sched = build_cosine_schedule(
        opt, args.epochs * steps_ep,
        int(args.epochs * steps_ep * args.warmup_ratio),
    )

    # Save run metadata
    config = {
        "phase": "finetune",
        "domain": args.domain,
        "pretrained_checkpoint": str(args.checkpoint),
        "params_M": round(total_params / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "epochs": args.epochs,
        "batch_size": bs,
        "grad_accum_steps": ga,
        "effective_batch": bs * ga,
        "lr": args.lr,
        "encoder_lr": encoder_lr,
        "warmup_ratio": args.warmup_ratio,
        "scheduler": "cosine",
        "total_steps": args.epochs * steps_ep,
        "warmup_steps": int(args.epochs * steps_ep * args.warmup_ratio),
        "steps_per_epoch": steps_ep,
        "max_tokens": args.max_tokens,
        "unfreeze_layers": args.unfreeze_layers,
        "contrastive_weight": args.contrastive_weight,
        "cross_level_weight": args.cross_level_weight,
        "smooth_ap_weight": args.smooth_ap_weight,
        "rank_weight": args.rank_weight,
        "memory_bank_size": args.memory_bank_size,
        "hard_neg_weight": args.hard_neg_weight,
        "seed": args.seed,
        "amp": amp,
    }
    with open(ckpt_mgr.run_dir / "config_finetune.json", "w") as f:
        json.dump(config, f, indent=2)
    save_environment(ckpt_mgr.run_dir)
    save_model_arch(ckpt_mgr.run_dir, model)
    log.info("Run directory: %s", ckpt_mgr.run_dir)

    # Resume from finetune checkpoint
    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt = ckpt_mgr.load(args.resume, model, opt, sched, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        log.info("Resumed from %s (epoch %d, best_val=%.4f)",
                 args.resume, start_epoch - 1, ckpt_mgr.best_val_loss)

    log.info("Starting training: %d epochs, BS=%d, GA=%d, LR=%.2e (encoder=%.2e)",
             args.epochs, bs, ga, args.lr, encoder_lr)

    METRIC_KEYS = get_metric_keys(args.domain)

    core_fully_frozen = not any(p.requires_grad for p in m.core.parameters())

    for ep in range(start_epoch, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)
        # --- Train ---
        t0 = time.time()
        model.train()
        if core_fully_frozen:
            m.core.eval()
        opt.zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in METRIC_KEYS}
        n_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {ep}", leave=False)

        for i, batch in enumerate(pbar):
            try:
                ms = train_step(batch, model, device, amp, args, ga, memory_bank)
            except RuntimeError as e:
                if _is_oom(e):
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    continue
                raise

            step_loss = ms["loss"].item()
            if math.isnan(step_loss):
                opt.zero_grad(set_to_none=True)
                continue

            for k in METRIC_KEYS:
                acc[k] += ms[k].item()
            n_steps += 1
            if args.domain == "code":
                pbar.set_postfix(
                    loss=f"{acc['loss']/n_steps:.3f}",
                    clone_acc=f"{acc['clone_acc']/n_steps:.1%}",
                    map_r=f"{acc['map_at_r']/n_steps:.3f}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{acc['loss']/n_steps:.3f}",
                    acc_h=f"{acc['acc_h']/n_steps:.1%}",
                )

            if (i + 1) % ga == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

        train_elapsed = time.time() - t0
        t_metrics = {k: v / max(1, n_steps) for k, v in acc.items()}
        log.info("Epoch %d training done in %.0fs (%d steps)", ep, train_elapsed, n_steps)

        # --- Validation ---
        v_metrics = None
        val_elapsed = 0.0
        if args.val_shards_dir and (ep % args.val_every == 0 or ep == args.epochs):
            if val_loader is None:
                log.info("Loading validation dataset from %s ...", args.val_shards_dir)
                val_ds = HierarchicalPairDataset(
                    args.val_shards_dir, tokenizer, max_tokens=args.max_tokens,
                    encoder=encoder, encoder_cfg=encoder_cfg,
                )
                val_loader = make_loader(val_ds, shuffle=False, drop_last=False)
            max_vb = args.val_max_batches if args.val_max_batches > 0 else None
            log.info("Running validation (max_batches=%s) ...", max_vb or "all")
            t1 = time.time()
            model.eval()
            v_acc = {k: 0.0 for k in METRIC_KEYS}
            vn = 0
            for batch in val_loader:
                try:
                    vm = eval_step(batch, model, device, amp, args)
                except RuntimeError as e:
                    if _is_oom(e):
                        torch.cuda.empty_cache()
                        continue
                    raise
                for k in METRIC_KEYS:
                    v_acc[k] += vm[k].item()
                vn += 1
                if max_vb and vn >= max_vb:
                    break
            val_elapsed = time.time() - t1
            v_metrics = {k: v / max(1, vn) for k, v in v_acc.items()}

        # Log
        current_lr = sched.get_last_lr()[0]
        msg = f"Epoch {ep:2d}/{args.epochs:2d} [train {train_elapsed:.0f}s"
        if v_metrics:
            msg += f" | val {val_elapsed:.0f}s"
        if args.domain == "code":
            msg += (f"] loss={t_metrics['loss']:.4f} con={t_metrics['con']:.4f}"
                    f" sap={t_metrics['sap']:.4f}"
                    f" clone_acc={t_metrics['clone_acc']:.4f}"
                    f" MAP@R={t_metrics['map_at_r']:.4f}")
            if v_metrics:
                msg += (f" | val_loss={v_metrics['loss']:.4f}"
                        f" val_clone_acc={v_metrics['clone_acc']:.4f}"
                        f" val_MAP@R={v_metrics['map_at_r']:.4f}")
        else:
            msg += (f"] loss={t_metrics['loss']:.4f} con={t_metrics['con']:.4f}"
                    f" cross={t_metrics['cross']:.4f} sap={t_metrics['sap']:.4f}"
                    f" rank={t_metrics['rank']:.4f} acc_h={t_metrics['acc_h']:.4f}")
            if v_metrics:
                msg += (f" | val_loss={v_metrics['loss']:.4f}"
                        f" val_acc_h={v_metrics['acc_h']:.4f}")
        log.info(msg)

        # Save checkpoint + metrics
        ckpt_mgr.save(
            model, opt, sched, ep, global_step=global_step,
            train_metrics=t_metrics, val_metrics=v_metrics,
            lr=current_lr, train_elapsed_s=train_elapsed,
            val_elapsed_s=val_elapsed,
        )

    log.info("=" * 50)
    log.info("Fine-tuning complete. Best val loss: %.4f (epoch %d)",
             ckpt_mgr.best_val_loss, ckpt_mgr.best_epoch)
    log.info("Checkpoints: %s", ckpt_mgr.phase_dir)


if __name__ == "__main__":
    main()
