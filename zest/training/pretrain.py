"""
Token-level MLM pretraining for ThemeEncoder.

Usage:
    python -m zest.training.pretrain --model base --epochs 8
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from zest.model.theme_encoder import create_theme_encoder
from zest.tokenizer import ZESTTokenizer
from zest.data.pretrain import MLMDataset, mlm_collate_fn
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, setup_logging,
    make_run_dir, save_environment, save_model_arch,
)

GRAD_CLIP = 1.0


def _is_oom(err):
    return "out of memory" in str(err).lower()


def train_step(batch, model, device, amp, ga):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["token_labels"].to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(ids, mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               labels.view(-1), ignore_index=-100)
        (loss / ga).backward()

    with torch.no_grad():
        valid = labels != -100
        acc = torch.tensor(0.0, device=device)
        if valid.any():
            acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean()

    return {"loss": loss.detach(), "acc": acc.detach()}


@torch.no_grad()
def eval_step(batch, model, device, amp):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["token_labels"].to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(ids, mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               labels.view(-1), ignore_index=-100)

    valid = labels != -100
    acc = torch.tensor(0.0, device=device)
    if valid.any():
        acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean()
    return {"loss": loss, "acc": acc}


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
    
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    if args.checkpoint:
        run_dir = Path(args.checkpoint).parent.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = make_run_dir(args.output_dir, args.model)

    keep_n = 999 if args.keep_all_checkpoints else 3
    ckpt_mgr = CheckpointManager(run_dir, phase="pretrain", keep_last_n=keep_n)
    log = setup_logging("pretrain", str(ckpt_mgr.log_path))

    tokenizer = ZESTTokenizer(args.tokenizer)

    train_ds = MLMDataset(
        args.train_data, tokenizer,
        is_training=True,
        max_tokens=args.max_tokens, trie_dropout=args.trie_dropout,
        shards_per_epoch=args.shards_per_epoch,
        unique_shards=args.unique_shards, seed=args.seed,
    )
    val_ds = MLMDataset(
        args.val_data, tokenizer,
        is_training=False,
        max_tokens=args.max_tokens,
    )

    collate = lambda b: mlm_collate_fn(
        b, tokenizer.pad_id, tokenizer.mask_id, tokenizer.vocab_size,
        mask_prob=args.mask_prob,
    )

    model = create_theme_encoder(args.model, tokenizer.vocab_size, args.max_tokens).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s (%.1fM params)", args.model, total_params / 1e6)

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

    opt = optim.AdamW(model.parameters(), lr=args.lr)

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
    }
    with open(ckpt_mgr.run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    save_environment(ckpt_mgr.run_dir)
    save_model_arch(ckpt_mgr.run_dir, model)
    log.info("Run directory: %s", ckpt_mgr.run_dir)

    # Resume from checkpoint
    start_epoch = 1
    if args.checkpoint:
        ckpt = ckpt_mgr.load(args.checkpoint, model, opt, sched, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info("Resumed from %s (epoch %d, best_val=%.4f)",
                 args.checkpoint, start_epoch - 1, ckpt_mgr.best_val_loss)

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
        losses, accs = 0.0, 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep}", leave=False)
        for i, batch in enumerate(pbar):
            try:
                m = train_step(batch, model, device, amp, ga)
            
            except RuntimeError as e:
                if _is_oom(e):
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    continue
                raise
            
            losses += m["loss"].item()
            accs += m["acc"].item()
            pbar.set_postfix(loss=f"{losses/(i+1):.3f}", acc=f"{accs/(i+1):.1%}")
            
            if (i + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

        train_elapsed = time.time() - t0
        current_lr = sched.get_last_lr()[0]
        train_loss = losses / max(1, len(loader))
        train_acc = accs / max(1, len(loader))
        t_metrics = {"loss": train_loss, "acc": train_acc}

        # --- Validation ---
        v_metrics = None
        val_elapsed = 0.0
        if ep % args.val_every == 0 or ep == args.epochs:
            t1 = time.time()
            model.eval()
            vl, va = 0.0, 0.0
            for batch in val_loader:
                vm = eval_step(batch, model, device, amp)
                vl += vm["loss"].item()
                va += vm["acc"].item()
            val_elapsed = time.time() - t1
            v_metrics = {
                "loss": vl / max(1, len(val_loader)),
                "acc": va / max(1, len(val_loader)),
            }

        # Log
        msg = f"Epoch {ep:2d}/{args.epochs:2d} [train {train_elapsed:.0f}s"
        if v_metrics:
            msg += f" | val {val_elapsed:.0f}s"
        msg += f"] loss={train_loss:.4f} acc={train_acc:.4f}"
        if v_metrics:
            msg += f" | val_loss={v_metrics['loss']:.4f} val_acc={v_metrics['acc']:.4f}"
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
