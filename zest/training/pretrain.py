"""
Token-level MLM pretraining for ThemeEncoder.

Usage:
    python -m zest.training.pretrain --model efficient-deep --epochs 8
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from zest.model.theme_encoder import create_theme_encoder
from zest.tokenizer import ZESTTokenizer
from zest.data.pretrain import PretrainDataset, pretrain_collate_fn
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, setup_logging,
)

GRAD_CLIP = 1.0


def _is_oom(err):
    return "out of memory" in str(err).lower()


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ThemeEncoder MLM pretraining")
    p.add_argument("--model", default="base", choices=["base", "efficient-deep", "deep-wide"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tokenizer", default="data/tokens.json")
    p.add_argument("--sequences", default="data/sequences")
    p.add_argument("--train-ids", default="data/train_ids.json")
    p.add_argument("--val-ids", default="data/val_ids.json")
    p.add_argument("--output-dir", default="outputs/models")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--checkpoint", default=None, help="Warm-start from checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            amp = True

    ckpt_mgr = CheckpointManager(args.output_dir, "theme_encoder_mlm", args.model, "mlm",
                                  run_dir=args.run_dir)
    log = setup_logging("pretrain", str(ckpt_mgr.log_path))

    tokenizer = ZESTTokenizer(args.tokenizer)
    train_ids = json.loads(open(args.train_ids).read()) if args.train_ids else None
    val_ids = json.loads(open(args.val_ids).read()) if args.val_ids else None

    train_ds = PretrainDataset(args.sequences, tokenizer,
                                is_training=True, ids=train_ids)
    val_ds = PretrainDataset(args.sequences, tokenizer,
                              is_training=False, ids=val_ids)

    collate = lambda b: pretrain_collate_fn(
        b, tokenizer.pad_id, tokenizer.mask_id, tokenizer.vocab_size, args.mask_prob)

    model = create_theme_encoder(args.model, tokenizer.vocab_size, args.max_tokens).to(device)
    if args.checkpoint:
        log.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s (%.1fM params)", args.model, total_params / 1e6)

    bs, ga = args.batch_size, args.grad_accum_steps
    nw = args.num_workers
    pf = 2 if nw > 0 else None

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    loader = DataLoader(train_ds, bs, shuffle=True, num_workers=nw,
                        pin_memory=True, collate_fn=collate, drop_last=True,
                        prefetch_factor=pf, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=nw,
                             pin_memory=True, collate_fn=collate,
                             prefetch_factor=pf, persistent_workers=nw > 0)

    steps_ep = max(1, len(loader) // ga)
    sched = build_cosine_schedule(opt, args.epochs * steps_ep,
                                   int(args.epochs * steps_ep * args.warmup_ratio))

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses, accs, n = 0.0, 0.0, 0
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {ep}", leave=False)):
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
            n += 1
            if (i + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        # Validation
        model.eval()
        vl, va, vn = 0.0, 0.0, 0
        for batch in val_loader:
            vm = eval_step(batch, model, device, amp)
            vl += vm["loss"].item()
            va += vm["acc"].item()
            vn += 1

        log.info("Epoch %d/%d loss=%.4f acc=%.4f | val_loss=%.4f val_acc=%.4f",
                 ep, args.epochs, losses / max(n, 1), accs / max(n, 1),
                 vl / max(vn, 1), va / max(vn, 1))
        ckpt_mgr.save(model, opt, sched, ep, val_loss=vl / max(vn, 1))

    log.info("=" * 50)
    log.info("MLM pretraining complete. Best val loss: %.4f (epoch %d)",
             ckpt_mgr.state.best_val_loss, ckpt_mgr.state.best_epoch)
    log.info("Checkpoints: %s", ckpt_mgr.models_dir)


if __name__ == "__main__":
    main()
