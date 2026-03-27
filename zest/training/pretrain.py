"""
Masked Profile Modeling (MPM) pretraining for TraitEncoder.

Two-phase training:
  Phase 1a (frozen): Encoder frozen, train profile head on unmasked sequences.
  Phase 1b (e2e):    Full model, 15% masking. Loss = profile KL + token MLM.

Usage:
    python -m zest.training.pretrain --model efficient-deep --frozen-epochs 2 --e2e-epochs 6
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from zest.model.trait_encoder import TraitEncoder, create_trait_encoder
from zest.tokenizer import ZESTTokenizer
from zest.data.pretrain import PretrainDataset, pretrain_collate_fn
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, build_llrd_param_groups, setup_logging,
)

GRAD_CLIP = 1.0


def kl_profile_loss(logits, targets, mask=None):
    kl = F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="none").sum(-1)
    if mask is None:
        return kl.mean()
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def _is_oom(err):
    return "out of memory" in str(err).lower()


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------

def train_step_frozen(batch, model, device, amp, ga):
    ids = batch["orig_input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    tgt = batch["targets"].to(device)
    gi = batch["global_indices"].to(device)
    tl = batch["token_lengths"].to(device)
    tm = batch["target_mask"].to(device) if batch.get("target_mask") is not None else None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        embs = m.core(ids, mask)
        logits = m.profile_head(embs, tl, gi, m.global_pos_emb)
        loss = kl_profile_loss(logits, tgt, tm)
        (loss / ga).backward()

    with torch.no_grad():
        pred = logits.argmax(-1)
        true = tgt.argmax(-1)
        valid = tm > 0 if tm is not None else torch.ones_like(true, dtype=torch.bool)
        acc = ((pred == true) & valid).sum() / valid.sum().clamp(min=1)
    return {"loss": loss.detach(), "prof_acc": acc.detach()}


def train_step_e2e(batch, model, device, amp, ga, tok_weight=0.1):
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    tgt = batch["targets"].to(device)
    gi = batch["global_indices"].to(device)
    tl = batch["token_lengths"].to(device)
    tm = batch["target_mask"].to(device) if batch.get("target_mask") is not None else None
    tok_labels = batch.get("token_labels")
    if tok_labels is not None:
        tok_labels = tok_labels.to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        prof_logits, _, tok_logits, _ = m.forward_mpm_mlm(ids, mask, gi, tl)
        loss_prof = kl_profile_loss(prof_logits, tgt, tm)
        loss_tok = (F.cross_entropy(tok_logits.view(-1, tok_logits.size(-1)),
                                     tok_labels.view(-1), ignore_index=-100)
                    if tok_labels is not None else tok_logits.new_tensor(0.0))
        loss = loss_prof + tok_weight * loss_tok
        (loss / ga).backward()

    with torch.no_grad():
        pred = prof_logits.argmax(-1)
        true = tgt.argmax(-1)
        valid = tm > 0 if tm is not None else torch.ones_like(true, dtype=torch.bool)
        prof_acc = ((pred == true) & valid).sum() / valid.sum().clamp(min=1)
        tok_acc = torch.tensor(0.0, device=device)
        if tok_labels is not None:
            tv = tok_labels != -100
            if tv.any():
                tok_acc = (tok_logits.argmax(-1)[tv] == tok_labels[tv]).float().mean()

    return {"loss": loss.detach(), "prof_loss": loss_prof.detach(),
            "tok_loss": loss_tok.detach(), "prof_acc": prof_acc.detach(),
            "tok_acc": tok_acc.detach()}


@torch.no_grad()
def eval_step(batch, model, device, amp):
    ids = batch["orig_input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    tgt = batch["targets"].to(device)
    gi = batch["global_indices"].to(device)
    tl = batch["token_lengths"].to(device)
    tm = batch["target_mask"].to(device) if batch.get("target_mask") is not None else None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        embs = m.core(ids, mask)
        logits = m.profile_head(embs, tl, gi, m.global_pos_emb)
        loss = kl_profile_loss(logits, tgt, tm)

    pred = logits.argmax(-1)
    true = tgt.argmax(-1)
    valid = tm > 0 if tm is not None else torch.ones_like(true, dtype=torch.bool)
    acc = ((pred == true) & valid).sum() / valid.sum().clamp(min=1)
    return {"loss": loss, "prof_acc": acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TraitEncoder MPM pretraining")
    p.add_argument("--model", default="base", choices=["base", "efficient-deep", "deep-wide"])
    p.add_argument("--frozen-epochs", type=int, default=2)
    p.add_argument("--e2e-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--llrd-decay", type=float, default=0.85)
    p.add_argument("--token-mlm-weight", type=float, default=0.1)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tokenizer", default="data/tokens.json")
    p.add_argument("--sequences", default="data/sequences")
    p.add_argument("--emissions", default="data/emissions")
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

    ckpt_mgr = CheckpointManager(args.output_dir, "trait_encoder_mpm", args.model, "mpm",
                                  run_dir=args.run_dir)
    log = setup_logging("pretrain", str(ckpt_mgr.log_path))

    tokenizer = ZESTTokenizer(args.tokenizer)
    train_ids = json.loads(open(args.train_ids).read()) if args.train_ids else None
    val_ids = json.loads(open(args.val_ids).read()) if args.val_ids else None

    train_ds = PretrainDataset(args.sequences, args.emissions, tokenizer,
                                is_training=True, hmm_ids=train_ids)
    val_ds = PretrainDataset(args.sequences, args.emissions, tokenizer,
                              is_training=False, hmm_ids=val_ids)

    collate_nomask = lambda b: pretrain_collate_fn(
        b, tokenizer.pad_id, tokenizer.mask_id, tokenizer.vocab_size, False, 0)
    collate_mask = lambda b: pretrain_collate_fn(
        b, tokenizer.pad_id, tokenizer.mask_id, tokenizer.vocab_size, True, args.mask_prob)

    model = create_trait_encoder(args.model, tokenizer.vocab_size, args.max_tokens).to(device)
    if args.checkpoint:
        log.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s (%.1fM params)", args.model, total_params / 1e6)

    bs, ga = args.batch_size, args.grad_accum_steps
    nw = args.num_workers
    pf = 2 if nw > 0 else None
    total_epochs = args.frozen_epochs + args.e2e_epochs

    # ---- Phase 1a: frozen encoder ----
    if args.frozen_epochs > 0:
        log.info("=" * 50)
        log.info("Phase 1a: frozen encoder (%d epochs)", args.frozen_epochs)
        model.freeze_core()
        model.freeze_for_similarity()
        for p in model.profile_head.parameters():
            p.requires_grad = True
        for p in model.global_pos_emb.parameters():
            p.requires_grad = True

        loader = DataLoader(train_ds, bs, shuffle=True, num_workers=nw,
                            pin_memory=True, collate_fn=collate_nomask, drop_last=True,
                            prefetch_factor=pf, persistent_workers=nw > 0)
        val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=nw,
                                 pin_memory=True, collate_fn=collate_nomask,
                                 prefetch_factor=pf, persistent_workers=nw > 0)

        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        steps_ep = max(1, len(loader) // ga)
        sched = build_cosine_schedule(opt, args.frozen_epochs * steps_ep,
                                       int(args.frozen_epochs * steps_ep * args.warmup_ratio))

        for ep in range(1, args.frozen_epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            losses, accs, n = 0.0, 0.0, 0
            for i, batch in enumerate(tqdm(loader, desc=f"Epoch {ep} [frozen]", leave=False)):
                try:
                    m = train_step_frozen(batch, model, device, amp, ga)
                except RuntimeError as e:
                    if _is_oom(e):
                        torch.cuda.empty_cache()
                        opt.zero_grad(set_to_none=True)
                        continue
                    raise
                losses += m["loss"].item()
                accs += m["prof_acc"].item()
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
                va += vm["prof_acc"].item()
                vn += 1

            log.info("Epoch %d/%d [FROZEN] loss=%.4f acc=%.4f | val_loss=%.4f val_acc=%.4f",
                     ep, total_epochs, losses / max(n, 1), accs / max(n, 1),
                     vl / max(vn, 1), va / max(vn, 1))
            ckpt_mgr.save(model, opt, sched, ep, val_loss=vl / max(vn, 1))

        del loader, val_loader, opt, sched
        torch.cuda.empty_cache()

    # ---- Phase 1b: end-to-end ----
    if args.e2e_epochs > 0:
        log.info("=" * 50)
        log.info("Phase 1b: end-to-end (%d epochs)", args.e2e_epochs)
        model.unfreeze_all()
        model.freeze_for_similarity()
        for m in [model.profile_head, model.mlm_head, model.token_mlm_proj,
                  model.global_pos_emb]:
            for p in m.parameters():
                p.requires_grad = True

        head_lr = args.lr * 5
        groups = build_llrd_param_groups(model, args.lr, head_lr, args.llrd_decay)
        opt = optim.AdamW(groups, lr=args.lr)

        loader = DataLoader(train_ds, bs, shuffle=True, num_workers=nw,
                            pin_memory=True, collate_fn=collate_mask, drop_last=True,
                            prefetch_factor=pf, persistent_workers=nw > 0)
        val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=nw,
                                 pin_memory=True, collate_fn=collate_nomask,
                                 prefetch_factor=pf, persistent_workers=nw > 0)

        steps_ep = max(1, len(loader) // ga)
        sched = build_cosine_schedule(opt, args.e2e_epochs * steps_ep,
                                       int(args.e2e_epochs * steps_ep * args.warmup_ratio))

        offset = args.frozen_epochs
        for ep in range(1, args.e2e_epochs + 1):
            gep = offset + ep
            model.train()
            opt.zero_grad(set_to_none=True)
            acc_m = {"loss": 0, "prof_acc": 0, "tok_acc": 0}
            n = 0
            for i, batch in enumerate(tqdm(loader, desc=f"Epoch {gep} [e2e]", leave=False)):
                try:
                    m = train_step_e2e(batch, model, device, amp, ga, args.token_mlm_weight)
                except RuntimeError as e:
                    if _is_oom(e):
                        torch.cuda.empty_cache()
                        opt.zero_grad(set_to_none=True)
                        continue
                    raise
                for k in acc_m:
                    acc_m[k] += m[k].item() if hasattr(m[k], "item") else m[k]
                n += 1
                if (i + 1) % ga == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)

            model.eval()
            vl, va, vn = 0.0, 0.0, 0
            for batch in val_loader:
                vm = eval_step(batch, model, device, amp)
                vl += vm["loss"].item()
                va += vm["prof_acc"].item()
                vn += 1

            log.info("Epoch %d/%d [E2E] loss=%.4f prof_acc=%.4f tok_acc=%.4f | "
                     "val_loss=%.4f val_acc=%.4f",
                     gep, total_epochs, acc_m["loss"] / max(n, 1),
                     acc_m["prof_acc"] / max(n, 1), acc_m["tok_acc"] / max(n, 1),
                     vl / max(vn, 1), va / max(vn, 1))
            ckpt_mgr.save(model, opt, sched, gep, val_loss=vl / max(vn, 1))

    log.info("=" * 50)
    log.info("MPM pretraining complete. Best val loss: %.4f (epoch %d)",
             ckpt_mgr.state.best_val_loss, ckpt_mgr.state.best_epoch)
    log.info("Checkpoints: %s", ckpt_mgr.models_dir)


if __name__ == "__main__":
    main()
