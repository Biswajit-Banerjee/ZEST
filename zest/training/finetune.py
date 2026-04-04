"""
Hierarchical contrastive fine-tuning for TraitEncoder.

Trains a pretrained TraitEncoder model on CATH structural pairs using three
complementary losses at Architecture (A), Topology (T), and Homologous
Superfamily (H) levels:

  1. Multi-level InfoNCE — contrastive loss with per-level temperatures
  2. SmoothAP — differentiable Average Precision for direct mAP optimisation
  3. Margin ranking — enforces cosine ordering proportional to HHblits probability

A cross-batch memory bank provides additional negatives for InfoNCE.

Usage:
    python -m zest.training.finetune --checkpoint path/to/mpm.pth --epochs 8
"""
import argparse
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from zest.model.trait_encoder import TraitEncoder
from zest.tokenizer import ZESTTokenizer
from zest.data.finetune import HierarchicalPairDataset, finetune_collate_fn
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, build_llrd_param_groups, setup_logging,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-batch memory bank
# ---------------------------------------------------------------------------

class MemoryBank:
    """FIFO queue of detached embeddings + labels for extra InfoNCE negatives."""

    def __init__(self, capacity: int, embed_dim: int, device: torch.device):
        self.capacity = capacity
        self.embeddings = torch.zeros(capacity, embed_dim, device=device)
        self.a_labels = torch.zeros(capacity, dtype=torch.long, device=device)
        self.t_labels = torch.zeros(capacity, dtype=torch.long, device=device)
        self.h_labels = torch.zeros(capacity, dtype=torch.long, device=device)
        self._ptr = 0
        self._full = False

    def enqueue(self, emb, a, t, h):
        n = emb.shape[0]
        if n == 0:
            return
        emb = emb.detach()
        if n >= self.capacity:
            emb, a, t, h = emb[-self.capacity:], a[-self.capacity:], t[-self.capacity:], h[-self.capacity:]
            n = self.capacity
        end = self._ptr + n
        if end <= self.capacity:
            self.embeddings[self._ptr:end] = emb
            self.a_labels[self._ptr:end] = a
            self.t_labels[self._ptr:end] = t
            self.h_labels[self._ptr:end] = h
        else:
            first = self.capacity - self._ptr
            self.embeddings[self._ptr:] = emb[:first]
            self.a_labels[self._ptr:] = a[:first]
            self.t_labels[self._ptr:] = t[:first]
            self.h_labels[self._ptr:] = h[:first]
            rest = n - first
            self.embeddings[:rest] = emb[first:]
            self.a_labels[:rest] = a[first:]
            self.t_labels[:rest] = t[first:]
            self.h_labels[:rest] = h[first:]
        self._ptr = end % self.capacity
        if end >= self.capacity:
            self._full = True

    def get(self):
        size = self.capacity if self._full else self._ptr
        if size == 0:
            return None
        return (self.embeddings[:size], self.a_labels[:size],
                self.t_labels[:size], self.h_labels[:size])


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def hierarchical_infonce(embeddings, a_labels, t_labels, h_labels,
                         temp_a=0.10, temp_t=0.07, temp_h=0.05,
                         weight_a=0.15, weight_t=0.35, weight_h=0.50,
                         mem_emb=None, mem_a=None, mem_t=None, mem_h=None,
                         hard_neg_weight=1.0):
    """
    Multi-level InfoNCE with optional memory bank and hard negative weighting.

    Hard negatives: entries sharing a coarser label but differing at the current
    level (e.g. same topology, different superfamily) receive upweighted
    contribution to the denominator.
    """
    N = embeddings.shape[0]
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if mem_emb is not None and mem_emb.shape[0] > 0:
        keys = torch.cat([embeddings, mem_emb.to(device)])
        key_a = torch.cat([a_labels, mem_a.to(device)])
        key_t = torch.cat([t_labels, mem_t.to(device)])
        key_h = torch.cat([h_labels, mem_h.to(device)])
    else:
        keys, key_a, key_t, key_h = embeddings, a_labels, t_labels, h_labels
    K = keys.shape[0]

    sim = embeddings @ keys.T
    self_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
    self_mask[:N, :N] = torch.eye(N, dtype=torch.bool, device=device)

    results = {}
    total = zero
    sim_ns = sim.masked_fill(self_mask, -1e9)
    top1 = sim_ns.argmax(dim=1)

    for name, ql, kl, temp, w, coarse_q, coarse_k in [
        ("a", a_labels, key_a, temp_a, weight_a, None, None),
        ("t", t_labels, key_t, temp_t, weight_t, a_labels, key_a),
        ("h", h_labels, key_h, temp_h, weight_h, t_labels, key_t),
    ]:
        pos = (ql.unsqueeze(1) == kl.unsqueeze(0)) & ~self_mask
        has_pos = pos.any(dim=1)
        if not has_pos.any():
            results[f"loss_{name}"] = zero
            results[f"acc_{name}"] = zero
            continue

        logits = (sim / temp).masked_fill(self_mask, -1e9)
        exp = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        exp = exp.masked_fill(self_mask, 0.0)

        if hard_neg_weight > 1.0 and coarse_q is not None:
            hard = ((coarse_q.unsqueeze(1) == coarse_k.unsqueeze(0))
                    & (ql.unsqueeze(1) != kl.unsqueeze(0)) & ~self_mask)
            exp = exp * torch.where(hard, hard_neg_weight, 1.0)

        pos_exp = (exp * pos.float()).sum(1)
        loss_i = -torch.log(pos_exp / (exp.sum(1) + 1e-8) + 1e-8)
        level_loss = loss_i[has_pos].mean()
        acc = pos.gather(1, top1.unsqueeze(1)).squeeze(1)[has_pos].float().mean()

        results[f"loss_{name}"] = level_loss
        results[f"acc_{name}"] = acc
        total = total + w * level_loss

    results["loss"] = total
    return results


def hierarchical_smooth_ap(embeddings, a_labels, t_labels, h_labels,
                           temp=0.01, weight_a=0.15, weight_t=0.35, weight_h=0.50):
    """
    Differentiable Average Precision (SmoothAP) at each hierarchy level.

    Uses sigmoid approximation of the step function so gradients flow
    through the ranking operation.
    """
    N = embeddings.shape[0]
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if N > 512:
        idx = torch.randperm(N, device=device)[:512]
        embeddings = embeddings[idx]
        a_labels, t_labels, h_labels = a_labels[idx], t_labels[idx], h_labels[idx]
        N = 512

    sim = embeddings @ embeddings.T
    eye = torch.eye(N, dtype=torch.bool, device=device)
    not_self = (~eye).float().unsqueeze(1)
    not_j = (~eye).float().unsqueeze(0)
    diff = sim.unsqueeze(1) - sim.unsqueeze(2)
    soft = torch.sigmoid(diff / temp) * not_self * not_j

    results = {}
    total = zero
    for name, labels, w in [("a", a_labels, weight_a), ("t", t_labels, weight_t),
                             ("h", h_labels, weight_h)]:
        if w == 0:
            results[f"sap_{name}"] = zero
            continue
        pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
        has = pos.any(dim=1)
        if not has.any():
            results[f"sap_{name}"] = zero
            continue
        d_pos = 1.0 + (soft * pos.float().unsqueeze(1)).sum(2)
        d_all = 1.0 + soft.sum(2)
        prec = d_pos / (d_all + 1e-8)
        ap = (prec * pos.float()).sum(1) / pos.float().sum(1).clamp(min=1)
        level_loss = 1.0 - ap[has].mean()
        results[f"sap_{name}"] = level_loss
        total = total + w * level_loss

    results["sap_loss"] = total
    return results


def hierarchical_margin_ranking(anc_proj, pos_proj,
                                a_anc, t_anc, h_anc, a_pos, t_pos, h_pos,
                                probabilities, margin=1.0, min_neighbors=3,
                                level_weights=(0.2, 0.3, 0.5)):
    """
    Multi-level margin ranking loss enforcing cosine ordering proportional
    to HHblits probability differences within each hierarchy group.
    """
    pair_cos = (anc_proj * pos_proj).sum(-1)
    device = anc_proj.device
    zero = anc_proj.new_zeros(())

    shared_a = torch.where(a_anc == a_pos, a_anc, torch.tensor(-1, device=device))
    shared_t = torch.where(t_anc == t_pos, t_anc, torch.tensor(-1, device=device))
    shared_h = torch.where(h_anc == h_pos, h_anc, torch.tensor(-1, device=device))

    results = {}
    total = zero
    for name, shared, w in [("A", shared_a, level_weights[0]),
                             ("T", shared_t, level_weights[1]),
                             ("H", shared_h, level_weights[2])]:
        valid = shared != -1
        if not valid.any():
            results[f"rank_loss_{name}"] = zero
            continue

        vl, vc, vp = shared[valid], pair_cos[valid], probabilities[valid]
        sl, perm = torch.sort(vl)
        sc, sp = vc[perm], vp[perm]
        _, counts = torch.unique_consecutive(sl, return_counts=True)
        active = counts >= min_neighbors
        if not active.any():
            results[f"rank_loss_{name}"] = zero
            continue

        gids = torch.repeat_interleave(torch.arange(counts.shape[0], device=device), counts)
        amask = active[gids]
        sc, sp = sc[amask], sp[amask]
        gids = (active.long().cumsum(0) - 1)[gids[amask]]
        c = counts[active]
        offsets = torch.cat([c.new_zeros(1), c.cumsum(0)[:-1]])
        pos_in = torch.arange(sc.shape[0], device=device) - offsets[gids]
        mg = int(c.max().item())

        g_cos = pair_cos.new_zeros(c.shape[0], mg)
        g_prob = probabilities.new_zeros(c.shape[0], mg)
        g_valid = torch.zeros(c.shape[0], mg, dtype=torch.bool, device=device)
        g_cos[gids, pos_in] = sc
        g_prob[gids, pos_in] = sp
        g_valid[gids, pos_in] = True

        pdiff = g_prob.unsqueeze(1) - g_prob.unsqueeze(2)
        cdiff = g_cos.unsqueeze(1) - g_cos.unsqueeze(2)
        pmask = g_valid.unsqueeze(1) & g_valid.unsqueeze(2) & (pdiff > 1e-6)
        if not pmask.any():
            results[f"rank_loss_{name}"] = zero
            continue

        ploss = F.relu(margin * pdiff - cdiff) * pmask.float()
        pc = pmask.sum(dim=(1, 2))
        act = pc > 0
        if not act.any():
            results[f"rank_loss_{name}"] = zero
            continue

        gl = ploss.sum(dim=(1, 2)) / pc.clamp(min=1)
        level_loss = gl[act].mean()
        results[f"rank_loss_{name}"] = level_loss
        total = total + w * level_loss

    results["rank_loss"] = total
    return results


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_batch(model, tokens, mask):
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    embs = m._encode_tokens(tokens, mask)
    family = m.aggregator(embs, mask)
    proj = F.normalize(m.projector(family), p=2, dim=-1)
    return proj


def _sparse_mlm_loss(model, token_embs, token_labels):
    """Token-level MLM on masked positions only."""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    zero = token_embs.new_zeros(())
    if token_labels is None:
        return zero, zero
    masked = token_labels != -100
    if not masked.any():
        return zero, zero
    h = m.token_mlm_proj(token_embs)
    logits = F.linear(h, m.core.token_emb.weight)
    loss = F.cross_entropy(logits[masked], token_labels[masked])
    acc = (logits[masked].argmax(-1) == token_labels[masked]).float().mean()
    return loss, acc


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def train_step(model, batch, optimizer, scheduler, scaler, device,
               grad_clip, args, grad_accum_steps, micro_step, memory_bank):
    is_accum = (micro_step + 1) % grad_accum_steps != 0

    try:
        anc_tok = batch["anchor_ids"].to(device)
        anc_mask = batch["anchor_mask"].to(device)
        pos_tok = batch["positive_ids"].to(device)
        pos_mask = batch["positive_mask"].to(device)
        a_anc = batch["a_labels"].to(device)
        t_anc = batch["t_labels"].to(device)
        h_anc = batch["h_labels"].to(device)
        # For simplicity, anchor and positive share the same CATH labels
        a_pos, t_pos, h_pos = a_anc, t_anc, h_anc

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            anc_proj = _encode_batch(model, anc_tok, anc_mask)
            pos_proj = _encode_batch(model, pos_tok, pos_mask)

            all_emb = torch.cat([anc_proj, pos_proj])
            all_a = torch.cat([a_anc, a_pos])
            all_t = torch.cat([t_anc, t_pos])
            all_h = torch.cat([h_anc, h_pos])

            # Filter to same-SF pairs for InfoNCE
            same_sf = h_anc == h_pos
            mem = memory_bank.get() if memory_bank else None

            if same_sf.sum() > 0:
                sf_anc = anc_proj[same_sf]
                sf_pos = pos_proj[same_sf]
                sf_a, sf_t, sf_h = a_anc[same_sf], t_anc[same_sf], h_anc[same_sf]
                nce_emb = torch.cat([sf_anc, sf_pos])
                nce_a = torch.cat([sf_a, sf_a])
                nce_t = torch.cat([sf_t, sf_t])
                nce_h = torch.cat([sf_h, sf_h])

                infonce = hierarchical_infonce(
                    nce_emb, nce_a, nce_t, nce_h,
                    temp_a=args.temp_a, temp_t=args.temp_t, temp_h=args.temp_h,
                    weight_a=args.weight_a, weight_t=args.weight_t, weight_h=args.weight_h,
                    mem_emb=mem[0] if mem else None,
                    mem_a=mem[1] if mem else None,
                    mem_t=mem[2] if mem else None,
                    mem_h=mem[3] if mem else None,
                    hard_neg_weight=args.hard_neg_weight,
                )
            else:
                infonce = {"loss": torch.tensor(0.0, device=device)}

            # SmoothAP
            sap = {"sap_loss": torch.tensor(0.0, device=device)}
            if args.smooth_ap_weight > 0:
                sap = hierarchical_smooth_ap(
                    all_emb, all_a, all_t, all_h,
                    temp=args.smooth_ap_temp,
                    weight_a=args.weight_a, weight_t=args.weight_t, weight_h=args.weight_h,
                )

            # Margin ranking
            rank = {"rank_loss": torch.tensor(0.0, device=device)}
            if args.rank_weight > 0 and "probabilities" in batch:
                probs = batch["probabilities"].to(device)
                rank = hierarchical_margin_ranking(
                    anc_proj, pos_proj, a_anc, t_anc, h_anc, a_pos, t_pos, h_pos,
                    probs, margin=args.rank_margin,
                    level_weights=tuple(args.rank_level_weights),
                )

            loss = (args.contrastive_weight * infonce["loss"]
                    + args.smooth_ap_weight * sap["sap_loss"]
                    + args.rank_weight * rank["rank_loss"])
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if memory_bank:
            memory_bank.enqueue(all_emb, all_a, all_t, all_h)

        if not is_accum:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        return {
            "loss": (loss * grad_accum_steps).item(),
            "con": infonce["loss"].item(),
            "sap": sap["sap_loss"].item(),
            "rank": rank["rank_loss"].item(),
            "acc_h": infonce.get("acc_h", torch.tensor(0)).item(),
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _log.warning("OOM — skipping batch")
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            return None
        raise


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical contrastive fine-tuning")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="data/tokens.json")
    p.add_argument("--shards-dir", default="data/pair_shards/train")
    p.add_argument("--val-shards-dir", default="data/pair_shards/val")
    p.add_argument("--shards-per-epoch", type=int, default=None)
    p.add_argument("--output-dir", default="outputs/models")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--llrd-decay", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--bpe-dropout", type=float, default=0.0)

    p.add_argument("--unfreeze-layers", type=int, default=4,
                   help="Encoder layers to unfreeze from top (-1 = all)")
    p.add_argument("--unfreeze-embeddings", action="store_true")

    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument("--smooth-ap-weight", type=float, default=0.0)
    p.add_argument("--smooth-ap-temp", type=float, default=0.01)
    p.add_argument("--rank-weight", type=float, default=0.0)
    p.add_argument("--rank-margin", type=float, default=1.0)
    p.add_argument("--rank-level-weights", type=float, nargs=3, default=[0.2, 0.3, 0.5])

    p.add_argument("--temp-h", type=float, default=0.05)
    p.add_argument("--temp-t", type=float, default=0.07)
    p.add_argument("--temp-a", type=float, default=0.10)
    p.add_argument("--weight-h", type=float, default=0.50)
    p.add_argument("--weight-t", type=float, default=0.35)
    p.add_argument("--weight-a", type=float, default=0.15)

    p.add_argument("--memory-bank-size", type=int, default=0)
    p.add_argument("--hard-neg-weight", type=float, default=1.0)

    p.add_argument("--num-proj-layers", type=int, default=None)
    p.add_argument("--proj-dim", type=int, default=None)
    p.add_argument("--proj-ff-mult", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True

    # Load model
    overrides = {}
    if args.num_proj_layers is not None:
        overrides["num_proj_layers"] = args.num_proj_layers
    if args.proj_dim is not None:
        overrides["proj_dim"] = args.proj_dim
    if args.proj_ff_mult is not None:
        overrides["proj_ff_mult"] = args.proj_ff_mult

    model, ckpt = TraitEncoder.load_pretrained(args.checkpoint, device="cpu", **overrides)
    model = model.to(device)

    # Start fully frozen, then selectively unfreeze below
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

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ckpt_mgr = CheckpointManager(args.output_dir, "trait_encoder_hier", "finetune", "hier",
                                  run_dir=args.run_dir)
    log = setup_logging("finetune", str(ckpt_mgr.log_path))
    log.info("Model: %.1fM total, %.1fM trainable (%.1f%%)",
             total / 1e6, trainable / 1e6, 100 * trainable / total)

    # Tokenizer + data
    tokenizer = ZESTTokenizer(args.tokenizer)
    train_ds = HierarchicalPairDataset(
        args.shards_dir, tokenizer, max_tokens=args.max_tokens,
        shards_per_epoch=args.shards_per_epoch, bpe_dropout=args.bpe_dropout,
    )
    collate = lambda b: finetune_collate_fn(b, tokenizer.pad_id)

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
    optimizer = optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    # Memory bank
    proj_dim = m.proj_dim
    memory_bank = MemoryBank(args.memory_bank_size, proj_dim, device) if args.memory_bank_size > 0 else None

    bs = args.batch_size
    ga = args.grad_accum_steps
    nw = args.num_workers
    pf = 6 if nw > 0 else None

    # Estimate steps per epoch (streaming dataset — approximate from first epoch)
    log.info("Starting training: %d epochs, BS=%d, GA=%d, LR=%.2e (encoder=%.2e)",
             args.epochs, bs, ga, args.lr, encoder_lr)

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        loader = DataLoader(train_ds, batch_size=bs, collate_fn=collate,
                            num_workers=nw, pin_memory=True,
                            prefetch_factor=pf if nw > 0 else None)

        # Build scheduler on first epoch (need loader length)
        if epoch == 1:
            # Rough estimate — count batches on first pass
            est_steps = 100000  # placeholder, refined after first epoch
            total_steps = (est_steps // ga) * args.epochs
            warmup = int(total_steps * args.warmup_ratio)
            scheduler = build_cosine_schedule(optimizer, total_steps, warmup)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        acc_metrics = {"loss": 0, "con": 0, "sap": 0, "rank": 0, "acc_h": 0}
        n_steps, n_batches = 0, 0
        t0 = time.time()

        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
            m_step = train_step(model, batch, optimizer, scheduler, scaler, device,
                                args.grad_clip, args, ga, i % ga, memory_bank)
            if m_step is None:
                continue
            for k in acc_metrics:
                acc_metrics[k] += m_step.get(k, 0)
            n_steps += 1
            n_batches += 1

        elapsed = time.time() - t0
        avg = {k: v / max(n_steps, 1) for k, v in acc_metrics.items()}
        log.info("Epoch %d/%d [%.0fs] loss=%.4f con=%.4f sap=%.4f rank=%.4f acc_h=%.4f",
                 epoch, args.epochs, elapsed, avg["loss"], avg["con"],
                 avg["sap"], avg["rank"], avg["acc_h"])

        ckpt_mgr.add_metrics("train", epoch, avg)
        ckpt_mgr.save(model, optimizer, scheduler, epoch)

    log.info("=" * 50)
    log.info("Fine-tuning complete. Checkpoints: %s", ckpt_mgr.models_dir)


if __name__ == "__main__":
    main()
