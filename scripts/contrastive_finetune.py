#!/usr/bin/env python3
"""Simplified contrastive-only fine-tuning for ThemeEncoder.

A clean, single-GPU script that trains only the contrastive head
(aggregator + projector + learned temperature) with hierarchical InfoNCE
on CATH A/T/H labels.  No DDP, no residue MLM, no teacher KD.

Includes a P-K batch sampler that guarantees same-superfamily co-occurrence
in every batch, which is critical for the contrastive losses to fire.

Encoding path:
    Sequence -> ZESTTokenizer -> ThemeEncoder.encode_tokens()
    -> Aggregator -> Projector -> L2-normalize -> embedding

Loss (choose one via --probability-mode):
    "none"          — standard hierarchical_infonce (binary positives). Baseline.
    "margin"        — hierarchical_infonce + margin ranking using paired probability.
                       Spreads same-H cluster internally by probability order.
                       NOTE: tends to hurt mAP (spreads positives apart) — use only
                       if you specifically want relative rank-ordering, not for
                       maximizing AUROC/mAP or remote-homology retrieval.
    "soft"          — probability-weighted InfoNCE (per-positive weighted average).
                       High-probability (easy) pairs pulled harder than low-probability
                       (hard/remote) pairs.
    "hard_positive" — inverse-weighted InfoNCE: LOW-probability (hard/remote, but
                       still same-H) pairs get MORE pull, not less. This is the mode
                       aligned with "improve remote homology detection" — it spends
                       more gradient on the pairs the model is least likely to already
                       get right, instead of on easy/obvious same-superfamily pairs.
                       Pairs below --hard-positive-floor are treated as untrusted
                       broadcast noise and excluded entirely (weight 0), not amplified.

Regardless of mode, hierarchical_infonce ALWAYS runs at full weight (weight_a/t/h
from CLI) — the probability-mode losses are additive refinements on top of it,
never a replacement for the base attract/repel signal.

Includes a P-K batch sampler that guarantees same-superfamily co-occurrence
in every batch, which is critical for the contrastive losses to fire.

Usage:
    export TMPDIR="${HOME}/.tmp_dl"   # avoid shared /tmp tmpfs IPC crashes
    mkdir -p "$TMPDIR"
    python experiments/contrastive_finetune/train.py \
        --checkpoint outputs/.../best.pth \
        --shards-dir data/training/finetuning/train \
        --val-shards-dir data/training/finetuning/val \
        --epochs 20 --batch-size 128 --lr 5e-4 --amp \
        --probability-mode hard_positive
"""
import argparse
from collections import defaultdict
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zest.model.theme_encoder import ThemeEncoder
from zest.tokenizer import ZESTTokenizer
from zest.data.finetune import HierarchicalPairDataset, finetune_collate_fn
from zest.training.losses import (
    MemoryBank, hierarchical_infonce, hierarchical_smooth_ap,
    hierarchical_margin_ranking, selective_coarse_preservation_loss,
)
from zest.training.utils import (
    CheckpointManager, build_cosine_schedule, build_llrd_param_groups,
    setup_logging, save_environment, save_model_arch,
)


# ---------------------------------------------------------------------------
# P-K Batch Sampler — guarantees same-H co-occurrence
# ---------------------------------------------------------------------------

class PKBatchSampler(Sampler):
    """P-K metric learning batch sampler.

    Each batch: K superfamilies x P pairs each = K*P total pairs.
    This guarantees that every batch has multiple same-H entries, so
    contrastive losses at the H-level always have positives.

    Only same-SF pairs (h_labels == h_pos_labels) are included.
    """

    def __init__(self, h_labels, h_pos_labels, active,
                 sfs_per_batch=16, pairs_per_sf=8, seed=42):
        self.K = sfs_per_batch
        self.P = pairs_per_sf
        self.batch_size = sfs_per_batch * pairs_per_sf
        self.seed = seed
        self._epoch = 0

        # Only include within-SF pairs
        within_sf = h_labels[active] == h_pos_labels[active]
        within_active = active[within_sf]

        # Build SF -> [indices] mapping
        self._sf_to_indices = defaultdict(list)
        for idx in within_active:
            sf = int(h_labels[idx])
            self._sf_to_indices[sf].append(int(idx))

        # Filter SFs with too few pairs
        self._sf_keys = [sf for sf, idxs in self._sf_to_indices.items()
                         if len(idxs) >= 2]
        self._all_within = [int(i) for i in within_active]
        self._n_batches = max(1, len(self._all_within) // self.batch_size)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        # Shuffle per-SF pools
        pools = {}
        for sf in self._sf_keys:
            pool = list(self._sf_to_indices[sf])
            rng.shuffle(pool)
            pools[sf] = pool

        for _ in range(self._n_batches):
            batch = []
            # Sample K superfamilies
            if len(self._sf_keys) >= self.K:
                chosen_sfs = rng.sample(self._sf_keys, self.K)
            else:
                chosen_sfs = list(self._sf_keys)
                while len(chosen_sfs) < self.K:
                    chosen_sfs.append(rng.choice(self._sf_keys))

            for sf in chosen_sfs:
                pool = pools.get(sf, [])
                if len(pool) >= self.P:
                    drawn = pool[:self.P]
                    pools[sf] = pool[self.P:]
                elif pool:
                    drawn = list(pool)
                    # Refill and reshuffle
                    refill = list(self._sf_to_indices[sf])
                    rng.shuffle(refill)
                    needed = self.P - len(drawn)
                    drawn.extend(refill[:needed])
                    pools[sf] = refill[needed:]
                else:
                    # Pool exhausted, refill
                    refill = list(self._sf_to_indices[sf])
                    rng.shuffle(refill)
                    drawn = refill[:self.P]
                    pools[sf] = refill[self.P:]
                batch.extend(drawn)

            yield batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _encode(model, tokens, mask):
    """tokens -> L2-normalised projection."""
    m = _get_model(model)
    embs = m.encode_tokens(tokens, mask)
    family = m.aggregator(embs, mask)
    return F.normalize(m.projector(family), p=2, dim=-1)


def _encode_pair(model, anc_tok, anc_mask, pos_tok, pos_mask):
    """Encode anchor + positive, batching when seq lengths match."""
    if anc_tok.shape[1] == pos_tok.shape[1]:
        tok = torch.cat([anc_tok, pos_tok])
        msk = torch.cat([anc_mask, pos_mask])
        proj = _encode(model, tok, msk)
        B = anc_tok.shape[0]
        return proj[:B], proj[B:]
    return _encode(model, anc_tok, anc_mask), _encode(model, pos_tok, pos_mask)


# ---------------------------------------------------------------------------
# Probability-weighted InfoNCE — correct per-positive weighted average
# ---------------------------------------------------------------------------

def _paired_weighted_infonce(
    embeddings, h_labels, pair_weights, temp=0.05,
    mem_emb=None, mem_h=None,
):
    """Shared core for 'soft' and 'hard_positive' modes.

    H-level InfoNCE where the known (anchor_i, positive_i) pair gets a
    custom weight (pair_weights[i]), and all other same-H entries in the
    batch default to weight 1.0 (standard binary positive).

    Uses weighted AVERAGE of per-positive log-ratios (SupCon formulation),
    NOT log of a weighted sum — this ensures every positive gets its own
    gradient term scaled by weight, rather than letting the exp() peak
    swamp low-weight positives.

    pair_weights: [B] weight for each (anchor_i, positive_i) pair, >= 0.
                  Caller decides what this represents (probability for
                  'soft', inverse-probability for 'hard_positive').

    After concatenation: embeddings = [anc_0..anc_B, pos_0..pos_B], shape [2B, D].
    """
    B = pair_weights.shape[0]
    N = embeddings.shape[0]  # = 2B
    device = embeddings.device
    zero = embeddings.new_zeros(())

    if mem_emb is not None and mem_emb.shape[0] > 0:
        keys = torch.cat([embeddings, mem_emb.to(device)])
        key_h = torch.cat([h_labels, mem_h.to(device)])
    else:
        keys, key_h = embeddings, h_labels
    K = keys.shape[0]

    sim = embeddings @ keys.T  # [N, K]
    self_mask = torch.zeros(N, K, dtype=torch.bool, device=device)
    self_mask[:N, :N] = torch.eye(N, dtype=torch.bool, device=device)

    same_h = h_labels.unsqueeze(1) == key_h.unsqueeze(0)  # [N, K]
    pos = same_h & ~self_mask
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return {"loss": zero, "acc_h": zero, "n_pos_per_query": zero}

    # Build weight matrix [N, K]:
    # - default 1.0 for all same-H positives (standard binary contribution)
    # - override the known paired entries (anc_i <-> pos_i) with pair_weights[i]
    w = pos.float()

    if N >= 2 * B and K >= 2 * B:
        idx = torch.arange(B, device=device)
        w[idx, B + idx] = pair_weights          # anc_i -> pos_i
        w[B + idx, idx] = pair_weights          # pos_i -> anc_i

    logits = sim / temp  # [N, K]
    logits_masked = logits.masked_fill(self_mask, -1e9)

    log_sum_exp = torch.logsumexp(logits_masked, dim=1, keepdim=True)  # [N, 1]
    log_prob = logits_masked - log_sum_exp  # [N, K]

    per_pos_nll = -log_prob  # [N, K]
    numerator = (w * per_pos_nll).sum(1)
    denominator = w.sum(1).clamp(min=1e-8)
    loss_per_query = numerator / denominator
    loss = loss_per_query[has_pos].mean()

    top1 = logits_masked.argmax(dim=1)
    acc = pos.gather(1, top1.unsqueeze(1)).squeeze(1)[has_pos].float().mean()

    n_pos = pos.float().sum(1)[has_pos].mean()

    return {"loss": loss, "acc_h": acc, "n_pos_per_query": n_pos.detach()}


def probability_weighted_infonce(embeddings, h_labels, probabilities, temp=0.05,
                                  mem_emb=None, mem_h=None):
    """'soft' mode: high-probability (easy) pairs pulled harder."""
    return _paired_weighted_infonce(
        embeddings, h_labels, probabilities, temp=temp,
        mem_emb=mem_emb, mem_h=mem_h,
    )


def hard_positive_weighted_infonce(embeddings, h_labels, probabilities, temp=0.05,
                                    gamma=1.0, prob_floor=0.3,
                                    mem_emb=None, mem_h=None):
    """'hard_positive' mode: low-probability (hard/remote) pairs pulled harder.

    weight = (1 - probability)^gamma for probability >= prob_floor, else 0.

    The floor exists because very low broadcast probabilities can mean either
    "genuinely remote homolog" (what we want to amplify) or "noisy/low-coverage
    broadcast label" (what we do NOT want to amplify) — probability alone can't
    distinguish these, so untrustworthy pairs are dropped rather than upweighted.
    """
    weight = torch.where(
        probabilities >= prob_floor,
        (1.0 - probabilities).clamp(min=0.0) ** gamma,
        torch.zeros_like(probabilities),
    )
    return _paired_weighted_infonce(
        embeddings, h_labels, weight, temp=temp,
        mem_emb=mem_emb, mem_h=mem_h,
    )


# ---------------------------------------------------------------------------
# Single train / eval step
# ---------------------------------------------------------------------------

def compute_loss(model, batch, device, amp, args,
                  memory_bank=None, teacher=None):
    """Forward pass -> contrastive loss (mode-dependent) + optional teacher KD."""
    anc_tok = batch["anchor_ids"].to(device, non_blocking=True)
    anc_mask = batch["anchor_mask"].to(device, non_blocking=True)
    pos_tok = batch["positive_ids"].to(device, non_blocking=True)
    pos_mask = batch["positive_mask"].to(device, non_blocking=True)
    a_anc = batch["a_labels"].to(device, non_blocking=True)
    t_anc = batch["t_labels"].to(device, non_blocking=True)
    h_anc = batch["h_labels"].to(device, non_blocking=True)
    a_pos = batch["a_pos_labels"].to(device, non_blocking=True)
    t_pos = batch["t_pos_labels"].to(device, non_blocking=True)
    h_pos = batch["h_pos_labels"].to(device, non_blocking=True)
    probs = batch["probabilities"].to(device, non_blocking=True)

    prob_mode = getattr(args, "probability_mode", "none")

    # Hard-fail fast if probabilities aren't normalised to [0,1] — a silent
    # scale mismatch here (e.g. raw 0-100 HHblits values) poisons both the
    # margin-ranking hinge and the soft/hard_positive weighting without ever
    # raising an error, so we check it explicitly every step it matters.
    if prob_mode in ("margin", "soft", "hard_positive"):
        pmax = probs.max().item() if probs.numel() > 0 else 0.0
        assert pmax <= 1.0 + 1e-4, (
            f"probabilities must be normalised to [0,1], got max={pmax:.3f}. "
            f"Fix at the data-loading layer (divide by 100 if HHblits raw scale)."
        )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        anc_proj, pos_proj = _encode_pair(model, anc_tok, anc_mask, pos_tok, pos_mask)

        all_emb = torch.cat([anc_proj, pos_proj])
        all_a = torch.cat([a_anc, a_pos])
        all_t = torch.cat([t_anc, t_pos])
        all_h = torch.cat([h_anc, h_pos])

        m = _get_model(model)
        tau = m.temperature
        scale = tau / 0.07  # init_temp
        temp_a = args.temp_a * scale
        temp_t = args.temp_t * scale
        temp_h = args.temp_h * scale

        mem = memory_bank.sample(args.memory_bank_samples) if memory_bank else None

        # ── Base hierarchical InfoNCE — ALWAYS full weight, all modes.
        # This is the core attract/repel signal. Probability-mode losses
        # below are additive refinements, never a substitute for this.
        infonce = hierarchical_infonce(
            all_emb, all_a, all_t, all_h,
            temp_a=temp_a, temp_t=temp_t, temp_h=temp_h,
            weight_a=args.weight_a, weight_t=args.weight_t, weight_h=args.weight_h,
            mem_emb=mem[0] if mem else None,
            mem_a=mem[1] if mem else None,
            mem_t=mem[2] if mem else None,
            mem_h=mem[3] if mem else None,
            hard_neg_weight=args.hard_neg_weight,
        )

        loss = infonce["loss"]
        rank_loss = torch.tensor(0.0, device=device)
        soft_loss = torch.tensor(0.0, device=device)
        hp_loss = torch.tensor(0.0, device=device)
        kd_loss = torch.tensor(0.0, device=device)
        n_pos = torch.tensor(0.0, device=device)

        if prob_mode == "margin":
            rank = hierarchical_margin_ranking(
                anc_proj, pos_proj,
                a_anc, t_anc, h_anc, a_pos, t_pos, h_pos,
                probs, margin=args.rank_margin,
                level_weights=(0.0, 0.0, 1.0),
            )
            rank_loss = rank["rank_loss"]
            loss = loss + args.rank_weight * rank_loss

        elif prob_mode == "soft":
            soft_result = probability_weighted_infonce(
                all_emb, all_h, probs,
                temp=temp_h,
                mem_emb=mem[0] if mem else None,
                mem_h=mem[3] if mem else None,
            )
            soft_loss = soft_result["loss"]
            n_pos = soft_result.get("n_pos_per_query", torch.tensor(0.0))
            loss = loss + args.soft_weight * soft_loss

        elif prob_mode == "hard_positive":
            hp_result = hard_positive_weighted_infonce(
                all_emb, all_h, probs,
                temp=temp_h,
                gamma=args.hard_positive_gamma,
                prob_floor=args.hard_positive_floor,
                mem_emb=mem[0] if mem else None,
                mem_h=mem[3] if mem else None,
            )
            hp_loss = hp_result["loss"]
            n_pos = hp_result.get("n_pos_per_query", torch.tensor(0.0))
            loss = loss + args.hard_positive_weight * hp_loss

        # ── Teacher KD: coarse-preservation regularization ──
        if teacher is not None and args.teacher_weight > 0:
            with torch.no_grad():
                t_anc_proj, t_pos_proj = _encode_pair(
                    teacher, anc_tok, anc_mask, pos_tok, pos_mask)
                teacher_emb = torch.cat([t_anc_proj, t_pos_proj])
            label_valid = (all_h >= 0)
            kd_loss = selective_coarse_preservation_loss(
                all_emb, teacher_emb, all_h, label_valid,
                temperature=args.teacher_temperature,
            )
            loss = loss + args.teacher_weight * kd_loss

        # Optional SmoothAP — directly optimizes a differentiable AP surrogate,
        # the loss most directly aligned with the mAP metric being tracked.
        sap_loss = torch.tensor(0.0, device=device)
        if args.smooth_ap_weight > 0:
            sap = hierarchical_smooth_ap(
                all_emb, all_a, all_t, all_h,
                temp=args.smooth_ap_temp,
                weight_a=args.weight_a, weight_t=args.weight_t, weight_h=args.weight_h,
            )
            sap_loss = sap["sap_loss"]
            loss = loss + args.smooth_ap_weight * sap_loss

    metrics = {
        "loss": loss.detach(),
        "con": infonce["loss"].detach(),
        "sap": sap_loss.detach(),
        "rank": rank_loss.detach(),
        "soft": soft_loss.detach(),
        "hp": hp_loss.detach(),
        "kd": kd_loss.detach(),
        "acc_a": infonce.get("acc_a", torch.tensor(0.0, device=device)).detach(),
        "acc_t": infonce.get("acc_t", torch.tensor(0.0, device=device)).detach(),
        "acc_h": infonce.get("acc_h", torch.tensor(0.0, device=device)).detach(),
        "tau": tau.detach(),
        "n_pos": n_pos.detach(),
    }
    return loss, (all_emb, all_a, all_t, all_h), metrics


METRIC_KEYS = ("loss", "con", "sap", "rank", "soft", "hp", "kd",
               "acc_a", "acc_t", "acc_h", "tau", "n_pos")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Contrastive-only fine-tuning")
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--truncate-layers",  type=int,   default=None,
                   help="Use only the first N encoder layers from the checkpoint.")
    p.add_argument("--resume",           default=None)
    p.add_argument("--tokenizer",        default=os.path.join(
                       str(Path(__file__).resolve().parents[2]), "zest_tokens.json"))
    p.add_argument("--shards-dir",       default="data/training/finetuning/train")
    p.add_argument("--val-shards-dir",   default="data/training/finetuning/val")
    p.add_argument("--shards-per-epoch", type=int, default=None)
    p.add_argument("--output-dir",       default="outputs/contrastive_finetune")
    p.add_argument("--run-dir",          default=None)

    # Training — defaults match the working unified finetune stage-B
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--grad-accum-steps", type=int,   default=16)
    p.add_argument("--lr",               type=float, default=3e-5,
                   help="Head LR (aggregator, projector, log_temperature)")
    p.add_argument("--encoder-lr",       type=float, default=3e-6,
                   help="Encoder LR (unfrozen layers)")
    p.add_argument("--warmup-ratio",     type=float, default=0.025)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--llrd-decay",       type=float, default=0.9)
    p.add_argument("--max-tokens",       type=int,   default=512)
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--bpe-dropout",      type=float, default=0.15)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--amp",              action="store_true")

    # Freezing
    p.add_argument("--unfreeze-layers",  type=int,   default=4,
                   help="Encoder layers to unfreeze from top (-1 = all)")
    p.add_argument("--unfreeze-embeddings", action="store_true")

    # Teacher KD — prevents catastrophic collapse of pretrained structure
    p.add_argument("--teacher-checkpoint", default=None,
                   help="Frozen teacher for KD. If None, uses the loaded "
                        "checkpoint itself as teacher (self-distillation)")
    p.add_argument("--teacher-weight",   type=float, default=0.1,
                   help="Weight for coarse-preservation KD loss")
    p.add_argument("--teacher-temperature", type=float, default=0.1)

    # Loss
    p.add_argument("--temp-h",           type=float, default=0.05)
    p.add_argument("--temp-t",           type=float, default=0.07)
    p.add_argument("--temp-a",           type=float, default=0.10)
    p.add_argument("--weight-h",         type=float, default=0.50)
    p.add_argument("--weight-t",         type=float, default=0.35)
    p.add_argument("--weight-a",         type=float, default=0.15)
    p.add_argument("--hard-neg-weight",  type=float, default=1.0)
    p.add_argument("--smooth-ap-weight", type=float, default=0.0)
    p.add_argument("--smooth-ap-temp",   type=float, default=0.01)
    p.add_argument("--memory-bank-size", type=int,   default=0)
    p.add_argument("--memory-bank-samples", type=int, default=4096)

    # Probability-based supervision (choose one)
    p.add_argument("--probability-mode",
                   choices=["none", "margin", "soft", "hard_positive"],
                   default="none",
                   help="How to use HHblits paired probability: "
                        "'none'=binary InfoNCE only (baseline), "
                        "'margin'=rank-order same-H pairs by probability "
                        "(spreads cluster internally — tends to HURT mAP), "
                        "'soft'=pull high-probability pairs harder, "
                        "'hard_positive'=pull LOW-probability (hard/remote) "
                        "pairs harder — aligned with remote-homology retrieval.")
    p.add_argument("--rank-weight",      type=float, default=0.05,
                   help="Weight for margin ranking loss (--probability-mode margin)")
    p.add_argument("--rank-margin",      type=float, default=0.2)
    p.add_argument("--soft-weight",      type=float, default=0.15,
                   help="Weight for soft probability-weighted loss (--probability-mode soft)")
    p.add_argument("--hard-positive-weight", type=float, default=0.15,
                   help="Weight for hard-positive loss (--probability-mode hard_positive)")
    p.add_argument("--hard-positive-gamma", type=float, default=1.0,
                   help="Focusing exponent: (1-prob)^gamma. Higher = more focus on hardest pairs.")
    p.add_argument("--hard-positive-floor", type=float, default=0.3,
                   help="Pairs with probability below this are excluded (weight=0), "
                        "not amplified — guards against amplifying broadcast label noise.")

    # Batch sampler
    p.add_argument("--structured-batching", action="store_true",
                   help="Use P-K batch sampler for guaranteed same-H co-occurrence")
    p.add_argument("--sfs-per-batch",    type=int, default=16,
                   help="K: superfamilies per batch (P-K sampler)")
    p.add_argument("--pairs-per-sf",     type=int, default=8,
                   help="P: pairs per superfamily (P-K sampler)")

    # Validation / checkpointing
    p.add_argument("--val-every",        type=int,   default=1)
    p.add_argument("--val-max-batches",  type=int,   default=200)
    p.add_argument("--save-every-steps", type=int,   default=0)
    p.add_argument("--keep-all-checkpoints", action="store_true")

    # Progress-bar averaging window (avoids the misleading cumulative-since-
    # epoch-start average that looks like "degradation" as it converges from
    # a noisy first batch toward the true steady-state value)
    p.add_argument("--log-window",       type=int,   default=200,
                   help="Rolling window size for the live progress-bar average. "
                        "Larger = less batch-composition noise in the display, "
                        "at the cost of reacting more slowly to real trends. "
                        "The end-of-epoch and validation numbers are unaffected "
                        "by this — they always average over the full epoch/val set.")

    # Augmentation
    p.add_argument("--crop-ratio",       type=float, default=0.7)
    p.add_argument("--crop-prob",        type=float, default=0.0)
    args = p.parse_args()
    if args.truncate_layers is not None and args.truncate_layers < 1:
        p.error("--truncate-layers must be positive")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    elif args.resume:
        run_dir = Path(args.resume).parent.parent
    else:
        from zest.training.utils import make_run_dir
        run_dir = make_run_dir(args.output_dir, "contrastive")

    keep_n = 999 if args.keep_all_checkpoints else 3
    ckpt_mgr = CheckpointManager(run_dir, phase="finetune", keep_last_n=keep_n)
    log = setup_logging("contrastive_ft", str(ckpt_mgr.log_path))

    # Load model
    model, _ = ThemeEncoder.load_pretrained(
        args.checkpoint, device="cpu", num_layers=args.truncate_layers
    )
    model = model.to(device)

    # Freeze everything, then selectively unfreeze
    for p in model.parameters():
        p.requires_grad = False
    m = _get_model(model)
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

    # Always train contrastive head
    for module in [m.aggregator, m.projector]:
        for p in module.parameters():
            p.requires_grad = True
    m.log_temperature.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %.1fM total, %.1fM trainable (%.1f%%)",
             total_params / 1e6, trainable / 1e6, 100 * trainable / total_params)

    # Tokenizer + data
    tokenizer = ZESTTokenizer(args.tokenizer)
    log.info("Loading training data from %s ...", args.shards_dir)
    train_ds = HierarchicalPairDataset(
        args.shards_dir, tokenizer, max_tokens=args.max_tokens,
        shards_per_epoch=args.shards_per_epoch, bpe_dropout=args.bpe_dropout,
        crop_ratio=args.crop_ratio, crop_prob=args.crop_prob,
    )
    collate = lambda b: finetune_collate_fn(b, tokenizer.pad_id)

    bs, ga = args.batch_size, args.grad_accum_steps
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)

    persistent = args.num_workers > 0

    def make_loader(ds, shuffle=True, drop_last=True, gen=None,
                    batch_sampler=None):
        if batch_sampler is not None:
            return torch.utils.data.DataLoader(
                ds, batch_sampler=batch_sampler,
                num_workers=args.num_workers, pin_memory=True,
                collate_fn=collate,
                persistent_workers=persistent,
                prefetch_factor=2 if args.num_workers > 0 else None,
            )
        return torch.utils.data.DataLoader(
            ds, bs, shuffle=shuffle, num_workers=args.num_workers,
            pin_memory=True, collate_fn=collate, drop_last=drop_last,
            generator=gen or loader_gen,
            persistent_workers=persistent,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    # Build batch sampler if requested
    pk_sampler = None
    if args.structured_batching:
        pk_sampler = PKBatchSampler(
            train_ds._h_labels, train_ds._h_pos_labels, train_ds._active,
            sfs_per_batch=args.sfs_per_batch,
            pairs_per_sf=args.pairs_per_sf,
            seed=args.seed,
        )
        bs = pk_sampler.batch_size  # override CLI batch-size
        if args.batch_size != bs:
            log.warning(
                "--batch-size %d IGNORED: --structured-batching is on, so "
                "batch size is forced to --sfs-per-batch(%d) x --pairs-per-sf(%d) "
                "= %d. To change batch size, edit those two flags instead.",
                args.batch_size, args.sfs_per_batch, args.pairs_per_sf, bs)
        log.info("PKBatchSampler: K=%d SFs x P=%d pairs = %d per batch, "
                 "%d usable SFs, %d batches/epoch",
                 args.sfs_per_batch, args.pairs_per_sf, bs,
                 len(pk_sampler._sf_keys), len(pk_sampler))
    else:
        log.info("WARNING: --structured-batching not set — same-H "
                 "co-occurrence per batch is NOT guaranteed. Probability-"
                 "mode losses (margin/soft/hard_positive) may silently see "
                 "few or zero eligible pairs per batch. Strongly recommend "
                 "--structured-batching whenever --probability-mode != none.")

    # Teacher model for KD regularization
    teacher = None
    if args.teacher_weight > 0:
        teacher_path = args.teacher_checkpoint or args.checkpoint
        log.info("Loading teacher from %s (weight=%.2f, temp=%.2f)",
                 teacher_path, args.teacher_weight, args.teacher_temperature)
        teacher, _ = ThemeEncoder.load_pretrained(teacher_path, device="cpu")
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # Optimizer
    encoder_lr = args.encoder_lr
    if args.llrd_decay < 1.0 and args.unfreeze_layers != 0:
        groups = build_llrd_param_groups(
            model, encoder_lr, args.lr, args.llrd_decay,
            head_prefixes=("aggregator", "projector", "log_temperature"))
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
    memory_bank = (MemoryBank(args.memory_bank_size, m.proj_dim, device)
                   if args.memory_bank_size > 0 else None)

    n_batches_ep = len(pk_sampler) if pk_sampler else max(1, len(train_ds) // bs)
    steps_ep = max(1, n_batches_ep // ga)
    sched = build_cosine_schedule(
        opt, args.epochs * steps_ep,
        int(args.epochs * steps_ep * args.warmup_ratio),
    )

    # Save config
    config = {
        "phase": "contrastive_finetune",
        "checkpoint": str(args.checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint or args.checkpoint),
        "params_M": round(total_params / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "epochs": args.epochs, "batch_size": bs, "grad_accum": ga,
        "lr": args.lr, "encoder_lr": encoder_lr,
        "encoder_layers": m.core.num_layers,
        "truncate_layers": args.truncate_layers,
        "unfreeze_layers": args.unfreeze_layers,
        "llrd_decay": args.llrd_decay,
        "temperatures": {"a": args.temp_a, "t": args.temp_t, "h": args.temp_h},
        "weights": {"a": args.weight_a, "t": args.weight_t, "h": args.weight_h},
        "hard_neg_weight": args.hard_neg_weight,
        "smooth_ap_weight": args.smooth_ap_weight,
        "memory_bank_size": args.memory_bank_size,
        "teacher_weight": args.teacher_weight,
        "teacher_temperature": args.teacher_temperature,
        "probability_mode": args.probability_mode,
        "rank_weight": args.rank_weight,
        "rank_margin": args.rank_margin,
        "soft_weight": args.soft_weight,
        "hard_positive_weight": args.hard_positive_weight,
        "hard_positive_gamma": args.hard_positive_gamma,
        "hard_positive_floor": args.hard_positive_floor,
        "structured_batching": args.structured_batching,
        "sfs_per_batch": args.sfs_per_batch,
        "pairs_per_sf": args.pairs_per_sf,
        "seed": args.seed, "amp": amp,
    }
    (ckpt_mgr.run_dir / "config.json").write_text(json.dumps(config, indent=2))
    save_environment(ckpt_mgr.run_dir)
    save_model_arch(ckpt_mgr.run_dir, model)
    log.info("Run directory: %s", ckpt_mgr.run_dir)
    log.info("Probability mode: %s", args.probability_mode)

    # Resume
    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt = ckpt_mgr.load(args.resume, model, opt, sched, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        log.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    # SIGTERM handler
    _shutdown = False
    def _sigterm(signum, frame):
        nonlocal _shutdown
        _shutdown = True
        log.info("SIGTERM — saving checkpoint after current step")
    signal.signal(signal.SIGTERM, _sigterm)

    # Validation
    val_ds = None
    val_loader = None
    val_gen = torch.Generator()
    val_gen.manual_seed(args.seed + 10000)

    def run_validation():
        nonlocal val_ds, val_loader
        if val_loader is None:
            log.info("Loading validation data from %s ...", args.val_shards_dir)
            val_ds = HierarchicalPairDataset(
                args.val_shards_dir, tokenizer, max_tokens=args.max_tokens)
            val_loader = make_loader(val_ds, shuffle=False, drop_last=False, gen=val_gen)
        val_gen.manual_seed(args.seed + 10000)
        model.eval()
        totals = {k: 0.0 for k in METRIC_KEYS}
        count = 0
        max_b = args.val_max_batches if args.val_max_batches > 0 else None
        t0 = time.time()
        for batch in val_loader:
            try:
                with torch.no_grad():
                    _, _, ms = compute_loss(model, batch, device, amp, args,
                                            teacher=teacher)
                for k in METRIC_KEYS:
                    totals[k] += ms[k].item()
                count += 1
                if max_b and count >= max_b:
                    break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise
        elapsed = time.time() - t0
        return {k: v / max(1, count) for k, v in totals.items()}, elapsed

    log.info("Starting training: %d epochs, BS=%d, GA=%d, LR=%.2e",
             args.epochs, bs, ga, args.lr)

    core_frozen = not any(p.requires_grad for p in m.core.parameters())

    loader = make_loader(train_ds, batch_sampler=pk_sampler)

    for ep in range(start_epoch, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)
        loader_gen.manual_seed(args.seed + ep)
        if pk_sampler:
            pk_sampler.set_epoch(ep)

        t0 = time.time()
        model.train()
        if core_frozen:
            m.core.eval()
        opt.zero_grad(set_to_none=True)

        acc = {k: 0.0 for k in METRIC_KEYS}          # full-epoch running total (for end-of-epoch log)
        window = {k: 0.0 for k in METRIC_KEYS}        # rolling window (for live progress bar)
        n_steps = 0
        n_window = 0
        pbar = tqdm(loader, desc=f"Epoch {ep}", leave=False)

        for i, batch in enumerate(pbar):
            try:
                loss, emb_data, ms = compute_loss(
                    model, batch, device, amp, args, memory_bank, teacher)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    continue
                raise

            if math.isnan(ms["loss"].item()):
                opt.zero_grad(set_to_none=True)
                continue

            (loss / ga).backward()
            for k in METRIC_KEYS:
                v = ms[k].item()
                acc[k] += v
                window[k] += v
            n_steps += 1
            n_window += 1

            if memory_bank:
                memory_bank.enqueue(*emb_data)

            # Live postfix uses a ROLLING WINDOW average, not the cumulative
            # since-epoch-start average — the cumulative version converges
            # from a noisy first batch toward steady state over the first
            # ~30-50 steps, which looks exactly like "degradation" but is
            # just an artifact of the running mean, not real model behavior.
            pbar.set_postfix(
                loss=f"{window['loss']/n_window:.3f}",
                accH=f"{window['acc_h']/n_window:.1%}",
                accT=f"{window['acc_t']/n_window:.1%}",
                tau=f"{ms['tau'].item():.4f}",
            )
            if n_window >= args.log_window:
                window = {k: 0.0 for k in METRIC_KEYS}
                n_window = 0

            if (i + 1) % ga == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    ckpt_mgr.save_step(model, opt, sched, ep, global_step)

                if _shutdown:
                    log.info("Saving checkpoint before shutdown...")
                    ckpt_mgr.save_step(model, opt, sched, ep, global_step)
                    return

        train_elapsed = time.time() - t0
        t_m = {k: v / max(1, n_steps) for k, v in acc.items()}

        # Validation
        v_m = None
        val_elapsed = 0.0
        if args.val_shards_dir and (ep % args.val_every == 0 or ep == args.epochs):
            v_m, val_elapsed = run_validation()

        msg = (f"Epoch {ep:2d}/{args.epochs:2d} [{train_elapsed:.0f}s]"
               f" loss={t_m['loss']:.4f} con={t_m['con']:.4f}"
               f" accA={t_m['acc_a']:.3f} accT={t_m['acc_t']:.3f} accH={t_m['acc_h']:.3f}"
               f" tau={t_m['tau']:.4f}")
        if args.teacher_weight > 0:
            msg += f" kd={t_m['kd']:.4f}"
        if args.probability_mode == "margin":
            msg += f" rank={t_m['rank']:.4f}"
        elif args.probability_mode == "soft":
            msg += f" soft={t_m['soft']:.4f} n_pos={t_m['n_pos']:.1f}"
        elif args.probability_mode == "hard_positive":
            msg += f" hp={t_m['hp']:.4f} n_pos={t_m['n_pos']:.1f}"
        if v_m:
            msg += (f" | val [{val_elapsed:.0f}s] loss={v_m['loss']:.4f}"
                    f" accH={v_m['acc_h']:.3f}")
        log.info(msg)

        ckpt_mgr.save(
            model, opt, sched, ep, global_step=global_step,
            train_metrics=t_m, val_metrics=v_m,
            lr=sched.get_last_lr()[0],
            train_elapsed_s=train_elapsed, val_elapsed_s=val_elapsed,
        )

    log.info("Training complete. Best val loss: %.4f (epoch %d)",
             ckpt_mgr.best_val_loss, ckpt_mgr.best_epoch)


if __name__ == "__main__":
    main()