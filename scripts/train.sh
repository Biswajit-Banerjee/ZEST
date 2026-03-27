#!/usr/bin/env bash
# ZEST training pipeline: MPM pretrain → hierarchical fine-tune
# Adjust paths and hyperparameters for your setup.

set -euo pipefail

MODEL="efficient-deep"   # base | efficient-deep | deep-wide
OUTPUT="outputs/models"

# ======================================================================
# Phase 1: Masked Profile Modeling (MPM) Pretraining
# ======================================================================
# 2 frozen epochs (profile head warmup) + 6 end-to-end epochs
# Token MLM is an auxiliary anti-forgetting loss (weight 0.1)

echo "=== Phase 1: MPM Pretraining ==="
python -m zest.training.pretrain \
    --model "$MODEL" \
    --frozen-epochs 2 \
    --e2e-epochs 6 \
    --batch-size 128 \
    --grad-accum-steps 8 \
    --lr 1e-4 \
    --llrd-decay 0.85 \
    --token-mlm-weight 0.1 \
    --warmup-ratio 0.05 \
    --max-tokens 1024 \
    --amp \
    --output-dir "$OUTPUT"

# Find the latest run directory
RUN_DIR=$(ls -td "$OUTPUT"/"$MODEL"_* | head -1)
MPM_CKPT="$RUN_DIR/models/lemon_mpm_best.pth"
echo "MPM checkpoint: $MPM_CKPT"

# ======================================================================
# Phase 2a: Hierarchical Fine-tuning (frozen encoder)
# ======================================================================
# Train projector + aggregator + similarity head with frozen encoder.
# Uses InfoNCE + SmoothAP + margin ranking.

echo "=== Phase 2a: Hierarchical Fine-tuning (frozen) ==="
python -m zest.training.finetune \
    --checkpoint "$MPM_CKPT" \
    --epochs 8 \
    --batch-size 1024 \
    --grad-accum-steps 1 \
    --lr 3e-4 \
    --encoder-lr-scale 0.0 \
    --unfreeze-layers 0 \
    --warmup-ratio 0.05 \
    --contrastive-weight 1.0 \
    --smooth-ap-weight 0.4 \
    --rank-weight 0.3 \
    --memory-bank-size 16384 \
    --hard-neg-weight 2.0 \
    --max-tokens 512 \
    --output-dir "$OUTPUT"

FROZEN_DIR=$(ls -td "$OUTPUT"/finetune_* | head -1)
FROZEN_CKPT="$FROZEN_DIR/models/lemon_hier_latest.pth"

# ======================================================================
# Phase 2b: Full unfreeze (fine-tune entire model)
# ======================================================================
# Unfreeze all encoder layers + embeddings. Lower encoder LR via LLRD.
# No warmup needed when resuming from trained weights.

echo "=== Phase 2b: Full Unfreeze ==="
python -m zest.training.finetune \
    --checkpoint "$FROZEN_CKPT" \
    --epochs 8 \
    --batch-size 128 \
    --grad-accum-steps 8 \
    --lr 5e-4 \
    --encoder-lr-scale 0.2 \
    --unfreeze-layers -1 \
    --unfreeze-embeddings \
    --llrd-decay 0.9 \
    --warmup-ratio 0.0 \
    --contrastive-weight 1.0 \
    --smooth-ap-weight 0.4 \
    --rank-weight 0.3 \
    --memory-bank-size 16384 \
    --hard-neg-weight 2.0 \
    --max-tokens 512 \
    --output-dir "$OUTPUT"

FINAL_DIR=$(ls -td "$OUTPUT"/finetune_* | head -1)
FINAL_CKPT="$FINAL_DIR/models/lemon_hier_latest.pth"

# ======================================================================
# Benchmark
# ======================================================================
echo "=== Benchmarking ==="
python -m zest.evaluation.benchmark \
    --checkpoint "$FINAL_CKPT" \
    --scope-fasta data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa \
    --retrieval-fasta data/cath-dataset-nonredundant-S20-v4_4_0.fa

echo "Done! Final checkpoint: $FINAL_CKPT"
