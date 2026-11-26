#!/usr/bin/env bash
# LEMON-ZEST training pipeline: MLM pretraining -> hierarchical contrastive fine-tuning -> retrieval eval
# Adjust data paths and hyperparameters for your environment.

set -euo pipefail

MODEL="${MODEL:-deep}"
TOKENIZER="${TOKENIZER:-data/tokens/tedlh_zest_32K.json}"
TRAIN_DATA="${TRAIN_DATA:-data/uniref90/train}"
VAL_DATA="${VAL_DATA:-data/uniref90/val}"
CATH_SHARDS="${CATH_SHARDS:-data/cath/shards}"
OUTPUT="${OUTPUT:-outputs/models}"

# ======================================================================
# Phase 1: Token + Residue MLM Pretraining (RecycledResidueDecoder)
# ======================================================================
echo "=== Phase 1: Pretraining ($MODEL) ==="
python -m zest.training.pretrain \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --batch-size 64 \
    --grad-accum-steps 4 \
    --lr 3e-4 \
    --max-tokens 1024 \
    --token-mlm-weight 0.7 \
    --residue-mlm-weight 0.3 \
    --span-mask \
    --amp \
    --output-dir "$OUTPUT"

RUN_DIR=$(ls -td "$OUTPUT"/"$MODEL"_* | head -1)
PRETRAIN_CKPT="$RUN_DIR/pretrain/best.pth"
echo "Pretrain checkpoint: $PRETRAIN_CKPT"

# ======================================================================
# Phase 2: Hierarchical Contrastive Fine-tuning (CATH)
# ======================================================================
echo "=== Phase 2: Contrastive Fine-tuning ==="
python -m zest.training.finetune \
    --checkpoint "$PRETRAIN_CKPT" \
    --tokenizer "$TOKENIZER" \
    --shards-dir "$CATH_SHARDS" \
    --epochs 8 \
    --batch-size 128 \
    --grad-accum-steps 4 \
    --lr 3e-4 \
    --encoder-lr-scale 0.2 \
    --unfreeze-layers -1 \
    --memory-bank-size 16384 \
    --hard-neg-weight 2.0 \
    --max-tokens 512 \
    --amp \
    --output-dir "$OUTPUT"

FINAL_DIR=$(ls -td "$OUTPUT"/finetune_* | head -1)
FINAL_CKPT="$FINAL_DIR/finetune/best.pth"

# ======================================================================
# Phase 3: SCOPe / CATH Retrieval Benchmark
# ======================================================================
echo "=== Phase 3: Retrieval Evaluation ==="
python -m zest.evaluation.benchmark \
    --checkpoint "$FINAL_CKPT" \
    --tokenizer "$TOKENIZER" \
    --scope-fasta data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa \
    --retrieval-fasta data/cath-dataset-nonredundant-S20-v4_4_0.fa

echo "Done! Final checkpoint: $FINAL_CKPT"
