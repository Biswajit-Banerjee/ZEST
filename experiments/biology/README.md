# Biology Domain: Protein Remote Homology

## Task
Detect remote homologues — proteins sharing the same structural fold/superfamily
but with low sequence identity (<20%). This is one of the hardest tasks in
computational biology and requires capturing long-range evolutionary patterns.

## Tokenizers
- **ZEST**: Domain-aware motif tokenizer (32K vocab, ~3.3 residues/token)
- **BPE-32K**: Standard byte-level BPE trained on UniRef (2.9 residues/token)
- **Single-AA**: Character-level baseline (20 tokens)

## Datasets
- **Pretraining**: UniRef50 sequences + Pfam HMM emission profiles
- **Fine-tuning**: CATH S20 hierarchical pairs (64 shards, 60M pairs)
- **Evaluation**: SCOPe 2.08 (40% identity), CATH S20 v4.4

## Model Sizes (layer pruning from 70M pretrained)
| Variant | Layers | Params | Source |
|---------|--------|--------|--------|
| Small   | 4      | ~30M   | Pruned from 70M |
| Medium  | 8      | ~43M   | Pruned from 70M |
| Full    | 16     | ~70M   | Original checkpoint |

## Experiments

### 1. Scaling curve (ZEST)
Prune 70M → {4L, 8L, 16L}, fine-tune each, benchmark on SCOPe40 + CATH S20.
Plot: params vs mAP (fold + superfamily).

### 2. Tokenizer ablation
Same 70M architecture, ZEST vs BPE-32K tokenizer.
Proves the performance gap is from tokenization, not model design.

### 3. τ(T) analysis
Compute mutual information between token vocabulary and structural labels.
Already computed:
- SCOPe40 fold:  τ(ZEST)/τ(BPE) = 1.70×
- SCOPe40 SF:    τ(ZEST)/τ(BPE) = 1.68×
- CATH S20 fold: τ(ZEST)/τ(BPE) = 1.64×
- CATH S20 SF:   τ(ZEST)/τ(BPE) = 1.57×

### 4. Baselines
- ESM-2 (8M, 35M, 150M, 650M) — single-AA tokenizer
- ProtTucker (ProtT5-XL 3B backbone)

## Running

```bash
# Prune and fine-tune
python experiments/biology/prune_and_finetune.py \
    --checkpoint data/checkpoints/trait_encoder_70M_mpm.pth \
    --layer-counts 4 8 16 \
    --strategy uniform

# Compute τ(T)
python experiments/biology/compute_tau.py \
    --scope-fasta data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa \
    --cath-fasta data/cath-dataset-nonredundant-S20-v4_4_0.fa

# Benchmark
python -m zest.evaluation.benchmark \
    --checkpoint outputs/biology/4L/models/trait_encoder_hier_best.pth \
    --retrieval-fasta data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
```
