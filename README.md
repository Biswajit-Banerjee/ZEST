# ZEST: Injecting Inductive Bias at Tokenizer level

Domain-informed tokenization enables small models to match or exceed
much larger baselines. We demonstrate this across two domains:
**proteins** (Nature's code) and **programming languages** (Human code).

## Key Result

A 70M-parameter model with ZEST tokenization matches a 3B-parameter
baseline (ProtTucker/ProtT5-XL) on fold-level protein remote homology —
a **43× parameter reduction** explained by a measurable information-theoretic
advantage: ZEST tokens carry **1.7× more task-relevant bits** than standard BPE.

## τ(T): Task-Relevant Bits Per Token

We introduce τ(T), a pre-training metric quantifying tokenizer inductive bias:

```
τ(T) = I(T; Y) / E[tokens per sequence]
```

where I(T; Y) is the mutual information between the token vocabulary and
task labels. Higher τ = more informative tokenization = better scaling.

## Repository Structure

```
zest/                          # Shared library
  model/                       # Transformer encoder, pruning, presets
  tokenizer.py                 # ZEST: Zoned Encoding of Sequence Traits
  data/                        # Pretrain (MPM) + fine-tune (hierarchical pairs)
  training/                    # Pretrain + contrastive fine-tune pipelines
  evaluation/                  # Retrieval benchmarks
  analysis/                    # τ(T) computation, scaling analysis

experiments/
  biology/                     # Protein remote homology
    compute_tau.py             # τ for ZEST vs BPE on SCOPe/CATH
    README.md                  # Full experiment plan
  code/                        # Code clone/defect detection
    build_tokenizer.py         # tree-sitter lexing → ZEST-Code vocab
    compute_tau.py             # τ for ZEST-Code vs BPE on POJ-104
    README.md                  # Full experiment plan
```

## Approach

### Train one large model, prune to many sizes

Instead of training N models from scratch, we pretrain one large model
per tokenizer and use **layer pruning** to create smaller variants:

```python
from zest.model import prune_checkpoint

# 70M (16 layers) → 30M (4 layers)
model_30M, _ = prune_checkpoint("path/to/70M.pth", keep_layers=4, strategy="uniform")

# 70M (16 layers) → 43M (8 layers)
model_43M, _ = prune_checkpoint("path/to/70M.pth", keep_layers=8, strategy="uniform")
```

Each pruned model gets hierarchical fine-tuning, producing multiple
data points on the scaling curve from a single pretrained checkpoint.

### Biology: Protein Remote Homology

- **ZEST tokenizer**: Greedy max-match over 32K structural motifs mined from Pfam
- **Pretraining**: Masked Profile Modeling on UniRef50
- **Fine-tuning**: Hierarchical contrastive learning on CATH pairs
- **Benchmarks**: SCOPe 2.08, CATH S20 v4.4 (mAP, fold AUC, top-k recall)

### Code: Clone & Defect Detection

- **ZEST-Code tokenizer**: Greedy max-match over 32K syntax patterns mined via tree-sitter
- **Pretraining**: Masked Language Modeling on CodeSearchNet (6 languages)
- **Benchmarks**: POJ-104 (MAP@R), Devign (accuracy)

## Quick Start

```bash
pip install -r requirements.txt

# Compute τ(T) for proteins (CPU only)
python experiments/biology/compute_tau.py \
    --scope-fasta data/scope40.fa --cath-fasta data/cath_s20.fa

# Build ZEST-Code tokenizer
python experiments/code/build_tokenizer.py \
    --codesearchnet-dir data/code/codesearchnet/ \
    --output data/code/zest_code_vocab.json
```

## Citation

## License

MIT
