# LEMON-ZEST: Evolution-Informed Tokenization for Efficient Protein Language Modeling

**ZEST** (*Zoned Encoding of Sequence Themes*) is a motif-level protein tokenizer built from evolutionarily conserved sequence patterns (~3.3 residues/token across a 32K vocabulary). **LEMON** (`ThemeEncoder`) pairs ZEST with a compact 200M-parameter Transformer encoder, a hierarchical contrastive projection head for remote homology retrieval, and an AlphaFold-style **RecycledResidueDecoder** for per-residue sequence and structure tasks.

## Architecture

```text
Amino-acid sequence
   └─► ZESTTokenizer (32K motif trie, ~3.3 aa/token)
          └─► Encoder (24× RoPE + SwiGLU Flash-Attention Transformer, D=768)
                 ├─► Representation Head: AttentionAggregator → Bottleneck MLP → 384-D L2 embedding
                 └─► RecycledResidueDecoder: Expand + FiLM(offset) → N× shared RefinementBlock
                     (±2 token cross-attn + ±16 residue self-attn + SwiGLU) → 384-D residue features & 20-AA logits
```

See [docs/lemon.mermaid](docs/lemon.mermaid) for the full diagram.

## Repository Structure

```text
zest/
  tokenizer.py             # ZEST greedy max-match motif trie tokenizer
  tokenizer_bpe.py         # Protein BPE tokenizer (for tokenization ablations)
  model/
    encoder.py             # RoPE + SwiGLU Flash-Attention Transformer encoder
    theme_encoder.py       # ThemeEncoder + presets (test / base / deep) + strict checkpoint loader
    heads.py               # RecycledResidueDecoder & RefinementBlock
    pooling.py             # AttentionAggregator
    positional.py          # Rotary & scaled positional embeddings
  data/
    pretrain.py            # UniRef MLM / MPM datasets with span masking & residue-budget truncation
    finetune.py            # Hierarchical CATH pair dataset (A/T/H levels)
    netsurfp.py            # Leakage-checked NetSurfP-3.0 / CB513 loader
    download.py            # Automated benchmark dataset fetcher
    loaders.py             # FASTA / Parquet / text sequence loaders
  training/
    pretrain.py            # Single-GPU / DDP token + residue MLM pretraining
    finetune.py            # Hierarchical contrastive + auxiliary residue fine-tuning
    losses.py              # Hierarchical InfoNCE, hard-positive InfoNCE, SmoothAP, margin ranking, memory bank
    steps.py               # Training / evaluation step helpers
    utils.py               # CheckpointManager, LLRD, cosine schedules, logging
  evaluation/
    benchmark.py           # SCOPe / CATH retrieval evaluation
    residue_probe.py       # Leakage-free residue-level linear probe with paired chain bootstrap
  analysis/
    tau.py                 # τ(T) task-relevant bits per token metric
data/tokens/               # Canonical 32K ZEST and BPE vocabularies
scripts/
  train.sh                 # End-to-end pretrain → contrastive fine-tune → retrieval eval template
  contrastive_finetune.py  # Clean single-GPU contrastive head fine-tuning with P-K batch sampling
experiments/biology/       # τ(T) computation scripts
tests/                     # Unit test suite (pytest)
```

> **Note on Benchmarking**: A unified multi-model benchmarking suite is tracked separately and will be added in a follow-up PR. Core SCOPe/CATH retrieval (`zest.evaluation.benchmark`) and leakage-free residue probing (`zest.evaluation.residue_probe`) are included in this package.

## Installation

```bash
pip install -e .
```

For development and running tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Quick Start

### Load a Pretrained Checkpoint & Extract Embeddings

```python
import torch
from zest import ZESTTokenizer, ThemeEncoder

tokenizer = ZESTTokenizer("data/tokens/tedlh_zest_32K.json")
model, ckpt = ThemeEncoder.load_pretrained("path/to/best.pth", device="cuda")
model.eval()

seq = "MKTLLLTLVVVTIVCLDLGYT"
ids = tokenizer.encode(seq)
tokens = torch.tensor([ids], device="cuda")
mask = torch.ones_like(tokens, dtype=torch.bool)

with torch.no_grad():
    # Sequence-level contrastive embedding [1, 384]
    seq_emb = model.project_family(tokens, mask, temperature=None)

    # Token-level encoder representations [1, T, 768]
    tok_embs = model.encode_tokens(tokens, mask)
```

### Extract Per-Residue Representations & Logits (`RecycledResidueDecoder`)

```python
segments = tokenizer._segment(list(seq), dropout=0.0)
token_ids = [tid for _, tid in segments]
token_lengths = [length for length, _ in segments]
n_res = sum(token_lengths)

tokens = torch.tensor([token_ids], device="cuda")
mask = torch.ones_like(tokens, dtype=torch.bool)
lengths = torch.tensor([token_lengths], device="cuda")
global_idx = torch.arange(n_res, device="cuda").unsqueeze(0)

with torch.no_grad():
    aa_logits, res_features = model.forward_residue_mlm(
        tokens, mask, lengths, global_idx, return_features=True
    )
    # aa_logits: [1, L, 20], res_features: [1, L, 384]
```

### Training

```bash
# 1. Pretrain (single GPU or torchrun DDP)
python -m zest.training.pretrain \
    --model deep \
    --tokenizer data/tokens/tedlh_zest_32K.json \
    --train-data path/to/uniref_train_shards \
    --val-data path/to/uniref_val_shards \
    --span-mask --amp

# 2. Hierarchical contrastive fine-tuning (CATH)
python -m zest.training.finetune \
    --checkpoint path/to/pretrain/best.pth \
    --tokenizer data/tokens/tedlh_zest_32K.json \
    --shards-dir path/to/cath_pair_shards \
    --amp

# 3. Compute τ(T) tokenizer information metric (CPU)
python experiments/biology/compute_tau.py \
    --scope-fasta data/scope40.fa --cath-fasta data/cath_s20.fa
```

## License

MIT — see [LICENSE](LICENSE).
