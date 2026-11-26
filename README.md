# LEMON-ZEST: Evolution-Informed Tokenization for Efficient Protein Language Modeling

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://neurips.cc/virtual/2026/loc/atlanta/poster/153755)
[![arXiv](https://img.shields.io/badge/arXiv-2609.37675-b31b1b.svg)](https://arxiv.org/abs/2609.37675)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-SUM0N%2FLemon-ffd21e.svg)](https://huggingface.co/SUM0N/Lemon)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Accepted at NeurIPS 2026!**
> 📄 **Paper**: [arXiv:2609.37675](https://arxiv.org/abs/2609.37675) · 🎯 **NeurIPS Poster Page**: [NeurIPS 2026 Atlanta](https://neurips.cc/virtual/2026/loc/atlanta/poster/153755) · 🤗 **Weights**: [SUM0N/Lemon](https://huggingface.co/SUM0N/Lemon)

**ZEST** (*Zoned Encoding of Sequence Traits*) is an evolution-informed protein tokenizer derived from conserved regions of multiple sequence alignments (~3.3 residues/token across a 32K vocabulary, enabling ≥3,380 residues inside a standard 1,024-token context window). **LEMON** (*Layered Extraction of Molecular Ordering from Nature*, `ThemeEncoder`) pairs ZEST with a compact 200M-parameter Transformer encoder, a hierarchical contrastive projection head for remote homology retrieval, and an AlphaFold-style **RecycledResidueDecoder** for iterative per-residue sequence and structure tasks.

## Architecture

```text
Amino-acid sequence
   └─► ZESTTokenizer (32K motif trie, ~3.3 aa/token; optional O(L) Rust backend)
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
  hf/                      # Self-contained Hugging Face Transformers integration
    configuration_lemon.py # LemonConfig
    modeling_lemon.py      # LemonModel, LemonForMaskedLM, LemonFor{Sequence,Token}Classification
    tokenization_lemon.py  # LemonTokenizer (auto-uses Rust backend when installed)
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
rust/zest_tokenizer/       # Fast O(L) flat-array Rust trie tokenizer (PyO3 + rayon, up to ~49× faster)
data/tokens/               # Canonical 32K ZEST and BPE vocabularies
scripts/
  train.sh                 # End-to-end pretrain → contrastive fine-tune → retrieval eval template
  contrastive_finetune.py  # Clean single-GPU contrastive head fine-tuning with P-K batch sampling
  export_to_hf.py          # Export .pth checkpoint → Hugging Face safetensors + remote-code bundle
experiments/biology/       # τ(T) computation scripts
tests/                     # Unit test suite (pytest)
```

> **Note on Benchmarking**: A unified multi-model benchmarking suite is tracked separately and will be added in a follow-up PR. Core SCOPe/CATH retrieval (`zest.evaluation.benchmark`) and leakage-free residue probing (`zest.evaluation.residue_probe`) are included in this package.

## Installation

```bash
pip install -e .
```

With Hugging Face Transformers support and test dependencies:

```bash
pip install -e ".[hf,dev]"
pytest tests/
```

Optional fast Rust trie tokenizer (requires Rust + `maturin` if building from source):

```bash
cd rust/zest_tokenizer
maturin develop --release
```

## Quick Start

### Hugging Face `transformers` & `pipeline()`

```python
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

repo_id = "SUM0N/Lemon"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()

seq = "MKTLLLTLVVVTIVCLDLGYTGEVKGKTVTIQGGVTTK"
inputs = tokenizer(seq, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
    seq_emb = out.pooler_output          # [1, 384] L2-normalized contrastive embedding
    tok_embs = out.last_hidden_state     # [1, T, 768]

# Per-residue embeddings & logits via RecycledResidueDecoder
res_inputs = tokenizer.encode_for_residues([seq], return_tensors="pt")
with torch.no_grad():
    aa_logits, res_features = model.encode_residues(**res_inputs)

# Standard pipeline()
extractor = pipeline("feature-extraction", model=repo_id, trust_remote_code=True)
features = extractor(seq)
```

### Native `zest` API

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
    seq_emb = model.project_family(tokens, mask, temperature=None)  # [1, 384]
    tok_embs = model.encode_tokens(tokens, mask)                    # [1, T, 768]
```

### Training & Exporting

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

# 3. Export checkpoint to Hugging Face safetensors bundle
python scripts/export_to_hf.py \
    --checkpoint path/to/best.pth \
    --tokenizer data/tokens/tedlh_zest_32K.json \
    --output-dir exports/lemon-hf \
    --verify

# 4. Compute τ(T) tokenizer information metric (CPU)
python experiments/biology/compute_tau.py \
    --scope-fasta data/scope40.fa --cath-fasta data/cath_s20.fa
```

## Citation

If you use **LEMON** or **ZEST** in your research, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{banerjee2026lemonzest,
  title     = {LEMON-ZEST: Evolution-Informed Tokenization for Efficient Protein Language Modeling},
  author    = {Banerjee, Biswajit and Carreno, Claudia Alvarez and Petrov, Anton S.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2609.37675},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.BM},
  url       = {https://arxiv.org/abs/2609.37675}
}
```

## License

MIT — see [LICENSE](LICENSE).
