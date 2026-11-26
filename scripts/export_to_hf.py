#!/usr/bin/env python3
"""Export a LEMON ``.pth`` checkpoint to a self-contained Hugging Face Hub directory.

Generates:
  - config.json (with auto_map for AutoModel / AutoTokenizer)
  - model.safetensors
  - configuration_lemon.py, modeling_lemon.py, tokenization_lemon.py
  - tokenizer_config.json, special_tokens_map.json, vocab_map.json
  - README.md (Hugging Face model card)

Usage:
    python scripts/export_to_hf.py \
        --checkpoint path/to/best.pth \
        --tokenizer data/tokens/tedlh_zest_32K.json \
        --output-dir exports/lemon-hf \
        --verify
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from zest.hf.configuration_lemon import LemonConfig
from zest.hf.modeling_lemon import LemonModel
from zest.hf.tokenization_lemon import LemonTokenizer
from zest.model.theme_encoder import ThemeEncoder
from zest.tokenizer import ZESTTokenizer


AUTO_MAP = {
    "AutoConfig": "configuration_lemon.LemonConfig",
    "AutoModel": "modeling_lemon.LemonModel",
    "AutoModelForMaskedLM": "modeling_lemon.LemonForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_lemon.LemonForSequenceClassification",
    "AutoModelForTokenClassification": "modeling_lemon.LemonForTokenClassification",
}

MODEL_CARD = """---
library_name: transformers
pipeline_tag: feature-extraction
license: mit
tags:
  - biology
  - protein
  - protein-language-model
  - lemon
  - zest
---

# LEMON-ZEST

**LEMON** (*Layered Extraction of Molecular Ordering from Nature*) paired with **ZEST** (*Zoned Encoding of Sequence Themes*) — a 200M-parameter motif-tokenized protein language model for fast remote homology retrieval and iterative per-residue sequence/structure decoding.

## Quick Start

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
    seq_embedding = out.pooler_output        # [1, 384] L2-normalized contrastive embedding
    token_embeddings = out.last_hidden_state # [1, T, 768]
```

### Per-Residue Representations (`RecycledResidueDecoder`)

```python
res_inputs = tokenizer.encode_for_residues([seq], return_tensors="pt")
with torch.no_grad():
    aa_logits, res_features = model.encode_residues(**res_inputs)
    # aa_logits: [1, L, 20], res_features: [1, L, 384]
```

### Standard `pipeline()` Support

```python
extractor = pipeline("feature-extraction", model="SUM0N/Lemon", trust_remote_code=True)
features = extractor("MKTLLLTLVVVTIVCLDLGYT")
```

### Optional Rust Tokenizer Acceleration

`LemonTokenizer` automatically uses the flat-array Rust trie backend when `zest-tokenizer` is installed (and falls back seamlessly to pure Python otherwise).
"""


def export_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    output_dir: str,
    dtype: str = "float32",
    verify: bool = False,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load source ThemeEncoder using strict guardrails
    zest_model, ckpt = ThemeEncoder.load_pretrained(checkpoint_path, device="cpu")
    zest_model.eval()

    num_cycles = zest_model.profile_expansion_head.num_cycles
    config = LemonConfig.from_zest_config(
        zest_model.config,
        num_cycles=num_cycles,
        architectures=["LemonModel"],
        auto_map=AUTO_MAP,
    )

    # 2. Construct HF LemonModel and transfer exact state_dict
    hf_model = LemonModel(config)
    src_sd = zest_model.state_dict()
    missing, unexpected = hf_model.load_state_dict(src_sd, strict=True)
    assert not missing and not unexpected

    if dtype == "bfloat16":
        hf_model = hf_model.to(torch.bfloat16)
    elif dtype == "float16":
        hf_model = hf_model.to(torch.float16)

    # 3. Save config + safetensors
    config.save_pretrained(out)
    save_file(
        {k: v.contiguous().cpu() for k, v in hf_model.state_dict().items()},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )

    # 4. Save tokenizer + auto_map registration
    hf_tok = LemonTokenizer(vocab_file=tokenizer_path)
    hf_tok.save_pretrained(out)
    tok_cfg_path = out / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text())
    tok_cfg["auto_map"] = {"AutoTokenizer": ["tokenization_lemon.LemonTokenizer", None]}
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")

    # 5. Copy self-contained HF remote-code files
    hf_pkg_dir = Path(__file__).resolve().parents[1] / "zest" / "hf"
    for fname in ("configuration_lemon.py", "modeling_lemon.py", "tokenization_lemon.py"):
        shutil.copy2(hf_pkg_dir / fname, out / fname)

    # 6. Write Model Card
    (out / "README.md").write_text(MODEL_CARD)

    if verify:
        _verify_export(zest_model, tokenizer_path, out, dtype=dtype)

    return out


def _verify_export(zest_model: ThemeEncoder, tokenizer_path: str, out: Path, dtype: str = "float32") -> None:
    from transformers import AutoModel, AutoTokenizer

    py_tok = ZESTTokenizer(tokenizer_path)
    hf_tok = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(str(out), trust_remote_code=True).eval()

    seqs = [
        "MKTLLLTLVVVTIVCLDLGYTGEVKGKTVTIQGGVTTK",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    ]
    for s in seqs:
        assert py_tok.encode(s) == hf_tok.encode(s), f"Token mismatch for {s}"

    batch = hf_tok.encode_for_residues(seqs, return_tensors="pt")
    with torch.no_grad():
        z_proj = zest_model.project_family(batch["input_ids"], batch["attention_mask"], temperature=None)
        z_aa, z_res = zest_model.forward_residue_mlm(
            batch["input_ids"], batch["attention_mask"],
            batch["token_lengths"], batch["global_indices"], return_features=True,
        )
        hf_out = hf_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        hf_aa, hf_res = hf_model.encode_residues(**batch)

    tol = 1e-5 if dtype == "float32" else 2e-2
    d_proj = (z_proj - hf_out.pooler_output.float()).abs().max().item()
    d_aa = (z_aa - hf_aa.float()).abs().max().item()
    d_res = (z_res - hf_res.float()).abs().max().item()
    print(f"Verification max_abs_diff: pooler={d_proj:.2e}, aa_logits={d_aa:.2e}, res_feats={d_res:.2e}")
    assert d_proj < tol and d_aa < tol and d_res < tol, f"Parity exceeded {tol}: {d_proj}, {d_aa}, {d_res}"


def main() -> None:
    p = argparse.ArgumentParser(description="Export LEMON .pth checkpoint to Hugging Face format")
    p.add_argument("--checkpoint", required=True, help="Path to LEMON .pth checkpoint")
    p.add_argument("--tokenizer", default="data/tokens/tedlh_zest_32K.json", help="Path to ZEST vocab JSON")
    p.add_argument("--output-dir", required=True, help="Destination directory for HF files")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--verify", action="store_true", help="Verify numerical parity via AutoModel/AutoTokenizer")
    args = p.parse_args()

    out = export_checkpoint(args.checkpoint, args.tokenizer, args.output_dir, dtype=args.dtype, verify=args.verify)
    print(f"Exported Hugging Face model to {out}")


if __name__ == "__main__":
    main()
