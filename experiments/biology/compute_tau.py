"""
Compute τ(T) for protein tokenizers on SCOPe and CATH datasets.

CPU-only, no model needed. Measures mutual information between
tokenizer vocabulary and structural classification labels.

Usage:
    python experiments/biology/compute_tau.py \
        --scope-fasta data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa \
        --cath-fasta data/cath-dataset-nonredundant-S20-v4_4_0.fa \
        --output outputs/biology/tau_results.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from zest.tokenizer import ZESTTokenizer
from zest.analysis.tau import compute_tau, print_tau, compare_tokenizers

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# FASTA parsers
# ---------------------------------------------------------------------------

def _load_cath_domain_map(path: str) -> Dict[str, str]:
    mapping = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) >= 6:
                mapping[p[0]] = f"{p[1]}.{p[2]}.{p[3]}.{p[4]}.{p[5]}"
    return mapping


def parse_scope_fasta(path: str) -> List[Dict]:
    entries, cur_id, cur_cls, seq = [], None, None, []

    def flush():
        if cur_id and seq:
            s = "".join(seq).upper()
            entries.append({"id": cur_id, "classification": cur_cls, "sequence": s})

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                seq = []
                parts = line[1:].split()
                cur_id = parts[0]
                cur_cls = parts[1] if len(parts) > 1 else None
            else:
                seq.append(line)
    flush()
    return [e for e in entries if e["classification"]
            and len(e["classification"].split(".")) >= 4]


def parse_cath_fasta(fasta: str, domain_list: str) -> List[Dict]:
    cmap = _load_cath_domain_map(domain_list)
    entries, cur_id, seq = [], None, []

    def flush():
        if cur_id and seq:
            cls = cmap.get(cur_id)
            if cls:
                entries.append({"id": cur_id, "classification": cls,
                                "sequence": "".join(seq).upper()})

    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                seq = []
                pp = line[1:].split("|")
                cur_id = pp[2].split("/")[0] if len(pp) >= 3 else None
            else:
                seq.append(line)
    flush()
    return [e for e in entries if e["classification"]
            and len(e["classification"].split(".")) >= 5]


def extract_labels(entries: List[Dict], dataset: str) -> Tuple[List[str], List[str]]:
    """Extract fold and superfamily labels."""
    folds, sfs = [], []
    for e in entries:
        p = e["classification"].split(".")
        if dataset == "cath":
            folds.append(".".join(p[:3]))  # C.A.T
            sfs.append(".".join(p[:4]))    # C.A.T.H
        else:
            folds.append(".".join(p[:2]))  # C.F
            sfs.append(".".join(p[:3]))    # C.F.SF
    return folds, sfs


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

def tokenize_single_aa(sequences: List[str]) -> List[List[str]]:
    return [[c for c in s.upper() if c in AA_SET] for s in sequences]


def tokenize_zest(sequences: List[str], tok: ZESTTokenizer) -> List[List[str]]:
    result = []
    for s in sequences:
        ids = tok.encode(s, add_special_tokens=False)
        result.append([tok.id_to_token.get(i, "<UNK>") for i in ids])
    return result


def tokenize_bpe(sequences: List[str], tok_path: str) -> List[List[str]]:
    """BPE tokenization using HuggingFace tokenizers."""
    from tokenizers import Tokenizer
    hf_tok = Tokenizer.from_file(tok_path)
    result = []
    for s in sequences:
        enc = hf_tok.encode(s.upper())
        result.append(enc.tokens)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(name: str, entries: List[Dict], ds_type: str,
                tokenizers: Dict[str, List[List[str]]]) -> Dict:
    sequences = [e["sequence"] for e in entries]
    folds, sfs = extract_labels(entries, ds_type)

    print(f"\n{'#'*60}")
    print(f"  {name}: {len(entries)} sequences")
    print(f"  Folds: {len(set(folds))}, SFs: {len(set(sfs))}")
    print(f"  Avg length: {np.mean([len(s) for s in sequences]):.0f} residues")
    print(f"{'#'*60}")

    results = {}
    for label_name, labels in [("fold", folds), ("superfamily", sfs)]:
        for tok_name, tok_seqs in tokenizers.items():
            key = f"{tok_name}_{label_name}"
            r = compute_tau(tok_seqs, labels, label_name)
            # Fix residues/token
            total_res = sum(len(s) for s in sequences)
            total_tok = sum(len(t) for t in tok_seqs)
            r["avg_residues_per_token"] = total_res / total_tok if total_tok else 1.0
            r["dataset"] = name
            results[f"{name}_{key}"] = r
            print_tau(f"{name} / {tok_name}", r)

    return results


def main():
    p = argparse.ArgumentParser(description="τ(T) for protein tokenizers")
    p.add_argument("--scope-fasta",
                   default="data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
    p.add_argument("--cath-fasta",
                   default="data/cath-dataset-nonredundant-S20-v4_4_0.fa")
    p.add_argument("--cath-domain-list",
                   default="data/cath-domain-list-v4_4_0.txt")
    p.add_argument("--zest-tokenizer", default="data/tokens.json")
    p.add_argument("--bpe-tokenizer", default="data/tedlh_bpe_32k.json")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    all_results = {}

    for fasta, dl, ds_type, ds_name in [
        (args.scope_fasta, None, "scope", "SCOPe40"),
        (args.cath_fasta, args.cath_domain_list, "cath", "CATH-S20"),
    ]:
        if not Path(fasta).exists():
            print(f"Skipping {ds_name}: {fasta} not found")
            continue
        if ds_type == "cath" and not Path(dl).exists():
            print(f"Skipping {ds_name}: domain list not found")
            continue

        entries = (parse_cath_fasta(fasta, dl) if ds_type == "cath"
                   else parse_scope_fasta(fasta))
        sequences = [e["sequence"] for e in entries]

        tokenizers = {"Single-AA": tokenize_single_aa(sequences)}

        if Path(args.zest_tokenizer).exists():
            zest = ZESTTokenizer(args.zest_tokenizer)
            tokenizers["ZEST"] = tokenize_zest(sequences, zest)

        if Path(args.bpe_tokenizer).exists():
            try:
                tokenizers["BPE-32K"] = tokenize_bpe(sequences, args.bpe_tokenizer)
            except Exception as e:
                print(f"BPE tokenizer failed: {e}, skipping")

        results = run_dataset(ds_name, entries, ds_type, tokenizers)
        all_results.update(results)

    if all_results:
        compare_tokenizers(all_results)

        # ZEST/BPE ratios
        print(f"\n  ZEST / BPE τ ratios:")
        for key, r in all_results.items():
            if "ZEST" in key:
                bpe_key = key.replace("ZEST", "BPE-32K")
                if bpe_key in all_results:
                    ratio = r["tau_bits_per_seq"] / all_results[bpe_key]["tau_bits_per_seq"]
                    print(f"    {key}: {ratio:.2f}×")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save = {}
        for k, r in all_results.items():
            sr = {kk: v for kk, v in r.items() if kk != "top_tokens"}
            sr["top_tokens"] = [{"token": t, "p": p, "mi": mi, "contrib": c}
                                for t, p, mi, c in r["top_tokens"]]
            save[k] = sr
        with open(args.output, "w") as f:
            json.dump(save, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
