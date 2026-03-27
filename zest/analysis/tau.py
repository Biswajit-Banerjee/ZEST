"""
τ(T) — Task-relevant bits per token.

Measures how much label information (e.g., fold, superfamily) a tokenizer's
vocabulary carries on average. Higher τ = more informative tokenization.

    τ(T) = I(T; Y) / E[tokens per sequence]

where I(T; Y) = Σ_t Σ_y P(t,y) log₂(P(t,y) / (P(t)·P(y)))

This is a *pre-training* metric — no model needed. It quantifies the
inductive bias baked into the tokenizer's vocabulary.

Usage:
    python -m zest.analysis.tau \
        --fasta data/scope40.fa \
        --tokenizers zest:data/tokens.json bpe:data/bpe_32k.json \
        --output outputs/tau_results.json
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Mutual Information computation
# ---------------------------------------------------------------------------

def compute_tau(
    tokenized_sequences: List[List[str]],
    labels: List[str],
    label_name: str = "label",
) -> Dict:
    """
    Compute τ(T) = I(T;Y) / avg_tokens_per_sequence.

    Args:
        tokenized_sequences: list of token-string lists, one per sequence
        labels: class label for each sequence
        label_name: descriptive name for the label level

    Returns:
        dict with τ, total MI, compression stats, top informative tokens
    """
    n_seqs = len(tokenized_sequences)
    assert n_seqs == len(labels)

    token_count: Counter = Counter()
    label_token_count: Counter = Counter()
    joint_count: Dict[str, Counter] = defaultdict(Counter)
    total_tokens = 0

    for tokens, label in zip(tokenized_sequences, labels):
        n_tok = len(tokens)
        total_tokens += n_tok
        label_token_count[label] += n_tok
        for t in tokens:
            token_count[t] += 1
            joint_count[t][label] += 1

    # MI(t; Y) for each token type
    token_mi = {}
    for t, t_count in token_count.items():
        p_t = t_count / total_tokens
        mi_t = 0.0
        for y, ty_count in joint_count[t].items():
            p_ty = ty_count / total_tokens
            p_y = label_token_count[y] / total_tokens
            if p_ty > 0 and p_t > 0 and p_y > 0:
                mi_t += p_ty * math.log2(p_ty / (p_t * p_y))
        token_mi[t] = mi_t

    total_mi = sum(token_mi.values())
    avg_tokens_per_seq = total_tokens / n_seqs
    tau = total_mi / avg_tokens_per_seq

    weighted_mi = sum(
        (token_count[t] / total_tokens) * token_mi[t] for t in token_count
    )

    n_unique_labels = len(set(labels))
    max_mi = math.log2(n_unique_labels) if n_unique_labels > 1 else 1.0

    # Top tokens by MI contribution
    contributions = sorted(
        [(t, token_count[t] / total_tokens, token_mi[t],
          (token_count[t] / total_tokens) * token_mi[t])
         for t in token_count],
        key=lambda x: -x[3],
    )

    return {
        "label_type": label_name,
        "n_sequences": n_seqs,
        "n_unique_labels": n_unique_labels,
        "max_possible_mi_bits": max_mi,
        "total_tokens": total_tokens,
        "unique_tokens": len(token_count),
        "avg_tokens_per_seq": avg_tokens_per_seq,
        "total_MI_bits": total_mi,
        "tau_bits_per_seq": tau,
        "weighted_avg_MI_per_token": weighted_mi,
        "top_tokens": contributions[:20],
    }


def print_tau(name: str, r: Dict):
    """Pretty-print τ results."""
    print(f"\n{'='*60}")
    print(f"  {name} | {r['label_type']} ({r['n_unique_labels']} classes)")
    print(f"{'='*60}")
    print(f"  Sequences: {r['n_sequences']:,}  |  Tokens: {r['total_tokens']:,}  "
          f"|  Vocab: {r['unique_tokens']:,}")
    print(f"  Avg tokens/seq: {r['avg_tokens_per_seq']:.1f}")
    print(f"  τ(T)   = {r['tau_bits_per_seq']:.4f} bits/sequence")
    print(f"  I(T;Y) = {r['total_MI_bits']:.4f} bits (of {r['max_possible_mi_bits']:.1f} max)")
    print(f"  E[MI]  = {r['weighted_avg_MI_per_token']:.6f} bits/token")
    print(f"  Top 10 tokens:")
    for t, p, mi, c in r["top_tokens"][:10]:
        d = t if len(t) <= 15 else t[:12] + "..."
        print(f"    {d:<16s} P={p:.5f}  MI={mi:.5f}  contrib={c:.6f}")


def compare_tokenizers(results: Dict[str, Dict]) -> None:
    """Print a summary comparison table."""
    print(f"\n{'='*70}")
    print(f"  τ(T) Summary")
    print(f"{'='*70}")
    print(f"  {'Tokenizer':<14} {'Label':<14} {'τ':>10} {'I(T;Y)':>10} "
          f"{'Tok/seq':>8} {'Ratio':>8}")
    print(f"  {'-'*14} {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    # Find BPE baseline for ratio computation
    bpe_tau = {}
    for key, r in results.items():
        if "bpe" in key.lower() or "BPE" in key:
            bpe_tau[r["label_type"]] = r["tau_bits_per_seq"]

    for key, r in results.items():
        parts = key.split("_", 1)
        tok_name = parts[0] if len(parts) > 1 else key
        label = r["label_type"]
        base = bpe_tau.get(label, r["tau_bits_per_seq"])
        ratio = r["tau_bits_per_seq"] / base if base > 0 else 0
        print(f"  {tok_name:<14} {label:<14} {r['tau_bits_per_seq']:>10.4f} "
              f"{r['total_MI_bits']:>10.4f} "
              f"{r['avg_tokens_per_seq']:>8.1f} {ratio:>7.2f}×")


# ---------------------------------------------------------------------------
# Generic tokenize helpers (domain-agnostic)
# ---------------------------------------------------------------------------

def tokenize_character(sequences: List[str], alphabet: set) -> List[List[str]]:
    """Character-level tokenization with a given alphabet."""
    return [[c for c in seq if c in alphabet] for seq in sequences]


def tokenize_with_callable(
    sequences: List[str],
    encode_fn: Callable[[str], List[int]],
    id_to_token: Dict[int, str],
) -> List[List[str]]:
    """Tokenize using any encoder that returns token IDs."""
    result = []
    for seq in sequences:
        ids = encode_fn(seq)
        tokens = [id_to_token.get(tid, f"<{tid}>") for tid in ids]
        result.append(tokens)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compute τ(T) for tokenizer comparison")
    p.add_argument("--sequences", required=True, nargs="+",
                   help="Sequence files (FASTA, JSON lines, etc.)")
    p.add_argument("--labels", required=True,
                   help="Label file (JSON: {seq_id: label}) or 'fasta-header' for SCOPe-style")
    p.add_argument("--label-levels", nargs="+", default=["fold", "superfamily"],
                   help="Which hierarchy levels to compute τ for")
    p.add_argument("--tokenizers", nargs="+", required=True,
                   help="Tokenizers as name:path pairs (e.g., zest:data/tokens.json)")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args()

    print("τ(T) computation — use domain-specific scripts in experiments/ "
          "for full pipeline with FASTA parsing.")
    print("This CLI provides the core computation engine.")


if __name__ == "__main__":
    main()
