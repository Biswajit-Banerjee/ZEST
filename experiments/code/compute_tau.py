"""
Compute τ(T) for code tokenizers on POJ-104 (clone detection labels).

CPU-only, no model needed. Measures mutual information between
tokenizer vocabulary and algorithm class labels.

Usage:
    python experiments/code/compute_tau.py \
        --poj104-dir data/code/poj104/ \
        --zest-vocab data/code/zest_code_vocab.json \
        --output outputs/code/tau_results.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from zest.analysis.tau import compute_tau, print_tau, compare_tokenizers


# ---------------------------------------------------------------------------
# POJ-104 loader
# ---------------------------------------------------------------------------

def load_poj104(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load POJ-104 dataset.

    Expected layout: data_dir/{problem_id}/{source_file.c}
    Or: data_dir/programs.jsonl with {code, label} per line.

    Returns:
        sources: list of source code strings
        labels: list of problem IDs (algorithm class)
    """
    data_path = Path(data_dir)
    sources, labels = [], []

    # Try JSONL format first
    jsonl = data_path / "programs.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                obj = json.loads(line)
                sources.append(obj["code"])
                labels.append(str(obj["label"]))
        return sources, labels

    # Try directory format: {problem_id}/{file}
    for problem_dir in sorted(data_path.iterdir()):
        if not problem_dir.is_dir():
            continue
        pid = problem_dir.name
        for src_file in sorted(problem_dir.glob("*.c")) + sorted(problem_dir.glob("*.cpp")):
            code = src_file.read_text(errors="replace")
            if len(code) > 10:
                sources.append(code)
                labels.append(pid)

    return sources, labels


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

def tokenize_character(sources: List[str]) -> List[List[str]]:
    """Character-level tokenization (printable ASCII)."""
    printable = set(chr(i) for i in range(32, 127))
    return [[c for c in s if c in printable] for s in sources]


def tokenize_bpe(sources: List[str], tok_path: str) -> List[List[str]]:
    """Standard BPE tokenization."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    return [tok.encode(s).tokens for s in sources]


def tokenize_zest_code(sources: List[str], vocab_path: str,
                       language: str = "c") -> List[List[str]]:
    """
    ZEST-Code tokenization: lex → greedy match against pattern vocabulary.

    This uses tree-sitter to convert source to lexer token stream,
    then applies greedy max-match against the mined patterns.
    """
    from experiments.code.build_tokenizer import lex_function

    with open(vocab_path) as f:
        vocab_map = json.load(f)

    # Build pattern → id mapping
    patterns = {}
    for i, centroid in enumerate(vocab_map.keys()):
        parts = tuple(centroid.split())
        patterns[parts] = centroid

    # Sort by length (longest first) for greedy matching
    sorted_patterns = sorted(patterns.keys(), key=len, reverse=True)

    result = []
    for source in sources:
        try:
            stream = lex_function(source, language)
        except Exception:
            result.append(stream if 'stream' in dir() else [])
            continue

        tokens = []
        i = 0
        while i < len(stream):
            matched = False
            for pattern in sorted_patterns:
                plen = len(pattern)
                if i + plen <= len(stream) and tuple(stream[i:i+plen]) == pattern:
                    tokens.append(patterns[pattern])
                    i += plen
                    matched = True
                    break
            if not matched:
                tokens.append(stream[i])  # single lexer token (base alphabet)
                i += 1
        result.append(tokens)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="τ(T) for code tokenizers")
    p.add_argument("--poj104-dir", required=True, help="POJ-104 dataset directory")
    p.add_argument("--zest-vocab", default=None, help="ZEST-Code vocab_map.json")
    p.add_argument("--bpe-tokenizer", default=None, help="BPE tokenizer JSON")
    p.add_argument("--language", default="c", help="Language for tree-sitter lexing")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    sources, labels = load_poj104(args.poj104_dir)
    print(f"POJ-104: {len(sources)} programs, {len(set(labels))} classes")

    tokenizers = {"Char": tokenize_character(sources)}

    if args.zest_vocab and Path(args.zest_vocab).exists():
        print("Tokenizing with ZEST-Code...")
        tokenizers["ZEST-Code"] = tokenize_zest_code(sources, args.zest_vocab, args.language)

    if args.bpe_tokenizer and Path(args.bpe_tokenizer).exists():
        print("Tokenizing with BPE...")
        tokenizers["BPE-Code"] = tokenize_bpe(sources, args.bpe_tokenizer)

    all_results = {}
    for tok_name, tok_seqs in tokenizers.items():
        key = f"{tok_name}_algorithm"
        r = compute_tau(tok_seqs, labels, "algorithm")
        r["dataset"] = "POJ-104"
        all_results[key] = r
        print_tau(f"POJ-104 / {tok_name}", r)

    if all_results:
        compare_tokenizers(all_results)

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
