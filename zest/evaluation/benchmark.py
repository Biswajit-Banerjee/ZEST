"""
Remote homology retrieval benchmark on SCOPe and CATH datasets.

Computes mAP, ROC-AUC, Top-k recall, fold-level AUC, and other retrieval
metrics. Automatically saves results to the model's run directory.

Usage:
    python -m zest.evaluation.benchmark \
        --checkpoint path/to/model.pth \
        --retrieval-fasta data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from zest.model.theme_encoder import ThemeEncoder
from zest.tokenizer import ZESTTokenizer


DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_TOKENS = 1024
TOKENIZER_PATH = "data/tokens/tedlh_zest_32K.json"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_sequences(model, tokenizer, sequences: List[str],
                     batch_size: int, max_tokens: int,
                     device: torch.device) -> np.ndarray:
    """Encode sequences into L2-normalised embeddings."""
    model.eval()
    pad_id = tokenizer.pad_id
    all_embs = []
    for start in tqdm(range(0, len(sequences), batch_size), desc="Encoding", leave=False):
        batch_seqs = sequences[start:start + batch_size]
        encoded = [tokenizer.encode(s)[:max_tokens] for s in batch_seqs]
        max_len = max(len(e) for e in encoded)
        ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(encoded), max_len, dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = model.project_family(ids, mask, normalize=True)
        all_embs.append(emb.float().cpu().numpy())
    return np.vstack(all_embs)


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

_SCOPE_RE = re.compile(r"^>(\S+)\s+.*\b([a-g]\.\d+\.\d+\.\d+)\b")
_CATH_DOMAIN_RE = re.compile(r"^>cath\|[\d.]+\|(\S+)")


def parse_scope_fasta(path: str) -> List[Dict]:
    entries = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and seq_parts:
                    m = _SCOPE_RE.match(header)
                    if m:
                        entries.append({
                            "id": m.group(1),
                            "classification": m.group(2),
                            "sequence": "".join(seq_parts),
                        })
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
    if header and seq_parts:
        m = _SCOPE_RE.match(header)
        if m:
            entries.append({
                "id": m.group(1), "classification": m.group(2),
                "sequence": "".join(seq_parts),
            })
    return entries


def parse_cath_fasta(path: str, domain_list: Optional[str] = None) -> List[Dict]:
    """Parse CATH FASTA; optionally enrich with domain list for classification."""
    cath_map = {}
    if domain_list and Path(domain_list).exists():
        with open(domain_list) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 6:
                    cath_map[parts[0]] = f"{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}.{parts[5]}"

    entries = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and seq_parts:
                    m = _CATH_DOMAIN_RE.match(header)
                    if m:
                        did = m.group(1)
                        cls = cath_map.get(did, "")
                        if cls:
                            entries.append({
                                "id": did, "classification": cls,
                                "sequence": "".join(seq_parts),
                            })
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
    if header and seq_parts:
        m = _CATH_DOMAIN_RE.match(header)
        if m:
            did = m.group(1)
            cls = cath_map.get(did, "")
            if cls:
                entries.append({
                    "id": did, "classification": cls,
                    "sequence": "".join(seq_parts),
                })
    return entries


def detect_and_parse(path: str, domain_list: Optional[str] = None):
    """Auto-detect SCOPe vs CATH FASTA format."""
    with open(path) as f:
        first = f.readline()
    if "cath|" in first.lower():
        entries = parse_cath_fasta(path, domain_list)
        return entries, "CATH"
    entries = parse_scope_fasta(path)
    return entries, "SCOPe"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(emb_matrix: np.ndarray,
                              sf_arr: np.ndarray,
                              fam_arr: np.ndarray,
                              fold_arr: np.ndarray) -> Dict:
    """Compute comprehensive retrieval metrics using cosine similarity."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    sim = emb_matrix @ emb_matrix.T
    np.fill_diagonal(sim, -1)
    N = sim.shape[0]

    # Remote homology: same superfamily, different family
    same_sf = sf_arr[:, None] == sf_arr[None, :]
    same_fam = fam_arr[:, None] == fam_arr[None, :]
    remote = same_sf & ~same_fam
    np.fill_diagonal(remote, False)

    # Queries: sequences with at least one remote homologue
    has_remote = remote.any(axis=1)
    query_idx = np.where(has_remote)[0]

    # Per-query AP
    aps = []
    for qi in query_idx:
        scores = sim[qi]
        labels = remote[qi].astype(int)
        mask = np.ones(N, dtype=bool)
        mask[qi] = False
        # Exclude same-family (trivial positives)
        mask &= ~same_fam[qi]
        # But keep remote homologues
        mask |= remote[qi]
        if labels[mask].sum() == 0:
            continue
        aps.append(average_precision_score(labels[mask], scores[mask]))

    mean_ap = float(np.mean(aps)) if aps else 0.0

    # Global ROC-AUC (remote homology detection)
    triu = np.triu_indices(N, k=1)
    y_true = remote[triu].astype(int)
    y_score = sim[triu]
    roc = float(roc_auc_score(y_true, y_score)) if y_true.sum() > 0 else 0.0

    # Top-k recall
    ranks = np.argsort(-sim, axis=1)
    top_k_recall = {}
    for k in [1, 5, 10, 20]:
        hits = 0
        for qi in query_idx:
            top = ranks[qi, :k]
            if remote[qi][top].any():
                hits += 1
        top_k_recall[f"top{k}_recall"] = hits / max(len(query_idx), 1)

    # Fold-level ROC-AUC
    same_fold = fold_arr[:, None] == fold_arr[None, :]
    fold_y = same_fold[triu].astype(int)
    fold_roc = float(roc_auc_score(fold_y, y_score)) if fold_y.sum() > 0 else 0.0

    # MRR
    mrrs = []
    for qi in query_idx:
        r = ranks[qi]
        hits = np.where(remote[qi][r])[0]
        if len(hits) > 0:
            mrrs.append(1.0 / (hits[0] + 1))
    mrr = float(np.mean(mrrs)) if mrrs else 0.0

    return {
        "n_sequences": N, "n_queries": len(query_idx),
        "roc_auc": roc, "mean_ap": mean_ap, "mrr": mrr,
        "fold_roc_auc": fold_roc,
        **top_k_recall,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(model, tokenizer, entries, batch_size, max_tokens, device,
                  dataset_name="SCOPe", label="model"):
    sequences = [e["sequence"] for e in entries]
    classifications = [e["classification"] for e in entries]

    # Parse hierarchy levels
    if dataset_name == "CATH":
        # CATH: C.A.T.H.S → sf=C.A.T.H, fam=C.A.T.H.S, fold=C.A.T
        sf_arr = np.array([".".join(c.split(".")[:4]) for c in classifications])
        fam_arr = np.array([".".join(c.split(".")[:5]) for c in classifications])
        fold_arr = np.array([".".join(c.split(".")[:3]) for c in classifications])
    else:
        # SCOPe: a.b.c.d → sf=a.b.c.d, fam=full, fold=a.b.c
        sf_arr = np.array(classifications)
        parts = [c.split(".") for c in classifications]
        fam_arr = sf_arr  # at SF level, fam=SF for remote homology
        fold_arr = np.array([".".join(p[:3]) for p in parts])
        # For SCOPe: same SF different family → need actual family labels
        # SCOPe fam = a.b.c.d (same as SF for 4-level), use fold for remote
        fam_arr = np.array([".".join(p[:4]) for p in parts])  # SF level
        # We define "family" as exact match and "remote" as same fold, diff SF
        sf_arr = np.array([".".join(p[:3]) for p in parts])  # fold-level
        fam_arr = np.array(classifications)  # SF-level

    emb = encode_sequences(model, tokenizer, sequences, batch_size, max_tokens, device)
    metrics = compute_retrieval_metrics(emb, sf_arr, fam_arr, fold_arr)
    metrics["label"] = label
    metrics["dataset"] = dataset_name
    return metrics


def print_metrics(metrics: Dict):
    ds = metrics.get("dataset", "Retrieval")
    print(f"\n{'=' * 50}")
    print(f"  {ds} — {metrics['label']}")
    print(f"  {metrics['n_sequences']} sequences, {metrics['n_queries']} queries")
    print(f"{'=' * 50}")
    for key in ["roc_auc", "mean_ap", "mrr", "fold_roc_auc",
                "top1_recall", "top5_recall", "top10_recall", "top20_recall"]:
        if key in metrics:
            print(f"  {key:<20s} {metrics[key]:.4f}")
    print()


def save_results(checkpoint_path: str, all_output: Dict, ckpt_name: str):
    """Auto-save results to the model's run directory."""
    path = Path(checkpoint_path).resolve()
    run_dir = path.parent.parent if path.parent.name == "models" else path.parent
    if not run_dir.is_dir():
        return

    out = run_dir / f"benchmark_{ckpt_name}.json"
    with open(out, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"Results saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ThemeEncoder retrieval benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=TOKENIZER_PATH)
    p.add_argument("--tokenizer-type", default="zest", choices=["zest", "bpe", "char"])
    p.add_argument("--retrieval-fasta", nargs="+", default=[])
    p.add_argument("--scope-fasta", default=None)
    p.add_argument("--cath-domain-list", default=None)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    paths = list(args.retrieval_fasta)
    if args.scope_fasta and args.scope_fasta not in paths:
        paths.append(args.scope_fasta)
    if not paths:
        print("No FASTA files provided. Use --retrieval-fasta or --scope-fasta.")
        sys.exit(1)

    if args.tokenizer_type == "bpe":
        from experiments.biology.build_protein_bpe_tokenizer import ProteinBPETokenizer
        tokenizer = ProteinBPETokenizer(args.tokenizer)
    elif args.tokenizer_type == "char":
        from experiments.biology.char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
    else:
        tokenizer = ZESTTokenizer(args.tokenizer)
    model, _ = ThemeEncoder.load_pretrained(args.checkpoint, device="cpu")
    model = model.to(device)
    ckpt_name = Path(args.checkpoint).stem

    all_output = {}
    for fasta in paths:
        entries, ds_name = detect_and_parse(fasta, args.cath_domain_list)
        min_levels = 5 if ds_name == "CATH" else 4
        entries = [e for e in entries if e["classification"]
                   and len(e["classification"].split(".")) >= min_levels]
        print(f"\n{ds_name}: {len(entries)} domains")

        metrics = run_benchmark(model, tokenizer, entries,
                                args.batch_size, args.max_tokens, device,
                                dataset_name=ds_name, label=ckpt_name)
        print_metrics(metrics)
        all_output[ds_name.lower()] = [metrics]

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_output, f, indent=2)

    save_results(args.checkpoint, all_output, ckpt_name)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
