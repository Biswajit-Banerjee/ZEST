"""Data loading helpers for various file formats (FASTA, parquet, text)."""
import glob
import gzip
import logging
import os
from typing import List, Optional

import pandas as pd

_log = logging.getLogger(__name__)


def load_parquet(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a single parquet file or concatenate all parquet files in a directory."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files in {path}")
        return pd.concat([pd.read_parquet(f, columns=columns) for f in files],
                         ignore_index=True)
    return pd.read_parquet(path, columns=columns)


def parse_fasta(path: str) -> List[str]:
    """Parse a single FASTA file, returning a list of sequences."""
    seqs: List[str] = []
    current: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        seqs.append("".join(current))
    return seqs


def load_sequences(path: str, seq_column: str = "sequence") -> List[str]:
    """Load sequences from FASTA, parquet, or line-delimited text.

    Supported formats:
      - Directory of .fasta/.fa files (protein shards)
      - Single .fasta/.fa file
      - Directory of .parquet files
      - Single .parquet file
      - .txt or .txt.gz (one sequence per line)
    """
    # --- FASTA (single file or directory of shards) ---
    fasta_exts = (".fasta", ".fa", ".fna", ".faa")
    if os.path.isdir(path):
        fasta_files = sorted(
            f for f in glob.glob(os.path.join(path, "*"))
            if any(f.endswith(ext) for ext in fasta_exts)
        )
        if fasta_files:
            seqs: List[str] = []
            for fp in fasta_files:
                seqs.extend(parse_fasta(fp))
            return seqs
        # fall through to parquet directory handling

    if any(path.endswith(ext) for ext in fasta_exts):
        return parse_fasta(path)

    # --- Text (line-delimited) ---
    if path.endswith(".txt") or path.endswith(".txt.gz"):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            return [line.strip() for line in f if line.strip()]

    # --- Parquet (single file or directory) ---
    df = load_parquet(path)
    for col in [seq_column, "raw_sequence", "sequence", "text", "code"]:
        if col in df.columns:
            seq_column = col
            break
    return df[seq_column].tolist()
