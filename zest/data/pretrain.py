"""
MPM (Masked Profile Modeling) pretraining dataset.

Each sample provides:
  - BPE token IDs (with optional masking for token-level MLM)
  - HMM emission profiles as soft regression targets
  - Token-to-residue alignment for the profile expansion head
"""
import glob
import logging
import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_log = logging.getLogger(__name__)

AA_TO_IDX = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7,
    "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
}
AA_SANITIZE = {"B": "D", "J": "L", "O": "K", "U": "C", "X": "A", "Z": "E"}


def _load_parquet(path: str, columns=None) -> pd.DataFrame:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files in {path}")
        return pd.concat([pd.read_parquet(f, columns=columns) for f in files],
                         ignore_index=True)
    return pd.read_parquet(path, columns=columns)


class PretrainDataset(Dataset):
    """
    Dataset for Masked Profile Modeling.

    Expects:
      - sequences_path: parquet with columns [hmm_id, raw_sequence, alignment_indices]
      - emissions_path: parquet with columns [hmm_id, emission_matrix]
    """

    def __init__(self, sequences_path: str, emissions_path: str, tokenizer,
                 is_training: bool = True, hmm_ids: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.max_tokens = max_tokens
        self._background = [0.0] * 20

        df = _load_parquet(sequences_path, columns=["hmm_id", "raw_sequence", "alignment_indices"])
        if hmm_ids is not None:
            df = df[df["hmm_id"].isin(set(hmm_ids))].reset_index(drop=True)

        em_df = _load_parquet(emissions_path, columns=["hmm_id", "emission_matrix"])
        self.emissions: Dict[str, list] = em_df.set_index("hmm_id")["emission_matrix"].to_dict()

        self._ids = df["hmm_id"].to_numpy()
        self._seqs = df["raw_sequence"].to_numpy()
        self._alns = df["alignment_indices"].tolist()
        _log.info("PretrainDataset: %d samples", len(df))

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int):
        hmm_id = self._ids[idx]
        raw_seq = self._seqs[idx]
        aln_idx = self._alns[idx]
        if not isinstance(aln_idx, list):
            aln_idx = list(aln_idx)

        emissions = self.emissions[hmm_id]
        dropout = random.uniform(0.0, 0.5) if self.is_training else 0.0

        text = self.tokenizer._ws.sub("", raw_seq).upper()
        segments = self.tokenizer._segment(text, dropout=dropout)
        if self.max_tokens:
            segments = segments[:self.max_tokens]

        token_ids = [tid for _, tid in segments]
        token_lengths = [length for length, _ in segments]
        n_residues = sum(token_lengths)

        # Emission targets aligned to residues
        kept_aln = aln_idx[:n_residues]
        n_em = len(emissions)
        targets, mask = [], []
        for ai in kept_aln:
            if isinstance(ai, (int, float)) and 0 <= int(ai) < n_em:
                targets.append(emissions[int(ai)])
                mask.append(1)
            else:
                targets.append(self._background)
                mask.append(0)

        # Residue-level AA labels for auxiliary MLM
        aa_labels = []
        for aa in text[:n_residues]:
            aa = AA_SANITIZE.get(aa, aa)
            aa_labels.append(AA_TO_IDX.get(aa, 0))

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "target_mask": torch.tensor(mask, dtype=torch.float32),
            "global_indices": torch.arange(len(targets), dtype=torch.long),
            "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
            "aa_labels": torch.tensor(aa_labels, dtype=torch.long),
        }


def pretrain_collate_fn(batch: List[Dict], pad_id: int = 0, mask_id: int = 4,
                        vocab_size: int = 32000, apply_mask: bool = True,
                        mask_prob: float = 0.15):
    """
    Collate with optional BERT-style masking for token-level MLM.

    Returns both masked and original token IDs so the profile head always
    sees unmasked input while the token MLM head predicts masked positions.
    """
    from torch.nn.utils.rnn import pad_sequence

    ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    tgt = pad_sequence([b["targets"] for b in batch], batch_first=True, padding_value=0.0)
    gi = pad_sequence([b["global_indices"] for b in batch], batch_first=True, padding_value=0)
    tl = pad_sequence([b["token_lengths"] for b in batch], batch_first=True, padding_value=0)
    tm = pad_sequence([b["target_mask"] for b in batch], batch_first=True, padding_value=0.0)
    aa = pad_sequence([b["aa_labels"] for b in batch], batch_first=True, padding_value=-100)

    attn_mask = ids != pad_id
    masked_ids = ids.clone()
    token_labels = None

    if apply_mask and mask_prob > 0:
        prob = torch.full(ids.shape, mask_prob).masked_fill_(~attn_mask, 0.0)
        to_mask = torch.bernoulli(prob).bool()

        # 80% replace with [MASK], 10% random, 10% keep
        replace = torch.bernoulli(torch.full(ids.shape, 0.8)).bool() & to_mask
        masked_ids[replace] = mask_id
        rand_pos = torch.bernoulli(torch.full(ids.shape, 0.5)).bool() & to_mask & ~replace
        masked_ids[rand_pos] = torch.randint(2, vocab_size, ids.shape)[rand_pos]

        token_labels = torch.full_like(ids, -100)
        token_labels[to_mask] = ids[to_mask]

    return {
        "input_ids": masked_ids,
        "orig_input_ids": ids,
        "attention_mask": attn_mask,
        "targets": tgt,
        "target_mask": tm,
        "global_indices": gi,
        "token_lengths": tl,
        "aa_labels": aa,
        "token_labels": token_labels,
    }
