"""Pretraining datasets.

- MLMDataset: Domain-agnostic masked language modeling (works for both code and proteins)
- ProteinMPMDataset: Protein-specific masked profile modeling with HMM emission targets
"""
import glob
import logging
import os
import random
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from zest.data.loaders import load_parquet, load_sequences

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain-agnostic MLM dataset
# ---------------------------------------------------------------------------

class MLMDataset(Dataset):
    """Domain-agnostic dataset for masked language model pretraining.

    Reads raw text sequences from parquet or text files, tokenizes on-the-fly
    with optional trie dropout, and returns token IDs. Masking is applied in
    the collate function.

    Streaming mode (parquet shards):
        When ``data_path`` is a directory of ``.parquet`` files and
        ``shards_per_epoch`` is set, only *N* shards are loaded into memory
        per epoch.  Call :meth:`load_epoch` before each epoch to rotate shards.
    """

    def __init__(self, data_path: str, tokenizer, *,
                 is_training: bool = True,
                 max_tokens: Optional[int] = None,
                 trie_dropout: float = 0.0,
                 seq_column: str = "sequence",
                 lang_column: Optional[str] = None,
                 shards_per_epoch: Optional[int] = None,
                 unique_shards: bool = False,
                 seed: int = 42,
                 encoder: Optional[callable] = None,
                 residue_mlm: bool = False):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.max_tokens = max_tokens
        self.trie_dropout = trie_dropout
        self.residue_mlm = residue_mlm
        self._seq_column = seq_column
        self._lang_column = lang_column
        self._seed = seed
        self._unique = unique_shards
        self._encoder = encoder
        self._langs: Optional[List[str]] = None

        # Discover parquet shards
        self._shard_paths: List[str] = []
        if os.path.isdir(data_path):
            self._shard_paths = sorted(glob.glob(os.path.join(data_path, "*.parquet")))

        self._shards_per_epoch = shards_per_epoch
        self.streaming = bool(self._shard_paths and shards_per_epoch)

        # Pre-compute non-overlapping shard schedule when unique_shards=True
        self._shard_schedule: Optional[List[List[str]]] = None
        if self.streaming and self._unique:
            rng = random.Random(self._seed)
            shuffled = list(self._shard_paths)
            rng.shuffle(shuffled)
            n = self._shards_per_epoch
            self._shard_schedule = [
                shuffled[i:i + n] for i in range(0, len(shuffled), n)
            ]
            total_ep = len(self._shard_schedule)
            _log.info("MLMDataset: unique-shard mode — %d shards, %d/epoch, %d epochs for full pass",
                      len(self._shard_paths), n, total_ep)

        if self.streaming:
            if not self._unique:
                _log.info("MLMDataset: streaming mode — %d shards, %d per epoch from %s",
                          len(self._shard_paths), shards_per_epoch, data_path)
            self._seqs: List[str] = []
            self.load_epoch(0)
        elif self._shard_paths and self._lang_column:
            self._seqs, self._langs = self._load_all_parquet_with_lang()
            _log.info("MLMDataset: %d sequences (with language) from %s", len(self._seqs), data_path)
        else:
            self._seqs = load_sequences(data_path, seq_column)
            _log.info("MLMDataset: %d sequences from %s", len(self._seqs), data_path)

    def _load_all_parquet_with_lang(self):
        """Load all parquet shards, returning (seqs, langs) lists."""
        seqs, langs = [], []
        for path in self._shard_paths:
            df = pd.read_parquet(path)
            col = self._seq_column
            for c in [col, "raw_sequence", "sequence", "text", "code"]:
                if c in df.columns:
                    col = c
                    break
            seqs.extend(df[col].tolist())
            if self._lang_column in df.columns:
                langs.extend(df[self._lang_column].tolist())
        return seqs, langs if langs else None

    @property
    def num_epochs_for_full_pass(self) -> Optional[int]:
        """Number of epochs needed to see every shard exactly once (unique mode only)."""
        return len(self._shard_schedule) if self._shard_schedule else None

    def load_epoch(self, epoch: int) -> None:
        """Load a new subset of shards for this epoch.  No-op if not streaming."""
        if not self.streaming:
            return

        if self._shard_schedule is not None:
            idx = epoch % len(self._shard_schedule)
            selected = self._shard_schedule[idx]
        else:
            n = min(self._shards_per_epoch, len(self._shard_paths))
            rng = random.Random(self._seed + epoch)
            selected = rng.sample(self._shard_paths, n)

        seqs: List[str] = []
        langs: List[str] = [] if self._lang_column else None
        for path in selected:
            df = pd.read_parquet(path)
            col = self._seq_column
            for c in [col, "raw_sequence", "sequence", "text", "code"]:
                if c in df.columns:
                    col = c
                    break
            seqs.extend(df[col].tolist())
            if langs is not None and self._lang_column in df.columns:
                langs.extend(df[self._lang_column].tolist())
        self._seqs = seqs
        self._langs = langs
        _log.info("Epoch %d: loaded %d shards (%d sequences)", epoch, len(selected), len(self._seqs))

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int):
        dropout = self.trie_dropout if self.is_training else 0.0
        seq = self._seqs[idx]

        # Residue-level MLM path: need token_lengths + aa_labels
        if self.residue_mlm and self._encoder is None and hasattr(self.tokenizer, '_segment'):
            text = self.tokenizer._ws.sub("", seq).upper()
            symbols = list(text)
            segments = self.tokenizer._segment(symbols, dropout)
            if self.max_tokens and len(segments) > self.max_tokens:
                segments = segments[:self.max_tokens]
            token_ids = [tid for _, tid in segments]
            token_lengths = [length for length, _ in segments]
            n_residues = sum(token_lengths)
            aa_labels = [
                AA_TO_IDX.get(AA_SANITIZE.get(c, c), 0)
                for c in text[:n_residues]
            ]
            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
                "aa_labels": torch.tensor(aa_labels, dtype=torch.long),
                "global_indices": torch.arange(n_residues, dtype=torch.long),
            }

        # Standard token-only path
        if self._encoder is not None and self._langs is not None:
            lang = self._langs[idx]
            enc_fn = lambda src: self._encoder(src, lang)
            ids = self.tokenizer.encode(seq, dropout=dropout, encoder=enc_fn)
        else:
            ids = self.tokenizer.encode(seq, dropout=dropout,
                                        encoder=self._encoder)
        if self.max_tokens and len(ids) > self.max_tokens:
            ids = ids[:self.max_tokens]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def mlm_collate_fn(batch: List[Dict], pad_id: int = 0, mask_id: int = 4,
                   vocab_size: int = 32000, mask_prob: float = 0.15):
    """Collate with BERT-style masking: 80% [MASK], 10% random, 10% keep."""
    ids = pad_sequence([b["input_ids"] for b in batch],
                       batch_first=True, padding_value=pad_id)
    attn_mask = ids != pad_id
    masked_ids = ids.clone()
    token_labels = torch.full_like(ids, -100)

    if mask_prob > 0:
        prob = torch.full(ids.shape, mask_prob).masked_fill_(~attn_mask, 0.0)
        to_mask = torch.bernoulli(prob).bool()

        replace = torch.bernoulli(torch.full(ids.shape, 0.8)).bool() & to_mask
        masked_ids[replace] = mask_id
        rand_pos = torch.bernoulli(torch.full(ids.shape, 0.5)).bool() & to_mask & ~replace
        masked_ids[rand_pos] = torch.randint(2, vocab_size, ids.shape)[rand_pos]

        token_labels[to_mask] = ids[to_mask]

    out = {
        "input_ids": masked_ids,
        "attention_mask": attn_mask,
        "token_labels": token_labels,
    }

    # Residue-level fields (present when residue_mlm=True in dataset)
    if "token_lengths" in batch[0]:
        out["token_lengths"] = pad_sequence(
            [b["token_lengths"] for b in batch], batch_first=True, padding_value=0)
        out["global_indices"] = pad_sequence(
            [b["global_indices"] for b in batch], batch_first=True, padding_value=0)
        out["aa_labels"] = pad_sequence(
            [b["aa_labels"] for b in batch], batch_first=True, padding_value=-100)

    return out


# ---------------------------------------------------------------------------
# Protein-specific MPM dataset (kept for biology experiments)
# ---------------------------------------------------------------------------

AA_TO_IDX = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7,
    "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
}
AA_SANITIZE = {"B": "D", "J": "L", "O": "K", "U": "C", "X": "A", "Z": "E"}


class ProteinMPMDataset(Dataset):
    """Protein masked profile modeling dataset.

    Expects:
      - sequences_path: parquet with columns [hmm_id, raw_sequence, alignment_indices]
      - emissions_path: parquet with columns [hmm_id, emission_matrix]
    """

    def __init__(self, sequences_path: str, emissions_path: str, tokenizer,
                 is_training: bool = True, hmm_ids: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None, trie_dropout: float = 0.0):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.max_tokens = max_tokens
        self.trie_dropout = trie_dropout
        self._background = [0.0] * 20

        df = load_parquet(sequences_path, columns=["hmm_id", "raw_sequence", "alignment_indices"])
        if hmm_ids is not None:
            df = df[df["hmm_id"].isin(set(hmm_ids))].reset_index(drop=True)

        em_df = load_parquet(emissions_path, columns=["hmm_id", "emission_matrix"])
        self.emissions: Dict[str, list] = em_df.set_index("hmm_id")["emission_matrix"].to_dict()

        self._ids = df["hmm_id"].to_numpy()
        self._seqs = df["raw_sequence"].to_numpy()
        self._alns = df["alignment_indices"].tolist()
        _log.info("ProteinMPMDataset: %d samples", len(df))

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int):
        hmm_id = self._ids[idx]
        raw_seq = self._seqs[idx]
        aln_idx = self._alns[idx]
        if not isinstance(aln_idx, list):
            aln_idx = list(aln_idx)

        emissions = self.emissions[hmm_id]
        dropout = self.trie_dropout if self.is_training else 0.0

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


def mpm_collate_fn(batch: List[Dict], pad_id: int = 0, mask_id: int = 4,
                   vocab_size: int = 32000, mask_prob: float = 0.15):
    """Collate for MPM: pads all fields including emission targets and AA labels."""
    ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    tgt = pad_sequence([b["targets"] for b in batch], batch_first=True, padding_value=0.0)
    gi = pad_sequence([b["global_indices"] for b in batch], batch_first=True, padding_value=0)
    tl = pad_sequence([b["token_lengths"] for b in batch], batch_first=True, padding_value=0)
    tm = pad_sequence([b["target_mask"] for b in batch], batch_first=True, padding_value=0.0)
    aa = pad_sequence([b["aa_labels"] for b in batch], batch_first=True, padding_value=-100)

    attn_mask = ids != pad_id
    masked_ids = ids.clone()
    token_labels = torch.full_like(ids, -100)

    if mask_prob > 0:
        prob = torch.full(ids.shape, mask_prob).masked_fill_(~attn_mask, 0.0)
        to_mask = torch.bernoulli(prob).bool()

        replace = torch.bernoulli(torch.full(ids.shape, 0.8)).bool() & to_mask
        masked_ids[replace] = mask_id
        rand_pos = torch.bernoulli(torch.full(ids.shape, 0.5)).bool() & to_mask & ~replace
        masked_ids[rand_pos] = torch.randint(2, vocab_size, ids.shape)[rand_pos]

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
