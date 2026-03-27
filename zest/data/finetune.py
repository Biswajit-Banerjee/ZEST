"""
Hierarchical CATH pair dataset for contrastive fine-tuning.

Each sample yields an anchor/positive protein pair with hierarchy labels at
Architecture (C.A), Topology (C.A.T), and Superfamily (C.A.T.H) levels.
Pairs are streamed from sharded parquet files for memory efficiency.
"""
import glob
import logging
import os
import pickle
import random
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import IterableDataset

_log = logging.getLogger(__name__)


class HierarchicalPairDataset(IterableDataset):
    """
    Sharded pair dataset with CATH hierarchy labels.

    Directory layout::

        pair_shards/
            seq_map.pkl          # {seq_id: raw_sequence}
            cath_index.pkl       # {cath_id: [seq_ids]}
            train/*.parquet      # (Query_ID, Template_ID, CATH_query, CATH_template)
            val/*.parquet

    Each parquet shard contains protein pairs. The CATH ID ``C.A.T.H`` is
    decomposed into three hierarchy levels for multi-level contrastive learning.
    """

    def __init__(self, shards_dir: str, tokenizer, max_tokens: int = 512,
                 shards_per_epoch: Optional[int] = None,
                 shuffle_buffer: int = 50_000, seed: int = 42,
                 bpe_dropout: float = 0.0):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.bpe_dropout = bpe_dropout
        self._epoch = 0

        self.shard_paths = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No parquet shards in {shards_dir}")
        self.num_shards = len(self.shard_paths)
        self.shards_per_epoch = shards_per_epoch or self.num_shards

        parent = os.path.dirname(shards_dir.rstrip("/"))
        with open(os.path.join(parent, "seq_map.pkl"), "rb") as f:
            self.seq_map: Dict[str, str] = pickle.load(f)
        with open(os.path.join(parent, "cath_index.pkl"), "rb") as f:
            cath_to_seqs: Dict[str, List[str]] = pickle.load(f)

        # Build integer label maps for each hierarchy level
        a_set, t_set, h_set = set(), set(), set()
        for cid in sorted(cath_to_seqs.keys()):
            parts = cid.split(".")
            if len(parts) >= 4:
                a_set.add(".".join(parts[:2]))
                t_set.add(".".join(parts[:3]))
                h_set.add(cid)
        self.a_map = {s: i for i, s in enumerate(sorted(a_set))}
        self.t_map = {s: i for i, s in enumerate(sorted(t_set))}
        self.h_map = {s: i for i, s in enumerate(sorted(h_set))}

        _log.info("HierarchicalPairDataset: %d shards, %d sequences, "
                   "%d A / %d T / %d H labels",
                   self.num_shards, len(self.seq_map),
                   len(self.a_map), len(self.t_map), len(self.h_map))

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _tokenize(self, seq: str) -> List[int]:
        dropout = self.bpe_dropout if self.bpe_dropout > 0 else 0.0
        ids = self.tokenizer.encode(seq, dropout=dropout)
        if self.max_tokens and len(ids) > self.max_tokens:
            ids = ids[:self.max_tokens]
        return ids

    def _cath_labels(self, cath_id: str):
        parts = cath_id.split(".")
        a_key = ".".join(parts[:2])
        t_key = ".".join(parts[:3])
        return (self.a_map.get(a_key, -1),
                self.t_map.get(t_key, -1),
                self.h_map.get(cath_id, -1))

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(self.num_shards))
        rng.shuffle(indices)
        selected = indices[:self.shards_per_epoch]

        buffer = []
        for shard_idx in selected:
            df = pd.read_parquet(self.shard_paths[shard_idx])
            for _, row in df.iterrows():
                qid, tid = row["Query_ID"], row["Template_ID"]
                cq, ct = row["CATH_query"], row["CATH_template"]

                seq_q = self.seq_map.get(qid)
                seq_t = self.seq_map.get(tid)
                if seq_q is None or seq_t is None:
                    continue

                a_q, t_q, h_q = self._cath_labels(cq)
                if h_q < 0:
                    continue

                anc_ids = self._tokenize(seq_q)
                pos_ids = self._tokenize(seq_t)

                sample = {
                    "anchor_ids": anc_ids,
                    "positive_ids": pos_ids,
                    "a_label": a_q,
                    "t_label": t_q,
                    "h_label": h_q,
                }
                buffer.append(sample)

                if len(buffer) >= self.shuffle_buffer:
                    rng.shuffle(buffer)
                    yield from buffer
                    buffer.clear()

        if buffer:
            rng.shuffle(buffer)
            yield from buffer


def finetune_collate_fn(batch: List[Dict], pad_id: int = 0):
    """Collate anchor/positive pairs into padded tensors."""
    def _pad(seqs):
        max_len = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return ids, mask

    anc_ids, anc_mask = _pad([b["anchor_ids"] for b in batch])
    pos_ids, pos_mask = _pad([b["positive_ids"] for b in batch])

    return {
        "anchor_ids": anc_ids,
        "anchor_mask": anc_mask,
        "positive_ids": pos_ids,
        "positive_mask": pos_mask,
        "a_labels": torch.tensor([b["a_label"] for b in batch], dtype=torch.long),
        "t_labels": torch.tensor([b["t_label"] for b in batch], dtype=torch.long),
        "h_labels": torch.tensor([b["h_label"] for b in batch], dtype=torch.long),
    }
