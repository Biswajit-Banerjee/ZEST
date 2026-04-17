"""
Hierarchical CATH pair dataset for contrastive fine-tuning.

Each sample yields an anchor/positive protein pair with hierarchy labels at
Architecture (C.A), Fold/Topology (C.A.T), and Superfamily (C.A.T.H) levels.
"""
import glob
import logging
import os
import pickle
import random
import time
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from zest.data.pretrain import AA_TO_IDX, AA_SANITIZE

_log = logging.getLogger(__name__)


def _tokenize_chunk(args):
    """Worker function for multiprocessing pre-tokenisation.

    Each worker reconstructs a lightweight tokenizer from ``tok_path`` so
    the trie lives in the child process (no pickle of the full object).
    """
    chunk, tok_path, max_tokens, dropout, seed, encoder_cfg = args
    random.seed(seed)

    cfg = encoder_cfg or {}
    tok_type = cfg.get("tokenizer_type", "zest")
    domain = cfg.get("domain", "protein")

    if tok_type == "bpe":
        if domain == "code":
            from experiments.code.build_bpe_tokenizer import CodeBPETokenizer, make_bpe_encoder
            tok = CodeBPETokenizer(tok_path)
            encoder = make_bpe_encoder(tok_path, cfg.get("language", "c"))
        else:
            from experiments.biology.build_protein_bpe_tokenizer import ProteinBPETokenizer
            tok = ProteinBPETokenizer(tok_path)
            encoder = None
    elif tok_type == "char":
        from experiments.biology.char_tokenizer import CharTokenizer
        tok = CharTokenizer()
        encoder = None
    else:
        from zest.tokenizer import ZESTTokenizer
        tok = ZESTTokenizer(tok_path)
        encoder = None
        if domain == "code" and "language" in cfg:
            from experiments.code.build_role_tokenizer import make_role_encoder
            encoder = make_role_encoder(cfg["language"])

    residue_mlm = cfg.get("residue_mlm", False)

    results = []
    for idx, seq in chunk:
        if encoder is not None:
            result = encoder(seq)
            if result and isinstance(result[0], int):
                ids = result
            else:
                ids = tok.encode_symbols(result, dropout=dropout) if hasattr(tok, 'encode_symbols') else tok.encode(seq, dropout=dropout)
            if max_tokens and len(ids) > max_tokens:
                ids = ids[:max_tokens]
            results.append((idx, ids))
        elif residue_mlm and hasattr(tok, '_segment'):
            text = tok._ws.sub("", seq).upper()
            symbols = list(text)
            segments = tok._segment(symbols, dropout)
            if max_tokens and len(segments) > max_tokens:
                segments = segments[:max_tokens]
            token_ids = [tid for _, tid in segments]
            token_lengths = [length for length, _ in segments]
            n_residues = sum(token_lengths)
            aa_labels = [AA_TO_IDX.get(AA_SANITIZE.get(c, c), 0) for c in text[:n_residues]]
            results.append((idx, {"ids": token_ids, "token_lengths": token_lengths, "aa_labels": aa_labels}))
        else:
            ids = tok.encode(seq, dropout=dropout)
            if max_tokens and len(ids) > max_tokens:
                ids = ids[:max_tokens]
            results.append((idx, ids))
    return results


class HierarchicalPairDataset(Dataset):
    """
    Map-style pair dataset with CATH hierarchy labels.

    All pair metadata is pre-loaded into compact numpy arrays at init.
    Sequences are pre-tokenized at each epoch start to refresh BPE dropout;
    the token cache is shared across DataLoader workers via fork (COW).

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
                 shards_per_epoch: Optional[int] = None, seed: int = 42,
                 bpe_dropout: float = 0.0, encoder=None,
                 encoder_cfg: Optional[Dict] = None, residue_mlm: bool = False,
                 crop_ratio: float = 0.0, crop_prob: float = 0.0,
                 crop_small_threshold: int = 50,
                 **kwargs):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.seed = seed
        self.bpe_dropout = bpe_dropout
        self._encoder = encoder
        self._encoder_cfg = encoder_cfg or {}
        self.residue_mlm = residue_mlm
        if residue_mlm:
            self._encoder_cfg["residue_mlm"] = True
        self.crop_ratio = crop_ratio
        self.crop_prob = crop_prob
        self.crop_small_threshold = crop_small_threshold
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

        # Pre-load all pair metadata into compact numpy arrays
        self._preload_pairs()

        # Fold (topology) frequency per pair — for balanced sampling & crop
        from collections import Counter
        fold_counts = Counter(self._t_labels.tolist())
        self._pair_fold_sizes = np.array(
            [fold_counts[t] for t in self._t_labels], dtype=np.int32)
        n_small = (self._pair_fold_sizes < self.crop_small_threshold).sum()

        # Active indices (subset when shards_per_epoch < num_shards)
        self._active = np.arange(len(self._q_idx))
        self._token_cache = [None] * len(self._sequences)

        _log.info("HierarchicalPairDataset: %d shards, %d valid pairs, "
                   "%d unique seqs, %d A / %d T / %d H labels, "
                   "%d pairs from small folds (<%d)",
                   self.num_shards, len(self._q_idx), len(self._sequences),
                   len(self.a_map), len(self.t_map), len(self.h_map),
                   n_small, self.crop_small_threshold)

        # Pre-tokenize for first epoch so DataLoader can start immediately
        self.set_epoch(0)

    # ------------------------------------------------------------------
    # One-time pair pre-loading (vectorised pandas, per-shard streaming)
    # ------------------------------------------------------------------

    def _preload_pairs(self):
        """Load all pair metadata from parquet into numpy arrays."""
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()

        sorted_keys = sorted(self.seq_map.keys())
        sid_to_idx = {s: i for i, s in enumerate(sorted_keys)}
        self._sequences = [self.seq_map[s] for s in sorted_keys]

        # Pre-compute cath_id → label dicts (C-level hash lookup, no lambdas)
        cath_a = {c: self.a_map.get('.'.join(c.split('.')[:2]), -1) for c in self.h_map}
        cath_t = {c: self.t_map.get('.'.join(c.split('.')[:3]), -1) for c in self.h_map}
        cath_h = dict(self.h_map)

        def _load_shard(args):
            i, path = args
            df = pd.read_parquet(path, columns=['Query_ID', 'Template_ID', 'CATH_query', 'CATH_template', 'Probability'])
            df['_q'] = df['Query_ID'].map(sid_to_idx)
            df['_t'] = df['Template_ID'].map(sid_to_idx)
            df = df.dropna(subset=['_q', '_t'])

            cath_q = df['CATH_query']
            cath_templ = df['CATH_template']
            a_q = cath_q.map(cath_a).fillna(-1).astype(np.int32)
            t_q = cath_q.map(cath_t).fillna(-1).astype(np.int32)
            h_q = cath_q.map(cath_h).fillna(-1).astype(np.int32)
            a_t = cath_templ.map(cath_a).fillna(-1).astype(np.int32)
            t_t = cath_templ.map(cath_t).fillna(-1).astype(np.int32)
            h_t = cath_templ.map(cath_h).fillna(-1).astype(np.int32)
            p_v = df['Probability'].astype(np.float32)

            mask = (h_q >= 0) & (h_t >= 0)
            return (
                df.loc[mask, '_q'].to_numpy(dtype=np.int32),
                df.loc[mask, '_t'].to_numpy(dtype=np.int32),
                a_q[mask].to_numpy(), t_q[mask].to_numpy(), h_q[mask].to_numpy(),
                a_t[mask].to_numpy(), t_t[mask].to_numpy(), h_t[mask].to_numpy(),
                p_v[mask].to_numpy(), np.full(int(mask.sum()), i, dtype=np.int16),
            )

        workers = min(8, self.num_shards)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_load_shard, enumerate(self.shard_paths)))

        self._q_idx = np.concatenate([r[0] for r in results])
        self._t_idx = np.concatenate([r[1] for r in results])
        self._a_labels = np.concatenate([r[2] for r in results])
        self._t_labels = np.concatenate([r[3] for r in results])
        self._h_labels = np.concatenate([r[4] for r in results])
        self._a_pos_labels = np.concatenate([r[5] for r in results])
        self._t_pos_labels = np.concatenate([r[6] for r in results])
        self._h_pos_labels = np.concatenate([r[7] for r in results])
        self._probabilities = np.concatenate([r[8] for r in results])
        self._shard_ids = np.concatenate([r[9] for r in results])

        _log.info("Pre-loaded %d pairs from %d shards in %.1fs (%d threads)",
                  len(self._q_idx), self.num_shards, time.time() - t0, workers)

    # ------------------------------------------------------------------
    # Per-epoch pre-tokenisation (parallel, preserves BPE dropout)
    # ------------------------------------------------------------------

    def _retokenize(self, seq_indices: np.ndarray):
        """Tokenize unique sequences in parallel using multiprocessing."""
        t0 = time.time()
        unique_idx = np.unique(seq_indices)
        seqs_to_tok = [(int(i), self._sequences[i]) for i in unique_idx]

        workers = min(os.cpu_count() or 4, max(1, len(seqs_to_tok) // 500))
        workers = min(workers, 16)

        dropout = self.bpe_dropout if self.bpe_dropout > 0 else 0.0
        tok_path = self.tokenizer.path if hasattr(self.tokenizer, 'path') else None

        if workers > 1 and tok_path and len(seqs_to_tok) > 1000:
            # Split list of tuples into roughly equal chunks
            k, rem = divmod(len(seqs_to_tok), workers)
            chunks = []
            start = 0
            for i in range(workers):
                end = start + k + (1 if i < rem else 0)
                chunks.append(seqs_to_tok[start:end])
                start = end
            args = [
                (chunk, tok_path, self.max_tokens, dropout,
                 self.seed + self._epoch + i, self._encoder_cfg)
                for i, chunk in enumerate(chunks)
            ]
            with Pool(workers) as pool:
                results = pool.map(_tokenize_chunk, args)
            for partial in results:
                for idx, ids in partial:
                    self._token_cache[idx] = ids
        else:
            for idx, seq in seqs_to_tok:
                if self._encoder is not None:
                    ids = self._encoder(seq)
                else:
                    ids = self.tokenizer.encode(seq, dropout=dropout)
                if self.max_tokens and len(ids) > self.max_tokens:
                    ids = ids[:self.max_tokens]
                self._token_cache[idx] = ids

        _log.info("Pre-tokenized %d unique sequences in %.1fs (%d workers, epoch %d)",
                  len(unique_idx), time.time() - t0, workers, self._epoch)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._active)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        if self.shards_per_epoch < self.num_shards:
            rng = random.Random(self.seed + epoch)
            selected = set(rng.sample(range(self.num_shards), self.shards_per_epoch))
            self._active = np.where(np.isin(self._shard_ids, list(selected)))[0]
        else:
            self._active = np.arange(len(self._q_idx))

        # Pre-tokenize all sequences needed this epoch
        active_q = self._q_idx[self._active]
        active_t = self._t_idx[self._active]
        needed = np.concatenate([active_q, active_t])
        self._token_cache = [None] * len(self._sequences)
        self._retokenize(needed)

    # ------------------------------------------------------------------
    # Fold-balanced sampling
    # ------------------------------------------------------------------

    def get_sample_weights(self, min_fold_freq: int = 10):
        """Per-pair weights for fold-balanced sampling (clamped inverse sqrt).

        Folds below ``min_fold_freq`` pairs are clamped so singletons and
        near-singletons don't get extreme oversampling that leads to
        overfitting on the same few pairs.
        """
        from collections import Counter
        active_folds = self._t_labels[self._active]
        fold_counts = Counter(active_folds.tolist())
        weights = np.array([1.0 / np.sqrt(max(fold_counts[t], min_fold_freq))
                            for t in active_folds], dtype=np.float64)
        weights /= weights.sum()
        return torch.from_numpy(weights).double()

    # ------------------------------------------------------------------
    # Random crop augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _random_crop(entry, min_ratio=0.7):
        """Random contiguous crop of a token sequence (list or residue dict)."""
        if isinstance(entry, dict):
            ids = entry["ids"]
            tl = entry["token_lengths"]
            aa = entry["aa_labels"]
            n = len(ids)
            if n <= 4:
                return entry
            crop_n = random.randint(max(2, int(n * min_ratio)), n)
            start = random.randint(0, n - crop_n)
            new_tl = tl[start:start + crop_n]
            res_start = sum(tl[:start])
            res_end = res_start + sum(new_tl)
            return {
                "ids": ids[start:start + crop_n],
                "token_lengths": new_tl,
                "aa_labels": aa[res_start:res_end],
            }
        else:
            n = len(entry)
            if n <= 4:
                return entry
            crop_n = random.randint(max(2, int(n * min_ratio)), n)
            start = random.randint(0, n - crop_n)
            return entry[start:start + crop_n]

    def _should_crop(self, real_idx):
        """Decide whether to crop; boost for small-but-not-singleton folds.

        Singletons (≤2 pairs) get base crop rate — oversampling them is
        already risky, aggressive cropping on top would overfit.
        Folds with 3–threshold pairs get 3× boost — enough pairs to
        benefit from view diversity but sparse enough to need it.
        """
        fold_size = self._pair_fold_sizes[real_idx]
        if fold_size <= 2:
            return random.random() < self.crop_prob
        if fold_size < self.crop_small_threshold:
            return random.random() < min(self.crop_prob * 3, 0.9)
        return random.random() < self.crop_prob

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        real = int(self._active[idx])
        cache_q = self._token_cache[self._q_idx[real]]
        cache_t = self._token_cache[self._t_idx[real]]

        # Random crop augmentation
        if self.crop_prob > 0 and self._should_crop(real):
            cache_q = self._random_crop(cache_q, self.crop_ratio)
            cache_t = self._random_crop(cache_t, self.crop_ratio)

        if isinstance(cache_q, dict):
            item = {
                "anchor_ids": cache_q["ids"],
                "positive_ids": cache_t["ids"],
                "anchor_token_lengths": cache_q["token_lengths"],
                "anchor_aa_labels": cache_q["aa_labels"],
                "positive_token_lengths": cache_t["token_lengths"],
                "positive_aa_labels": cache_t["aa_labels"],
            }
        else:
            item = {
                "anchor_ids": cache_q,
                "positive_ids": cache_t,
            }

        item.update({
            "a_label": int(self._a_labels[real]),
            "t_label": int(self._t_labels[real]),
            "h_label": int(self._h_labels[real]),
            "a_pos_label": int(self._a_pos_labels[real]),
            "t_pos_label": int(self._t_pos_labels[real]),
            "h_pos_label": int(self._h_pos_labels[real]),
            "probability": float(self._probabilities[real]),
        })
        return item


def finetune_collate_fn(batch: List[Dict], pad_id: int = 0,
                        mask_id: int = 4, vocab_size: int = 32000,
                        mask_prob: float = 0.0):
    """Collate anchor/positive pairs into padded tensors with optional light masking."""
    def _pad(seqs):
        max_len = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return ids, mask

    def _mask_tokens(ids, attn_mask):
        """BERT-style light masking for auxiliary MLM."""
        masked = ids.clone()
        labels = torch.full_like(ids, -100)
        prob = torch.full(ids.shape, mask_prob).masked_fill_(~attn_mask.bool(), 0.0)
        to_mask = torch.bernoulli(prob).bool()
        replace = torch.bernoulli(torch.full(ids.shape, 0.8)).bool() & to_mask
        masked[replace] = mask_id
        rand_pos = torch.bernoulli(torch.full(ids.shape, 0.5)).bool() & to_mask & ~replace
        masked[rand_pos] = torch.randint(2, vocab_size, ids.shape)[rand_pos]
        labels[to_mask] = ids[to_mask]
        return masked, labels

    anc_ids, anc_mask = _pad([b["anchor_ids"] for b in batch])
    pos_ids, pos_mask = _pad([b["positive_ids"] for b in batch])

    if mask_prob > 0:
        anc_ids, anc_tok_labels = _mask_tokens(anc_ids, anc_mask)
        pos_ids, pos_tok_labels = _mask_tokens(pos_ids, pos_mask)

    out = {
        "anchor_ids": anc_ids,
        "anchor_mask": anc_mask,
        "positive_ids": pos_ids,
        "positive_mask": pos_mask,
        "a_labels": torch.tensor([b["a_label"] for b in batch], dtype=torch.long),
        "t_labels": torch.tensor([b["t_label"] for b in batch], dtype=torch.long),
        "h_labels": torch.tensor([b["h_label"] for b in batch], dtype=torch.long),
        "a_pos_labels": torch.tensor([b["a_pos_label"] for b in batch], dtype=torch.long),
        "t_pos_labels": torch.tensor([b["t_pos_label"] for b in batch], dtype=torch.long),
        "h_pos_labels": torch.tensor([b["h_pos_label"] for b in batch], dtype=torch.long),
        "probabilities": torch.tensor([b["probability"] for b in batch], dtype=torch.float32),
    }

    if mask_prob > 0:
        out["anchor_token_labels"] = anc_tok_labels
        out["positive_token_labels"] = pos_tok_labels

    if "anchor_token_lengths" in batch[0]:
        for prefix, key_prefix in [("anchor", "anchor"), ("positive", "positive")]:
            tl = pad_sequence(
                [torch.tensor(b[f"{prefix}_token_lengths"], dtype=torch.long) for b in batch],
                batch_first=True, padding_value=0)
            aa = pad_sequence(
                [torch.tensor(b[f"{prefix}_aa_labels"], dtype=torch.long) for b in batch],
                batch_first=True, padding_value=-100)
            B = tl.shape[0]
            max_res = int(tl.sum(dim=1).max().item())
            gi = torch.zeros(B, max_res, dtype=torch.long)
            for i in range(B):
                n = int(tl[i].sum().item())
                gi[i, :n] = torch.arange(n)
            out[f"{key_prefix}_token_lengths"] = tl
            out[f"{key_prefix}_aa_labels"] = aa
            out[f"{key_prefix}_global_indices"] = gi

    return out
