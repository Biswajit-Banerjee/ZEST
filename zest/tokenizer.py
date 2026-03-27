"""
ZEST Tokenizer: Zoned Encoding of Sequence Traits.

A domain-aware tokenizer that uses greedy max-match over a trie of
recurring patterns (traits) mined from domain data. Long patterns are
prioritized; unmatched regions fall back to base alphabet characters.
"""
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import torch


class _TrieNode:
    __slots__ = ["children", "token_id"]

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.token_id: int = -1


class ZESTTokenizer:
    """Fuzzy Greedy MaxMatch tokenizer backed by a prefix trie."""

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<EOS>", "<MASK>"]
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, vocab_path: str, verbose: bool = False):
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._root = _TrieNode()
        self._ws = re.compile(r"\s+")

        # Reserve indices 0-4 for special tokens.
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.vocab[tok] = i
            self.id_to_token[i] = tok

        # Indices 5-24: base alphabet (amino acids for biology).
        offset = len(self.SPECIAL_TOKENS)
        for i, aa in enumerate(self.AMINO_ACIDS):
            tid = offset + i
            self.vocab[aa] = tid
            self.id_to_token[tid] = aa
            self._insert(aa, tid)

        # BPE clusters: centroid → [member_1, member_2, ...]
        with open(vocab_path) as f:
            clusters: Dict[str, List[str]] = json.load(f)

        offset = len(self.vocab)
        for i, (centroid, members) in enumerate(clusters.items()):
            tid = offset + i
            self.id_to_token[tid] = centroid
            self.vocab[centroid] = tid
            self._insert(centroid, tid)
            for member in members:
                self.vocab[member] = tid
                self._insert(member, tid)

        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
        self.cls_id = self.vocab["<CLS>"]
        self.eos_id = self.vocab["<EOS>"]
        self.mask_id = self.vocab["<MASK>"]
        self.vocab_size = len(self.id_to_token)

        if verbose:
            print(f"ZEST tokenizer ready — vocab size: {self.vocab_size}")

    # ------------------------------------------------------------------
    # Trie operations
    # ------------------------------------------------------------------

    def _insert(self, token_str: str, token_id: int) -> None:
        node = self._root
        for ch in token_str:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.token_id = token_id

    def _match(self, text: str, start: int) -> List[Tuple[int, int]]:
        """Return all valid (length, token_id) pairs starting at *start*, longest first."""
        matches: List[Tuple[int, int]] = []
        node = self._root
        for i in range(start, len(text)):
            ch = text[i]
            if ch not in node.children:
                break
            node = node.children[ch]
            if node.token_id != -1:
                matches.append((i - start + 1, node.token_id))
        return matches[::-1]

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _segment(self, text: str, dropout: float = 0.0) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        i, n = 0, len(text)
        while i < n:
            matches = self._match(text, i)
            if not matches:
                segments.append((1, self.unk_id))
                i += 1
                continue
            length, tid = matches[0]
            if dropout > 0 and random.random() < dropout and len(matches) > 1:
                idx = random.randint(1, len(matches) - 1)
                length, tid = matches[idx]
            segments.append((length, tid))
            i += length
        return segments

    def encode(
        self,
        text: str,
        dropout: float = 0.0,
        add_special_tokens: bool = False,
    ) -> List[int]:
        text = self._ws.sub("", text).upper()
        ids = [tid for _, tid in self._segment(text, dropout)]
        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        skip = {self.pad_id, self.cls_id, self.eos_id, self.mask_id}
        parts = []
        for i in ids:
            if skip_special and i in skip:
                continue
            parts.append(self.id_to_token.get(i, ""))
        return "".join(parts)

    def batch_encode_plus(
        self,
        sequences: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        dropout: float = 0.0,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch = [self.encode(s, dropout=dropout) for s in sequences]
        n_special = 2 if add_special_tokens else 0

        processed = []
        for ids in batch:
            if max_length and truncation and len(ids) + n_special > max_length:
                ids = ids[: max_length - n_special]
            if add_special_tokens:
                ids = [self.cls_id] + ids + [self.eos_id]
            processed.append(ids)

        if padding:
            target = max_length or max(len(ids) for ids in processed)
            padded, masks = [], []
            for ids in processed:
                pad_n = max(0, target - len(ids))
                padded.append(ids + [self.pad_id] * pad_n)
                masks.append([1] * len(ids) + [0] * pad_n)
        else:
            padded, masks = processed, [[1] * len(ids) for ids in processed]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        # Re-extract cluster data from vocab (centroids only).
        clusters: Dict[str, List[str]] = {}
        offset = len(self.SPECIAL_TOKENS) + len(self.AMINO_ACIDS)
        centroid_ids = {tid for tid in self.id_to_token if tid >= offset}
        for tok, tid in self.vocab.items():
            if tid not in centroid_ids or tok in self.id_to_token.values():
                continue
            centroid = self.id_to_token[tid]
            clusters.setdefault(centroid, []).append(tok)
        with open(os.path.join(directory, "vocab_map.json"), "w") as f:
            json.dump(clusters, f)

    @classmethod
    def from_pretrained(cls, directory: str) -> "ZESTTokenizer":
        return cls(os.path.join(directory, "vocab_map.json"))
