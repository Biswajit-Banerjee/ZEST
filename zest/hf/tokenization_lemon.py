"""Self-contained Hugging Face PreTrainedTokenizer for ZEST protein motif vocabularies.

Uses the compiled Rust trie backend (`zest_tokenizer.RustZESTTokenizer`) automatically
when installed, and falls back transparently to a pure-Python trie with 100% bit-exact
output parity when the Rust wheel is absent.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

try:
    from zest_tokenizer import RustZESTTokenizer as _RustBackend
except ImportError:
    _RustBackend = None


VOCAB_FILES_NAMES = {"vocab_file": "vocab_map.json"}
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<EOS>", "<MASK>"]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: Dict[str, "_TrieNode"] = {}
        self.token_id: int = -1


class LemonTokenizer(PreTrainedTokenizer):
    """ZEST greedy longest-match protein motif tokenizer for Hugging Face Transformers."""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        cls_token: str = "<CLS>",
        eos_token: str = "<EOS>",
        mask_token: str = "<MASK>",
        model_max_length: int = 1024,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self._ws = re.compile(r"\s+")
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._root = _TrieNode()
        self._rust = None

        # Special tokens (IDs 0..4)
        for i, tok in enumerate(SPECIAL_TOKENS):
            self._vocab[tok] = i
            self._id_to_token[i] = tok

        # Base amino acids (IDs 5..24)
        offset = len(SPECIAL_TOKENS)
        for i, aa in enumerate(AMINO_ACIDS):
            tid = offset + i
            self._vocab[aa] = tid
            self._id_to_token[tid] = aa
            self._trie_insert(aa, tid)

        # Many-to-one clusters: {centroid: [members]}
        with open(vocab_file, "r", encoding="utf-8") as f:
            clusters: Dict[str, List[str]] = json.load(f)

        offset = len(self._id_to_token)
        for i, (centroid, members) in enumerate(clusters.items()):
            tid = offset + i
            self._id_to_token[tid] = centroid
            self._vocab[centroid] = tid
            self._trie_insert(centroid, tid)
            for member in (members or []):
                if member == centroid:
                    continue
                self._vocab[member] = tid
                self._trie_insert(member, tid)

        self._vocab_size = len(self._id_to_token)
        self._mask_id = self._vocab["<MASK>"]

        if _RustBackend is not None:
            self._rust = _RustBackend(vocab_file)

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def is_rust_backed(self) -> bool:
        return self._rust is not None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        # Return canonical 1-to-1 centroid mapping so len(get_vocab()) == vocab_size
        return {tok: tid for tid, tok in self._id_to_token.items()}

    def _trie_insert(self, pattern: str, token_id: int) -> None:
        node = self._root
        for ch in pattern:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _TrieNode()
                node.children[ch] = nxt
            node = nxt
        node.token_id = token_id

    def _py_segment(self, text: str, dropout: float = 0.0) -> Tuple[List[int], List[int]]:
        clean = self._ws.sub("", text).upper()
        n = len(clean)
        ids: List[int] = []
        lens: List[int] = []
        i = 0
        if dropout <= 0.0:
            while i < n:
                node = self._root
                best_len = 0
                best_tid = -1
                j = i
                while j < n:
                    nxt = node.children.get(clean[j])
                    if nxt is None:
                        break
                    node = nxt
                    if node.token_id != -1:
                        best_len = j - i + 1
                        best_tid = node.token_id
                    j += 1
                if best_tid != -1:
                    ids.append(best_tid)
                    lens.append(best_len)
                    i += best_len
                else:
                    ids.append(self._mask_id)
                    lens.append(1)
                    i += 1
        else:
            while i < n:
                matches: List[Tuple[int, int]] = []
                node = self._root
                j = i
                while j < n:
                    nxt = node.children.get(clean[j])
                    if nxt is None:
                        break
                    node = nxt
                    if node.token_id != -1:
                        matches.append((j - i + 1, node.token_id))
                    j += 1
                if not matches:
                    ids.append(self._mask_id)
                    lens.append(1)
                    i += 1
                else:
                    if len(matches) > 1 and random.random() < dropout:
                        length, tid = matches[random.randint(0, len(matches) - 2)]
                    else:
                        length, tid = matches[-1]
                    ids.append(tid)
                    lens.append(length)
                    i += length
        return ids, lens

    def segment(self, text: str, dropout: float = 0.0) -> Tuple[List[int], List[int]]:
        """Return ``(token_ids, token_lengths)`` for a single sequence."""
        if self._rust is not None:
            return self._rust.segment(text, dropout=dropout)
        return self._py_segment(text, dropout=dropout)

    def segment_batch(
        self,
        texts: List[str],
        dropout: float = 0.0,
        max_length: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Fast batch segmentation (multi-threaded in Rust when available)."""
        if self._rust is not None:
            return self._rust.segment_batch(texts, dropout=dropout, max_length=max_length)
        all_ids, all_lens = [], []
        for t in texts:
            ids, lens = self._py_segment(t, dropout=dropout)
            if max_length is not None:
                ids = ids[:max_length]
                lens = lens[:max_length]
            all_ids.append(ids)
            all_lens.append(lens)
        return all_ids, all_lens

    def encode_for_residues(
        self,
        sequences: List[str] | str,
        max_length: Optional[int] = None,
        dropout: float = 0.0,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor | list]:
        """Encode sequences with ``token_lengths`` and ``global_indices`` for RecycledResidueDecoder."""
        if isinstance(sequences, str):
            sequences = [sequences]
        batch_ids, batch_lens = self.segment_batch(sequences, dropout=dropout, max_length=max_length)
        max_tok = max((len(x) for x in batch_ids), default=1)
        max_res = max((sum(x) for x in batch_lens), default=1)

        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        padded_ids, attn_mask, padded_lens, global_indices = [], [], [], []
        for ids, lens in zip(batch_ids, batch_lens):
            n_tok = len(ids)
            pad_t = max_tok - n_tok
            padded_ids.append(ids + [pad_id] * pad_t)
            attn_mask.append([1] * n_tok + [0] * pad_t)
            padded_lens.append(lens + [0] * pad_t)
            n_res = sum(lens)
            global_indices.append(list(range(n_res)) + [0] * (max_res - n_res))

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                "token_lengths": torch.tensor(padded_lens, dtype=torch.long),
                "global_indices": torch.tensor(global_indices, dtype=torch.long),
            }
        return {
            "input_ids": padded_ids,
            "attention_mask": attn_mask,
            "token_lengths": padded_lens,
            "global_indices": global_indices,
        }

    # --- PreTrainedTokenizer contract ---

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        dropout = float(kwargs.get("dropout", 0.0))
        ids, _ = self.segment(text, dropout=dropout)
        return [self._id_to_token[i] for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._mask_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # LEMON is trained without <CLS>/<EOS> wrappers; keep exact sequence tokens.
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        total = len(token_ids_0) + (len(token_ids_1) if token_ids_1 is not None else 0)
        return [0] * total

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        fname = (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        dst = os.path.join(save_directory, fname)
        if os.path.abspath(self.vocab_file) != os.path.abspath(dst):
            shutil.copy2(self.vocab_file, dst)
        return (dst,)
