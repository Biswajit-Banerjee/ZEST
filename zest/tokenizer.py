"""
ZEST Tokenizer: Zoned Encoding of Sequence Themes.

A domain-agnostic tokenizer built on two simple primitives:

  1. An **alphabet** — a list of base symbol strings (e.g. amino acids,
     AST node types, musical notes, DNA bases — anything).

  2. A **vocab_map** — patterns over those symbols, with many-to-one
     clustering: {centroid_pattern: [member_1, member_2, ...]}.

The tokenizer itself never knows or cares what domain it's in. It
operates on `List[str]` — a sequence of alphabet symbols — and greedily
matches the longest known pattern at each position via a trie. Unmatched
symbols fall back to individual base-alphabet tokens.

Domain-specific encoding (turning raw input into alphabet symbols) is
done externally by passing an `encoder` callable, or by calling
`encode_symbols()` directly with pre-extracted symbols.

Usage:
    # Any domain — pass pre-extracted symbols
    tok = ZESTTokenizer("vocab_map.json", alphabet=["A", "C", "D", ...])
    ids = tok.encode_symbols(["A", "C", "D", "E", "F"])

    # Biology convenience — raw amino acid string
    tok = ZESTTokenizer("vocab_map.json")  # defaults to amino acid alphabet
    ids = tok.encode("ACDEFGHIKLMNPQRSTVWY")

    # Code convenience — raw source code with an encoder
    from experiments.code.build_tokenizer import lex_function
    tok = ZESTTokenizer("vocab_map.json", alphabet_path="alphabet.json")
    ids = tok.encode("def foo(x): return x", encoder=lambda s: lex_function(s, "python"))

    # Or pre-extract symbols yourself
    symbols = my_custom_encoder(raw_input)
    ids = tok.encode_symbols(symbols)
"""
import json
import os
import random
import re
from typing import Callable, Dict, List, Optional, Tuple

import torch


class _TrieNode:
    __slots__ = ["children", "token_id"]
    # When hours of leetcode finally makes sense
    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.token_id: int = -1


class ZESTTokenizer:
    """
    Domain-agnostic greedy max-match tokenizer backed by a symbol-level trie.

    Works with any domain whose data can be represented as a sequence of
    discrete alphabet symbols. Patterns (multi-symbol sequences) are matched
    greedily longest-first. Many-to-one clustering maps variant patterns to
    the same token ID.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<EOS>", "<MASK>"]
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        vocab_path: str,
        alphabet: Optional[List[str]] = None,
        alphabet_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            vocab_path:    path to vocab_map.json ({centroid: [members]})
            alphabet:      explicit list of base symbol strings. If None,
                           loaded from alphabet_path or defaults to amino acids.
            alphabet_path: path to a JSON list of base symbols. Used when
                           alphabet is not provided explicitly.
            verbose:       print summary on init
        """
        self.path = vocab_path
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._root = _TrieNode()
        self._symbol_to_id: Dict[str, int] = {}
        self._ws = re.compile(r"\s+")

        # ---- Special tokens (IDs 0-4) ----
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.vocab[tok] = i
            self.id_to_token[i] = tok

        # ---- Base alphabet ----
        if alphabet is not None:
            self._alphabet = list(alphabet)
        elif alphabet_path is not None and os.path.exists(alphabet_path):
            with open(alphabet_path) as f:
                self._alphabet = json.load(f)
        else:
            # Try auto-detecting alphabet file next to vocab
            auto_path = vocab_path.replace(".json", "_alphabet.json")
            if os.path.exists(auto_path):
                with open(auto_path) as f:
                    self._alphabet = json.load(f)
            else:
                self._alphabet = list(self.AMINO_ACIDS)

        offset = len(self.SPECIAL_TOKENS)
        for i, sym in enumerate(self._alphabet):
            tid = offset + i
            self.vocab[sym] = tid
            self.id_to_token[tid] = sym
            self._symbol_to_id[sym] = tid
            self._trie_insert([sym], tid)

        # ---- Patterns: centroid → [members] (many-to-one) ----
        with open(vocab_path) as f:
            clusters: Dict[str, List[str]] = json.load(f)

        offset = len(self.vocab)
        for i, (centroid, members) in enumerate(clusters.items()):
            tid = offset + i
            self.id_to_token[tid] = centroid
            self.vocab[centroid] = tid
            self._trie_insert(self._pattern_to_symbols(centroid), tid)

            for member in (members or []):
                if member == centroid:
                    continue
                self.vocab[member] = tid
                self._trie_insert(self._pattern_to_symbols(member), tid)

        # ---- Convenience IDs ----
        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]
        self.cls_id = self.vocab["<CLS>"]
        self.eos_id = self.vocab["<EOS>"]
        self.mask_id = self.vocab["<MASK>"]
        self.vocab_size = len(self.id_to_token)

        if verbose:
            n_many = sum(1 for m in clusters.values() if m and len(m) > 1)
            print(f"ZESTTokenizer ready — vocab size: {self.vocab_size}")
            print(f"  Alphabet:    {len(self._alphabet)} base symbols")
            print(f"  Patterns:    {len(clusters)} ({n_many} many-to-one)")

    # ------------------------------------------------------------------
    # Pattern → symbol list conversion
    # ------------------------------------------------------------------

    def _pattern_to_symbols(self, pattern: str) -> List[str]:
        """
        Convert a pattern string to a list of symbols for trie insertion.

        For single-char alphabets (biology): each character is a symbol.
        For multi-char alphabets (code, etc.): space-separated tokens.
        """
        if self._alphabet and all(len(s) == 1 for s in self._alphabet):
            return list(pattern)
        return pattern.split()

    # ------------------------------------------------------------------
    # Trie operations (symbol-level — each edge is a full symbol string)
    # ------------------------------------------------------------------

    def _trie_insert(self, symbols: List[str], token_id: int) -> None:
        """Insert a symbol sequence into the trie."""
        node = self._root
        for sym in symbols:
            if sym not in node.children:
                node.children[sym] = _TrieNode()
            node = node.children[sym]
        node.token_id = token_id

    def _trie_match(self, symbols: List[str], start: int) -> List[Tuple[int, int]]:
        """Return all valid (length, token_id) matches starting at *start*, longest first."""
        matches: List[Tuple[int, int]] = []
        node = self._root
        for i in range(start, len(symbols)):
            sym = symbols[i]
            if sym not in node.children:
                break
            node = node.children[sym]
            if node.token_id != -1:
                matches.append((i - start + 1, node.token_id))
        return matches[::-1]

    # ------------------------------------------------------------------
    # Core tokenization (domain-agnostic, operates on symbol lists)
    # ------------------------------------------------------------------

    def _segment(self, symbols: List[str], dropout: float = 0.0) -> List[Tuple[int, int]]:
        """
        Segment a symbol sequence into (length, token_id) pairs.

        Greedy longest-first matching. With dropout > 0, randomly picks
        a shorter match, forcing diverse segmentations during training.
        """
        segments: List[Tuple[int, int]] = []
        i, n = 0, len(symbols)
        while i < n:
            matches = self._trie_match(symbols, i)
            if not matches:
                segments.append((1, self.mask_id))
                i += 1
                continue
            length, tid = matches[0]
            if dropout > 0 and random.random() < dropout and len(matches) > 1:
                idx = random.randint(1, len(matches) - 1)
                length, tid = matches[idx]
            segments.append((length, tid))
            i += length
        return segments

    def encode_symbols(
        self,
        symbols: List[str],
        dropout: float = 0.0,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """
        Encode a pre-extracted list of alphabet symbols into token IDs.

        This is the core method — domain-agnostic. Call this when you've
        already converted your raw input into a symbol sequence.
        """
        ids = [tid for _, tid in self._segment(symbols, dropout)]
        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.eos_id]
        return ids

    # ------------------------------------------------------------------
    # Convenience encode (accepts raw input + optional encoder)
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        dropout: float = 0.0,
        add_special_tokens: bool = False,
        encoder: Optional[Callable] = None,
    ) -> List[int]:
        """
        Encode raw input into token IDs.

        The optional `encoder` callable converts raw input into either:
          - List[str]: alphabet symbols → trie matching with dropout
          - List[int]: pre-computed token IDs → pass through directly

        Returning List[int] is useful for hierarchical encoders (e.g. AST
        walk) that do their own vocab lookup internally. Returning List[str]
        is for flat encoders (e.g. lexers) that produce symbol streams.

        If no encoder is provided, the default splits input into individual
        characters (suitable for single-char alphabets like amino acids).
        """
        if encoder is not None:
            result = encoder(text)
            if result and isinstance(result[0], int):
                ids = result
            else:
                return self.encode_symbols(result, dropout, add_special_tokens)
        else:
            symbols = list(self._ws.sub("", text).upper())
            return self.encode_symbols(symbols, dropout, add_special_tokens)

        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to symbol strings."""
        skip = {self.pad_id, self.cls_id, self.eos_id, self.mask_id}
        parts = []
        for i in ids:
            if skip_special and i in skip:
                continue
            parts.append(self.id_to_token.get(i, ""))
        # Single-char alphabets: join without separator
        # Multi-char alphabets: join with " | "
        if self._alphabet and len(self._alphabet[0]) == 1:
            return "".join(parts)
        return " | ".join(parts)

    def batch_encode_plus(
        self,
        sequences: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        dropout: float = 0.0,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        encoder: Optional[Callable[[str], List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode raw inputs.

        Args:
            sequences:    list of raw input strings
            encoder:      optional callable to convert each string to symbols
            padding:      pad to max_length or longest in batch
            max_length:   maximum sequence length
            truncation:   truncate if exceeding max_length
            dropout:      probability of choosing shorter match (training)
            return_tensors: "pt" for PyTorch tensors
            add_special_tokens: prepend <CLS> and append <EOS>
        """
        batch = [self.encode(s, dropout=dropout, encoder=encoder) for s in sequences]
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
        """Save vocab_map.json and alphabet.json to a directory."""
        os.makedirs(directory, exist_ok=True)

        # Re-extract cluster data from vocab
        clusters: Dict[str, List[str]] = {}
        offset = len(self.SPECIAL_TOKENS) + len(self._alphabet)
        centroid_ids = {tid for tid in self.id_to_token if tid >= offset}
        for tok, tid in self.vocab.items():
            if tid not in centroid_ids or tok in self.id_to_token.values():
                continue
            centroid = self.id_to_token[tid]
            clusters.setdefault(centroid, []).append(tok)

        with open(os.path.join(directory, "vocab_map.json"), "w") as f:
            json.dump(clusters, f, indent=2)
        with open(os.path.join(directory, "vocab_map_alphabet.json"), "w") as f:
            json.dump(self._alphabet, f, indent=2)

    @classmethod
    def from_pretrained(cls, directory: str, **kwargs) -> "ZESTTokenizer":
        """Load tokenizer from a directory containing vocab_map.json."""
        vocab_path = os.path.join(directory, "vocab_map.json")
        alpha_path = os.path.join(directory, "vocab_map_alphabet.json")
        if os.path.exists(alpha_path):
            return cls(vocab_path, alphabet_path=alpha_path, **kwargs)
        return cls(vocab_path, **kwargs)
