"""
Protein BPE tokenizer wrapper.

Wraps a HuggingFace `tokenizers` BPE model (JSON) to expose the same
interface as ZESTTokenizer so that pretrain / finetune / benchmark scripts
work unchanged with `--tokenizer-type bpe`.

Special tokens (must match the BPE vocab JSON):
    0: [PAD]   1: [CLS]   2: [EOS]   3: [UNK]   4: [MASK]
"""
import re
from typing import Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer


class ProteinBPETokenizer:
    """Drop-in replacement for ZESTTokenizer backed by HuggingFace BPE."""

    def __init__(self, path: str, verbose: bool = False):
        self._tok = Tokenizer.from_file(path)
        self.path = path

        # Special token IDs (must match added_tokens in the JSON)
        self.pad_id = self._tok.token_to_id("[PAD]")
        self.cls_id = self._tok.token_to_id("[CLS]")
        self.eos_id = self._tok.token_to_id("[EOS]")
        self.unk_id = self._tok.token_to_id("[UNK]")
        self.mask_id = self._tok.token_to_id("[MASK]")
        self.vocab_size = self._tok.get_vocab_size()

        self._ws = re.compile(r"\s+")

        if verbose:
            print(f"ProteinBPETokenizer ready — vocab size: {self.vocab_size}")

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        dropout: float = 0.0,
        add_special_tokens: bool = False,
        encoder=None,
    ) -> List[int]:
        """Encode a protein sequence string into token IDs."""
        seq = self._ws.sub("", text).upper()
        enc = self._tok.encode(seq, add_special_tokens=False)
        ids = enc.ids
        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.eos_id]
        return ids

    def encode_symbols(
        self,
        symbols: List[str],
        dropout: float = 0.0,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """Encode from a list of single-char amino acid symbols."""
        return self.encode("".join(symbols), dropout=dropout,
                           add_special_tokens=add_special_tokens)

    def _segment(self, symbols: List[str],
                 dropout: float = 0.0) -> List[Tuple[int, int]]:
        """Segment symbols into ``(residue_length, token_id)`` pairs.

        Mirrors ``ZESTTokenizer._segment`` so that residue-level MLM works
        identically for the BPE baseline. Spans are derived from the
        HuggingFace character offsets and are guaranteed to be contiguous
        and to cover every input symbol exactly once, so that
        ``sum(lengths) == len(symbols)``.

        ``dropout`` is accepted for API parity but ignored: BPE segmentation
        is deterministic.
        """
        text = "".join(symbols)
        if not text:
            return []

        enc = self._tok.encode(text, add_special_tokens=False)
        segments: List[Tuple[int, int]] = []
        cursor = 0
        for tid, (start, end) in zip(enc.ids, enc.offsets):
            if end <= cursor:
                continue
            # Absorb any gap (characters the pre-tokenizer dropped) into
            # this token so residue indices stay aligned with `text`.
            length = end - cursor
            segments.append((length, tid))
            cursor = end

        # Any trailing characters not covered by a token map to UNK.
        while cursor < len(text):
            segments.append((1, self.unk_id))
            cursor += 1

        return segments

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to amino acid string."""
        if skip_special:
            skip = {self.pad_id, self.cls_id, self.eos_id, self.mask_id}
            ids = [i for i in ids if i not in skip]
        return self._tok.decode(ids)

    # ------------------------------------------------------------------
    # Batch encode (mirrors ZESTTokenizer.batch_encode_plus)
    # ------------------------------------------------------------------

    def batch_encode_plus(
        self,
        sequences: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        dropout: float = 0.0,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        encoder=None,
    ) -> Dict[str, torch.Tensor]:
        batch = [self.encode(s, add_special_tokens=add_special_tokens)
                 for s in sequences]

        if max_length and truncation:
            batch = [ids[:max_length] for ids in batch]

        if padding:
            target = max_length or max(len(ids) for ids in batch)
            padded, masks = [], []
            for ids in batch:
                pad_n = max(0, target - len(ids))
                padded.append(ids + [self.pad_id] * pad_n)
                masks.append([1] * len(ids) + [0] * pad_n)
        else:
            padded, masks = batch, [[1] * len(ids) for ids in batch]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}
