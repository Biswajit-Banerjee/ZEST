"""Regression tests for the residue-MLM position budget.

A batch of ``max_tokens`` tokens can span more residues than
``ThemeEncoder.global_pos_emb`` has rows, because a token covers several
residues (ZEST up to 10, BPE up to 5). Before the fix this raised
``IndexError: index out of range in self`` (a device-side assert on CUDA)
on the long-sequence pretraining shards.
"""
import torch

from zest.data.pretrain import _truncate_to_residues
from zest.model.theme_encoder import ThemeEncoder


def _segments(lengths):
    return [(n, 100 + i) for i, n in enumerate(lengths)]


def test_truncate_drops_whole_tokens():
    segs = _truncate_to_residues(_segments([4, 4, 4]), max_residues=10)
    assert [n for n, _ in segs] == [4, 4]
    assert sum(n for n, _ in segs) <= 10


def test_truncate_is_noop_when_within_budget():
    segs = _segments([3, 3])
    assert _truncate_to_residues(segs, max_residues=10) == segs
    assert _truncate_to_residues(segs, max_residues=None) == segs


def test_truncate_keeps_arrays_consistent():
    lengths = [7] * 20
    segs = _truncate_to_residues(_segments(lengths), max_residues=32)
    token_lengths = [n for n, _ in segs]
    n_residues = sum(token_lengths)
    aa_labels = list(range(n_residues))
    assert n_residues <= 32
    assert sum(token_lengths) == len(aa_labels)


def test_forward_residue_mlm_clamps_out_of_range_indices():
    max_tokens = 8
    model = ThemeEncoder(vocab_size=32, embed_dim=16, num_layers=1, nhead=2,
                         max_tokens=max_tokens, proj_dim=8, num_proj_layers=1)
    assert model.max_residues == 3 * max_tokens

    tokens = torch.randint(5, 32, (2, max_tokens))
    mask = torch.ones(2, max_tokens, dtype=torch.bool)
    token_lengths = torch.full((2, max_tokens), 5, dtype=torch.long)
    n_res = int(token_lengths[0].sum())          # 40 > 24 rows
    global_indices = torch.arange(n_res).unsqueeze(0).repeat(2, 1)

    logits = model.forward_residue_mlm(tokens, mask, token_lengths, global_indices)
    assert logits.shape[:2] == (2, n_res)
    assert torch.isfinite(logits).all()
