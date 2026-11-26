from pathlib import Path

import torch

from zest.data.pretrain import MLMDataset
from zest.model.theme_encoder import ThemeEncoder
from zest.training.pretrain import EqualizedSampler


ROOT = Path(__file__).parents[1]


def test_equalized_sampler_has_same_length_and_deterministic_order():
    first = list(EqualizedSampler(3, 8, seed=42))
    second = list(EqualizedSampler(3, 8, seed=42))

    assert first == second
    assert len(first) == 8
    assert set(first) == {0, 1, 2}


def test_rank_shard_assignment_is_disjoint():
    selected = [f"shard_{index}.parquet" for index in range(8)]
    assignments = []
    for rank in range(4):
        dataset = MLMDataset.__new__(MLMDataset)
        dataset._world_size = 4
        dataset._rank = rank
        assignments.append(dataset._rank_shards(selected))

    assert all(len(paths) == 2 for paths in assignments)
    assert len(set(sum(assignments, []))) == len(selected)


def test_pretrain_has_no_hardcoded_cuda_zero_path():
    source = (ROOT / "zest/training/pretrain.py").read_text()
    assert '"cuda:0"' not in source
    assert ".cuda(" not in source


def test_model_combines_token_and_residue_outputs_in_one_forward():
    model = ThemeEncoder(vocab_size=32, embed_dim=16, num_layers=1, nhead=2,
                         max_tokens=8, proj_dim=8, num_proj_layers=1,
                         residue_hidden_dim=16, residue_heads=2)
    tokens = torch.randint(5, 32, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)
    lengths = torch.full((2, 4), 2, dtype=torch.long)
    indices = torch.arange(8).repeat(2, 1)
    token_logits, residue_logits = model(tokens, mask, lengths, indices)

    assert token_logits.shape == (2, 4, 32)
    assert residue_logits.shape[:2] == (2, 8)
