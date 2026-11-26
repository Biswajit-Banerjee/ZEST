import numpy as np
import torch
import torch.nn.functional as F

from zest.data.finetune import HierarchicalPairDataset
from zest.training.losses import (
    _hierarchical_masks,
    hierarchical_infonce,
    hierarchical_smooth_ap,
)


def test_benchmark_aligned_masks_exclude_finer_levels():
    a = torch.tensor([0, 0, 0, 0, 1])
    t = torch.tensor([0, 0, 0, 1, 2])
    h = torch.tensor([0, 0, 1, 2, 3])
    eye = torch.eye(5, dtype=torch.bool)

    a_pos, a_valid = _hierarchical_masks(
        "a", a, t, h, a, t, h, eye, benchmark_aligned=True)
    t_pos, t_valid = _hierarchical_masks(
        "t", a, t, h, a, t, h, eye, benchmark_aligned=True)
    h_pos, h_valid = _hierarchical_masks(
        "h", a, t, h, a, t, h, eye, benchmark_aligned=True)

    assert torch.equal(a_pos[0], torch.tensor([False, False, False, True, False]))
    assert torch.equal(a_valid[0], torch.tensor([False, False, False, True, True]))
    assert torch.equal(t_pos[0], torch.tensor([False, False, True, False, False]))
    assert torch.equal(t_valid[0], torch.tensor([False, False, True, True, True]))
    assert torch.equal(h_pos[0], torch.tensor([False, True, False, False, False]))
    assert torch.equal(h_valid[0], torch.tensor([False, True, True, True, True]))


def test_remote_positive_sampling_uses_configured_probability_bands():
    dataset = HierarchicalPairDataset.__new__(HierarchicalPairDataset)
    dataset._active = np.arange(3)
    dataset._t_labels = np.array([0, 0, 1], dtype=np.int32)
    dataset._probabilities = np.array([0.70, 0.85, 0.95], dtype=np.float32)

    weights = dataset.get_sample_weights(
        fold_balanced=False,
        remote_positive=True,
        remote_low_threshold=0.80,
        remote_mid_threshold=0.90,
        remote_low_weight=4.0,
        remote_mid_weight=2.0,
    )

    torch.testing.assert_close(weights, torch.tensor([4 / 7, 2 / 7, 1 / 7], dtype=torch.float64))


def test_benchmark_aligned_losses_are_finite_and_differentiable():
    torch.manual_seed(7)
    raw = torch.randn(6, 8, requires_grad=True)
    embeddings = F.normalize(raw, dim=-1)
    a = torch.tensor([0, 0, 0, 0, 1, 1])
    t = torch.tensor([0, 0, 0, 1, 2, 2])
    h = torch.tensor([0, 0, 1, 2, 3, 4])

    infonce = hierarchical_infonce(
        embeddings, a, t, h, benchmark_aligned=True)
    smooth_ap = hierarchical_smooth_ap(
        embeddings, a, t, h, benchmark_aligned=True)
    loss = infonce["loss"] + smooth_ap["sap_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
