"""Tests for the paired chain-level bootstrap and the readout-capacity ladder.

The point of the paired test is that it resolves effects the marginal-interval
overlap check cannot. That claim is not taken on faith here: the central test
constructs data where the two marginal intervals overlap and the paired test is
still decisive, which is exactly the situation the L0 decoder deltas are in.
"""
import numpy as np
import pytest

from zest.evaluation.residue_probe import (PairingError, Probe, ProbeConfig,
                                           bootstrap_classification_by_chain,
                                           paired_bootstrap_classification_by_chain,
                                           paired_bootstrap_regression_by_chain,
                                           paired_compare_task, probe_task)


def _chain_index(n_chains: int, per_chain: int) -> np.ndarray:
    return np.repeat(np.arange(n_chains), per_chain)


def _heterogeneous_arms(n_chains=120, per_chain=60, seed=0):
    """Two arms differing by a small consistent margin, on very uneven chains.

    Chain difficulty varies wildly, which is what inflates each arm's marginal
    interval. Arm B is uniformly a little better on every chain, so the paired
    difference is stable even though the marginal distributions overlap heavily.
    """
    rng = np.random.default_rng(seed)
    chains = _chain_index(n_chains, per_chain)
    y_true = rng.integers(0, 3, size=chains.size)

    # Per-chain accuracy spread from very easy to very hard.
    difficulty = rng.uniform(0.25, 0.95, size=n_chains)[chains]
    correct_a = rng.random(chains.size) < difficulty
    # B fixes a fixed slice of A's mistakes: a consistent, modest improvement.
    rescued = (~correct_a) & (rng.random(chains.size) < 0.10)
    correct_b = correct_a | rescued

    def predictions(correct):
        wrong = (y_true + rng.integers(1, 3, size=chains.size)) % 3
        return np.where(correct, y_true, wrong)

    return y_true, predictions(correct_a), predictions(correct_b), chains


def test_paired_test_resolves_what_overlapping_intervals_cannot():
    """The motivating case: marginal intervals overlap, the paired test does not."""
    y_true, pred_a, pred_b, chains = _heterogeneous_arms()

    marginal_a = bootstrap_classification_by_chain(
        y_true, pred_a, chains, 3, replicates=2000)["accuracy"]
    marginal_b = bootstrap_classification_by_chain(
        y_true, pred_b, chains, 3, replicates=2000)["accuracy"]
    overlap = not (marginal_a["ci95_high"] < marginal_b["ci95_low"]
                   or marginal_b["ci95_high"] < marginal_a["ci95_low"])
    assert overlap, "fixture should produce overlapping marginal intervals"

    paired = paired_bootstrap_classification_by_chain(
        y_true, pred_a, pred_b, chains, 3, replicates=2000)["accuracy"]
    assert paired["delta"] > 0
    assert paired["ci95_low"] > 0, "paired interval should exclude zero"
    assert paired["significant_at_05"]


def test_paired_delta_matches_the_observed_difference():
    y_true, pred_a, pred_b, chains = _heterogeneous_arms(n_chains=40, per_chain=30)
    paired = paired_bootstrap_classification_by_chain(
        y_true, pred_a, pred_b, chains, 3, replicates=500)
    expected = float((y_true == pred_b).mean()) - float((y_true == pred_a).mean())
    assert paired["accuracy"]["delta"] == pytest.approx(expected, abs=1e-12)


def test_identical_arms_produce_zero_delta_and_a_non_significant_p_value():
    y_true, pred_a, _, chains = _heterogeneous_arms(n_chains=50, per_chain=40)
    paired = paired_bootstrap_classification_by_chain(
        y_true, pred_a, pred_a, chains, 3, replicates=1000)["accuracy"]
    assert paired["delta"] == pytest.approx(0.0, abs=1e-12)
    assert paired["ci95_low"] == pytest.approx(0.0, abs=1e-12)
    assert paired["ci95_high"] == pytest.approx(0.0, abs=1e-12)
    assert not paired["significant_at_05"]


def test_p_value_is_floored_at_bootstrap_resolution():
    """A bootstrap cannot report a p-value finer than 1/replicates."""
    rng = np.random.default_rng(3)
    chains = _chain_index(60, 50)
    y_true = rng.integers(0, 3, size=chains.size)
    pred_a = (y_true + 1) % 3          # always wrong
    pred_b = y_true.copy()             # always right
    paired = paired_bootstrap_classification_by_chain(
        y_true, pred_a, pred_b, chains, 3, replicates=1000)["accuracy"]
    assert paired["p_value"] == pytest.approx(1.0 / 1000)
    assert paired["p_value"] > 0


def test_pairing_rejects_mismatched_residue_counts():
    y_true, pred_a, pred_b, chains = _heterogeneous_arms(n_chains=10, per_chain=10)
    arm_a = {"y_true": y_true, "predictions": pred_a, "chains": chains}
    arm_b = {"y_true": y_true[:-1], "predictions": pred_b[:-1], "chains": chains[:-1]}
    with pytest.raises(PairingError, match="different residue counts"):
        paired_compare_task("ss3", arm_a, arm_b)


def test_pairing_rejects_disagreeing_ground_truth():
    """Different labels mean the arms are not scoring the same residues."""
    y_true, pred_a, pred_b, chains = _heterogeneous_arms(n_chains=10, per_chain=10)
    shifted = y_true.copy()
    shifted[0] = (shifted[0] + 1) % 3
    arm_a = {"y_true": y_true, "predictions": pred_a, "chains": chains}
    arm_b = {"y_true": shifted, "predictions": pred_b, "chains": chains}
    with pytest.raises(PairingError, match="ground-truth"):
        paired_compare_task("ss3", arm_a, arm_b)


def test_paired_regression_reports_signed_deltas():
    rng = np.random.default_rng(7)
    chains = _chain_index(80, 40)
    y_true = rng.normal(size=chains.size)
    pred_a = y_true + rng.normal(scale=0.9, size=chains.size)
    pred_b = y_true + rng.normal(scale=0.4, size=chains.size)

    paired = paired_bootstrap_regression_by_chain(
        y_true, pred_a, pred_b, chains, replicates=1000)
    # B is the better predictor: correlation up, error down.
    assert paired["pearson_r"]["delta"] > 0
    assert paired["pearson_r"]["ci95_low"] > 0
    assert paired["mae"]["delta"] < 0
    assert paired["mae"]["ci95_high"] < 0


def test_paired_compare_task_dispatches_regression_and_classification():
    y_true, pred_a, pred_b, chains = _heterogeneous_arms(n_chains=20, per_chain=25)
    classification = paired_compare_task(
        "ss3", {"y_true": y_true, "predictions": pred_a, "chains": chains},
        {"y_true": y_true, "predictions": pred_b, "chains": chains},
        replicates=200)
    assert "macro_f1" in classification

    continuous = np.linspace(0, 1, chains.size)
    regression = paired_compare_task(
        "rsa", {"y_true": continuous, "predictions": continuous, "chains": chains},
        {"y_true": continuous, "predictions": continuous, "chains": chains},
        replicates=200)
    assert "pearson_r" in regression


def test_paired_compare_task_rejects_unknown_task():
    with pytest.raises(ValueError, match="unknown task"):
        paired_compare_task("ramachandran", {}, {})


# ---------------------------------------------------------------------------
# Readout-capacity ladder
# ---------------------------------------------------------------------------

def _xor_like(n=1200, seed=1):
    """A target no linear readout can fit, but an MLP can.

    This is the readout-capacity argument in miniature: the information is
    fully present in the features, and a linear probe still reports near
    chance. A low linear probe score therefore cannot establish that a
    representation lost the signal.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    return X, y


def test_linear_readout_is_the_default():
    config = ProbeConfig()
    assert config.hidden_dims == ()
    assert config.describe() == "linear"


def test_hidden_dims_label_records_the_rung():
    assert ProbeConfig(hidden_dims=(512,)).describe() == "mlp:512"
    assert ProbeConfig(hidden_dims=(512, 256)).describe() == "mlp:512-256"


def test_mlp_readout_extracts_signal_a_linear_readout_cannot():
    X, y = _xor_like()
    common = dict(epochs=150, device="cpu", seed=0, min_steps=0)

    linear = Probe("ss3", ProbeConfig(**common)).fit(X, y).evaluate(X, y)
    mlp = Probe("ss3", ProbeConfig(hidden_dims=(64, 64), **common)).fit(X, y).evaluate(X, y)

    assert linear["accuracy"] < 0.65, "linear probe should fail on XOR"
    assert mlp["accuracy"] > 0.90, "MLP readout should recover it"
    assert mlp["accuracy"] > linear["accuracy"] + 0.25


def test_probe_task_records_the_readout_used():
    X, y = _xor_like(n=400)
    features = {"train": X, "val": X, "test": X}
    labels = {"train": y, "val": y, "test": y}
    config = ProbeConfig(hidden_dims=(32,), epochs=10, device="cpu", min_steps=0)
    metrics = probe_task("ss3", features, labels, config)
    assert metrics["readout"] == "mlp:32"
    assert metrics["fit_info"]["readout"] == "mlp:32"
    assert metrics["fit_info"]["n_readout_params"] > 0


def test_probe_task_persists_predictions_for_pairing():
    """Predictions must survive so two arms can be compared later."""
    X, y = _xor_like(n=300)
    chains = np.repeat(np.arange(30), 10)
    features = {"train": X, "val": X, "test": X}
    labels = {"train": y, "val": y, "test": y}
    metrics = probe_task("ss3", features, labels,
                         ProbeConfig(epochs=5, device="cpu", min_steps=0),
                         test_chains=chains)
    block = metrics["_predictions"]
    assert block["predictions"].shape == y.shape
    np.testing.assert_array_equal(block["y_true"], y)
    np.testing.assert_array_equal(block["chains"], chains)
    assert metrics["n_test_chains"] == 30


def test_predictions_are_available_without_requesting_a_bootstrap():
    """Pairing must not require paying for per-arm intervals first."""
    X, y = _xor_like(n=200)
    chains = np.repeat(np.arange(20), 10)
    metrics = probe_task("ss3", {"train": X, "val": X, "test": X},
                         {"train": y, "val": y, "test": y},
                         ProbeConfig(epochs=5, device="cpu", min_steps=0),
                         test_chains=chains, bootstrap_replicates=0)
    assert "_predictions" in metrics
    assert "intervals" not in metrics
