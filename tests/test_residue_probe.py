"""Tests for the residue probing protocol.

The leakage tests are the reason this file exists. The previous CB513 protocol
split 511 rows 80/20 at random while the file contained 79 duplicate sequences,
so ~39% of test residues were exact copies of training proteins. That inflated
every reported number and nothing in the code objected. Here it raises.

Metric tests use hand-computed cases rather than comparing against sklearn, so a
disagreement points at a specific arithmetic error.
"""
from pathlib import Path

import numpy as np
import pytest

from zest.evaluation.residue_probe import (IGNORE_INDEX, LeakageError, LinearProbe,
                                           ProbeConfig, _auroc, _mcc, _pearson,
                                           assert_no_leakage,
                                           classification_metrics, flatten,
                                           make_cross_split, make_split,
                                           probe_task, regression_metrics)

CB513_CSV = Path("data/cb513/CB513.csv")
requires_cb513 = pytest.mark.skipif(not CB513_CSV.exists(),
                                    reason="data/cb513/CB513.csv not present")


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_assert_no_leakage_raises_on_shared_sequence():
    with pytest.raises(LeakageError, match="shares 1 sequence"):
        assert_no_leakage(["AAA", "BBB"], ["BBB"])


def test_assert_no_leakage_passes_on_disjoint_sets():
    report = assert_no_leakage(["AAA", "BBB"], ["CCC"], ["DDD"])
    assert report["n_train_unique"] == 2


def test_make_split_is_disjoint_even_when_sequences_repeat():
    sequences = ["A" * 10, "B" * 10, "C" * 10, "D" * 10] * 5   # every chain 5x
    split = make_split(sequences, seed=0)
    assert split.audit["n_unique_sequences"] == 4
    assert split.audit["n_redundant_rows"] == 16

    groups = {name: {sequences[i] for i in getattr(split, name)}
              for name in ("train", "val", "test")}
    assert not (groups["train"] & groups["test"])
    assert not (groups["train"] & groups["val"])


def test_make_split_keeps_one_row_per_sequence_when_deduplicating():
    sequences = ["A" * 10] * 4 + ["B" * 10] * 3 + ["C" * 10]
    split = make_split(sequences, train_frac=0.5, val_frac=0.0, seed=1)
    rows = split.train + split.val + split.test
    assert len(rows) == len(set(sequences))
    assert len(rows) == len(set(rows))


def test_make_split_indices_are_disjoint_and_in_range():
    sequences = [f"SEQ{i}" for i in range(50)]
    split = make_split(sequences, seed=3)
    everything = split.train + split.val + split.test
    assert len(everything) == len(set(everything))
    assert all(0 <= i < len(sequences) for i in everything)
    assert len(everything) == 50


@pytest.mark.parametrize("seed", [0, 1, 42, 7])
def test_make_split_never_leaks_across_seeds(seed):
    rng = np.random.default_rng(seed)
    pool = [f"S{i}" * 3 for i in range(40)]
    sequences = list(rng.choice(pool, size=120))    # heavy duplication
    split = make_split(sequences, seed=seed)
    train = {sequences[i] for i in split.train}
    assert not (train & {sequences[i] for i in split.test})


def test_make_split_without_dedup_still_refuses_to_leak():
    """Keeping duplicate rows is allowed, but grouping by sequence means a
    protein still cannot straddle the split."""
    sequences = ["A" * 5] * 3 + ["B" * 5] * 3 + ["C" * 5] * 3 + ["D" * 5] * 3
    split = make_split(sequences, train_frac=0.5, val_frac=0.25, seed=0,
                       deduplicate=False)
    train = {sequences[i] for i in split.train}
    assert not (train & {sequences[i] for i in split.test})
    assert len(split.train) + len(split.val) + len(split.test) == len(sequences)


def test_cross_split_drops_train_rows_that_appear_in_test():
    train = ["AAA", "BBB", "CCC", "DDD", "AAA"]
    test = ["CCC", "EEE"]
    split, audit = make_cross_split(train, test, val_frac=0.0, seed=0)
    kept = {train[i] for i in split.train}
    assert "CCC" not in kept, "test sequence must not remain in training"
    assert audit["train_rows_dropped_for_test_overlap"] == 1
    assert kept == {"AAA", "BBB", "DDD"}


def test_cross_split_leaves_the_test_set_untouched():
    train = ["AAA", "BBB"]
    test = ["BBB", "CCC", "CCC"]
    _, audit = make_cross_split(train, test, val_frac=0.0)
    assert audit["n_test"] == 3, "the published test set must not be modified"


def test_cross_split_val_is_disjoint_from_train():
    train = [f"S{i}" for i in range(40)]
    split, _ = make_cross_split(train, ["ZZZ"], val_frac=0.25, seed=5)
    assert not (set(split.train) & set(split.val))
    assert len(split.val) == 10


@requires_cb513
def test_old_protocol_leaked_and_new_protocol_does_not():
    """Regression test for the actual defect, on the real file."""
    import logging

    logging.disable(logging.WARNING)
    from zest.data.netsurfp import load_cb513_csv

    sequences = load_cb513_csv(CB513_CSV).sequences

    # The old protocol: permute 511 rows, take 80/20.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(sequences))
    cut = int(0.8 * len(sequences))
    old_train = {sequences[i] for i in order[:cut]}
    leaked = [sequences[i] for i in order[cut:] if sequences[i] in old_train]
    leaked_residues = sum(len(s) for s in leaked)
    test_residues = sum(len(sequences[i]) for i in order[cut:])
    assert leaked, "expected the documented leakage to reproduce"
    assert leaked_residues / test_residues > 0.3

    # The new protocol raises rather than leaking.
    split = make_split(sequences, seed=42)
    train = {sequences[i] for i in split.train}
    assert not (train & {sequences[i] for i in split.test})


# ---------------------------------------------------------------------------
# Flattening honours the masks
# ---------------------------------------------------------------------------

def test_flatten_selects_only_masked_residues():
    embeddings = [np.arange(12, dtype=np.float32).reshape(4, 3)]
    labels = np.array([[0, 1, 2, 0]])
    mask = np.array([[True, False, True, False]])
    X, y = flatten(embeddings, labels, mask)
    assert X.shape == (2, 3)
    assert np.array_equal(y, [0, 2])
    assert np.array_equal(X[0], [0, 1, 2])
    assert np.array_equal(X[1], [6, 7, 8])


def test_flatten_drops_ignore_index_labels():
    embeddings = [np.ones((3, 2), dtype=np.float32)]
    labels = np.array([[1, IGNORE_INDEX, 2]])
    mask = np.ones((1, 3), dtype=bool)
    _, y = flatten(embeddings, labels, mask)
    assert np.array_equal(y, [1, 2])


def test_flatten_drops_nan_regression_targets():
    embeddings = [np.ones((3, 2), dtype=np.float32)]
    labels = np.array([[0.5, np.nan, 0.25]], dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    _, y = flatten(embeddings, labels, mask)
    assert np.allclose(y, [0.5, 0.25])


def test_flatten_handles_truncated_embeddings_without_misaligning():
    """A chain truncated at max_tokens must lose its tail, not shift labels."""
    embeddings = [np.arange(4, dtype=np.float32).reshape(2, 2)]
    labels = np.array([[7, 8, 9, 10]])
    mask = np.ones((1, 4), dtype=bool)
    X, y = flatten(embeddings, labels, mask)
    assert np.array_equal(y, [7, 8])
    assert X.shape == (2, 2)


def test_flatten_respects_the_index_mapping():
    embeddings = [np.full((2, 2), 5.0, dtype=np.float32)]
    labels = np.array([[0, 0], [1, 1]])
    mask = np.ones((2, 2), dtype=bool)
    _, y = flatten(embeddings, labels, mask, indices=[1])
    assert np.array_equal(y, [1, 1]), "embedding 0 must pair with chain index 1"


def test_flatten_returns_empty_when_nothing_is_scored():
    X, y = flatten([np.ones((2, 2), dtype=np.float32)],
                   np.array([[0, 0]]), np.zeros((1, 2), dtype=bool))
    assert X.shape[0] == 0 and y.shape[0] == 0


def test_flatten_caps_rows_deterministically_before_concatenation():
    embeddings = [np.arange(40, dtype=np.float32).reshape(10, 4),
                  np.arange(40, 80, dtype=np.float32).reshape(10, 4)]
    labels = np.arange(20).reshape(2, 10)
    mask = np.ones((2, 10), dtype=bool)
    first_X, first_y = flatten(embeddings, labels, mask, max_rows=7, seed=3)
    second_X, second_y = flatten(embeddings, labels, mask, max_rows=7, seed=3)
    assert first_X.shape == (7, 4)
    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_y, second_y)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_classification_metrics_on_a_hand_computed_case():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    metrics = classification_metrics(y_true, y_pred, 3)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    # class 0: tp=1 fp=1 fn=1 -> P=R=0.5, F1=0.5
    assert metrics["per_class"][0]["f1"] == pytest.approx(0.5)
    # class 1: tp=2 fp=1 fn=0 -> P=2/3, R=1, F1=0.8
    assert metrics["per_class"][1]["f1"] == pytest.approx(0.8)
    # class 2: tp=1 fp=0 fn=1 -> P=1, R=0.5, F1=2/3
    assert metrics["per_class"][2]["f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_majority_floor_is_reported():
    y_true = np.array([2] * 7 + [0, 1, 1])
    metrics = classification_metrics(y_true, y_true, 3)
    assert metrics["majority_floor"] == pytest.approx(0.7)


def test_perfect_and_constant_predictions():
    y = np.array([0, 1, 2, 1])
    assert classification_metrics(y, y, 3)["accuracy"] == 1.0
    constant = classification_metrics(y, np.zeros_like(y), 3)
    assert constant["accuracy"] == pytest.approx(0.25)
    assert constant["macro_f1"] < 0.2


def test_mcc_matches_known_values():
    perfect = np.array([0, 0, 1, 1])
    assert _mcc(perfect, perfect) == pytest.approx(1.0)
    assert _mcc(perfect, 1 - perfect) == pytest.approx(-1.0)
    assert _mcc(perfect, np.zeros_like(perfect)) == pytest.approx(0.0)


def test_auroc_matches_known_values():
    y = np.array([0, 0, 1, 1])
    assert _auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert _auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert _auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    # One positive ranked below one negative out of 4 pairs -> 0.75.
    assert _auroc(y, np.array([0.1, 0.6, 0.5, 0.9])) == pytest.approx(0.75)


def test_auroc_is_nan_for_a_single_class():
    assert np.isnan(_auroc(np.zeros(4), np.arange(4.0)))


def test_pearson_and_regression_metrics():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _pearson(a, a) == pytest.approx(1.0)
    assert _pearson(a, -a) == pytest.approx(-1.0)
    assert np.isnan(_pearson(a, np.ones(4)))

    metrics = regression_metrics(a, a + 0.5)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["pearson_r"] == pytest.approx(1.0)


def test_binary_metrics_include_mcc_and_auroc():
    y_true = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y_true, np.array([0, 0, 1, 1]), 2,
                                     probs=np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _separable(n=600, dim=8, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    centres = rng.normal(scale=4.0, size=(classes, dim))
    X = centres[y] + rng.normal(scale=0.4, size=(n, dim))
    return X.astype(np.float32), y


def test_probe_learns_a_separable_classification_task():
    X, y = _separable()
    config = ProbeConfig(epochs=40, device="cpu", verbose=False)
    metrics = probe_task("ss3", {"train": X[:400], "val": X[400:500], "test": X[500:]},
                         {"train": y[:400], "val": y[400:500], "test": y[500:]}, config)
    assert metrics["accuracy"] > 0.9
    assert metrics["feature_dim"] == 8
    assert metrics["n_test"] == 100


def test_probe_is_near_chance_on_noise():
    """Guards against a probe that appears to work because of a label bug."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8)).astype(np.float32)
    y = rng.integers(0, 3, size=600)
    config = ProbeConfig(epochs=15, device="cpu")
    metrics = probe_task("ss3", {"train": X[:400], "val": X[400:500], "test": X[500:]},
                         {"train": y[:400], "val": y[400:500], "test": y[500:]}, config)
    assert metrics["accuracy"] < 0.55


def test_probe_learns_a_linear_regression_target():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 6)).astype(np.float32)
    w = rng.normal(size=6)
    y = (X @ w).astype(np.float32)
    config = ProbeConfig(epochs=120, lr=5e-2, device="cpu")
    metrics = probe_task("rsa", {"train": X[:600], "val": X[600:700], "test": X[700:]},
                         {"train": y[:600], "val": y[600:700], "test": y[700:]}, config)
    assert metrics["pearson_r"] > 0.95


def test_probe_rejects_unknown_task():
    with pytest.raises(ValueError, match="unknown task"):
        LinearProbe("ramachandran")


def test_probe_requires_fitting_before_predicting():
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearProbe("ss3").predict(np.zeros((2, 3), dtype=np.float32))


def test_standardization_handles_constant_features():
    """A dead feature column must not produce NaN through a zero divide."""
    X, y = _separable(n=300, dim=5)
    X[:, 2] = 1.0
    probe = LinearProbe("ss3", ProbeConfig(epochs=5, device="cpu")).fit(X, y)
    predictions, logits = probe.predict(X)
    assert np.isfinite(logits).all()
    assert predictions.shape == (300,)


def test_standardization_preparation_does_not_copy_full_feature_matrix():
    X, _ = _separable(n=100, dim=5)
    probe = LinearProbe("ss3", ProbeConfig(device="cpu"))
    prepared = probe._prepare(X, fit=True)
    assert np.shares_memory(prepared, X)
    assert probe._mean.shape == (1, 5)
    assert probe._std.shape == (1, 5)


def test_probe_is_deterministic_for_a_fixed_seed():
    X, y = _separable(n=300, dim=6)
    config = ProbeConfig(epochs=10, device="cpu", seed=7)
    first = LinearProbe("ss3", config).fit(X, y).evaluate(X, y)
    second = LinearProbe("ss3", config).fit(X, y).evaluate(X, y)
    assert first["accuracy"] == second["accuracy"]


def test_class_weighting_helps_the_minority_class():
    rng = np.random.default_rng(0)
    y = np.array([0] * 570 + [1] * 30)
    centres = np.array([[0.0, 0.0], [1.4, 1.4]])
    X = (centres[y] + rng.normal(scale=1.0, size=(600, 2))).astype(np.float32)

    plain = LinearProbe("disorder", ProbeConfig(epochs=60, device="cpu")).fit(X, y)
    weighted = LinearProbe("disorder", ProbeConfig(epochs=60, device="cpu",
                                                   class_weight=True)).fit(X, y)
    assert (weighted.evaluate(X, y)["per_class"][1]["recall"]
            >= plain.evaluate(X, y)["per_class"][1]["recall"])
