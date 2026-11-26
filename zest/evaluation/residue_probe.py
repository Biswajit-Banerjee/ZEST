"""Residue-level linear probing with an enforced no-leakage protocol.

This module exists because the previous CB513 probe was measuring the wrong
thing. Two defects, both quantified:

1. **Leakage.** CB513 as distributed contains 79 rows that repeat an earlier
   sequence. The old protocol split 511 chains 80/20 at random, so identical
   proteins landed on both sides: ~39% of test residues were exact copies of
   training proteins (measured across five seeds). Every previously reported
   L0/L1/L2 and ESM-2 number is inflated by that.
2. **Ignored evaluation mask.** ``cb513_mask`` marks the 58% of residues the
   published protocol actually scores. Reporting over all 144,011 residues is
   not comparable to any number in the literature.

Both are now structural rather than advisory: :func:`make_split` deduplicates
before splitting and then *asserts* zero overlap, and :func:`flatten` refuses to
run unless told explicitly whether to honour the mask.

The probe itself is a single linear layer, which is the point: it measures what
is linearly decodable from a frozen representation, so a fair comparison against
ESM-2 requires only that both sides get the same probe and the same residues.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_log = logging.getLogger(__name__)

IGNORE_INDEX = -100

CLASSIFICATION_TASKS = {"ss3": 3, "ss8": 8, "disorder": 2}
REGRESSION_TASKS = {"rsa", "asa"}


# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------

@dataclass
class Split:
    """Chain indices for one protocol, plus the audit that justifies it."""

    train: List[int]
    val: List[int]
    test: List[int]
    audit: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Split(train={len(self.train)}, val={len(self.val)}, "
                f"test={len(self.test)})")


class LeakageError(AssertionError):
    """Raised when train and evaluation splits share a sequence."""


def assert_no_leakage(train_sequences: Sequence[str], *eval_sets: Sequence[str]) -> Dict:
    """Fail if any evaluation sequence appears verbatim in training.

    Deliberately an exception, not a warning: the previous protocol degraded
    silently and produced numbers that looked reasonable.
    """
    train_set = set(train_sequences)
    report = {}
    for i, group in enumerate(eval_sets):
        overlap = [s for s in group if s in train_set]
        if overlap:
            residues = sum(len(s) for s in overlap)
            total = sum(len(s) for s in group) or 1
            raise LeakageError(
                f"eval set {i} shares {len(overlap)} sequence(s) with train "
                f"({residues} residues, {residues/total:.1%} of the set). "
                "Deduplicate before splitting.")
        report[f"eval_{i}_overlap"] = 0
    report["n_train_unique"] = len(train_set)
    return report


def make_split(sequences: Sequence[str], train_frac: float = 0.7,
               val_frac: float = 0.1, seed: int = 42,
               deduplicate: bool = True) -> Split:
    """Chain-level split with duplicates collapsed *before* partitioning.

    Deduplicating first is what prevents leakage: assigning whole sequence
    groups rather than rows means no protein can appear on both sides.
    """
    groups: Dict[str, List[int]] = {}
    for i, seq in enumerate(sequences):
        groups.setdefault(seq, []).append(i)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))

    n_train = int(train_frac * len(keys))
    n_val = int(val_frac * len(keys))
    buckets = {"train": order[:n_train],
               "val": order[n_train:n_train + n_val],
               "test": order[n_train + n_val:]}

    chosen: Dict[str, List[int]] = {}
    for name, positions in buckets.items():
        picked: List[int] = []
        for p in positions:
            rows = groups[keys[p]]
            picked.extend(rows[:1] if deduplicate else rows)
        chosen[name] = sorted(picked)

    split = Split(chosen["train"], chosen["val"], chosen["test"])
    split.audit = {
        "n_rows": len(sequences),
        "n_unique_sequences": len(keys),
        "n_redundant_rows": len(sequences) - len(keys),
        "deduplicated": deduplicate,
        "seed": seed,
    }
    split.audit.update(assert_no_leakage(
        [sequences[i] for i in split.train],
        [sequences[i] for i in split.val],
        [sequences[i] for i in split.test]))
    return split


def make_cross_split(train_sequences: Sequence[str], test_sequences: Sequence[str],
                     val_frac: float = 0.1, seed: int = 42) -> Tuple[Split, Dict]:
    """Protocol for a separate train file and test file (the preferred setup).

    Any test sequence also present in the training file is dropped from
    *training*, so the test set stays exactly as published.
    """
    test_set = set(test_sequences)
    seen: set = set()
    train_keep: List[int] = []
    dropped = 0
    for i, seq in enumerate(train_sequences):
        if seq in test_set:
            dropped += 1
            continue
        if seq in seen:
            continue
        seen.add(seq)
        train_keep.append(i)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_keep))
    n_val = int(val_frac * len(train_keep))
    val = sorted(train_keep[i] for i in order[:n_val])
    train = sorted(train_keep[i] for i in order[n_val:])

    audit = {
        "n_train_rows": len(train_sequences),
        "n_train_kept": len(train),
        "n_val": len(val),
        "n_test": len(test_sequences),
        "train_rows_dropped_for_test_overlap": dropped,
        "train_duplicate_rows_dropped": len(train_sequences) - dropped - len(train_keep),
        "seed": seed,
    }
    assert_no_leakage([train_sequences[i] for i in train], test_sequences,
                      [train_sequences[i] for i in val])
    return Split(train, val, [], audit), audit


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def flatten(embeddings: Sequence[np.ndarray], labels: np.ndarray,
            mask: np.ndarray, indices: Optional[Sequence[int]] = None,
            max_rows: Optional[int] = None, seed: int = 42,
            return_chains: bool = False,
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-chain residue embeddings and labels into ``(X, y)``.

    ``mask`` decides which residues are used at all, so passing the evaluation
    mask here is what keeps the protocol comparable. Rows where the embedding is
    shorter than the label array (a truncated chain) are dropped rather than
    silently misaligned.

    ``return_chains`` additionally returns the owning chain index per residue,
    which is what :func:`bootstrap_classification_by_chain` resamples. It is an
    option on this function rather than a second flattener because a separate
    implementation could drift from the masking rules above, and then the
    confidence intervals would describe different residues than the point
    estimate.
    """
    indices = list(range(len(embeddings)) if indices is None else indices)

    def keep_mask(position, chain):
        emb = embeddings[position]
        if emb is None or emb.shape[0] == 0:
            return emb, 0, None
        usable = min(emb.shape[0], labels.shape[1], mask.shape[1])
        keep = mask[chain, :usable].copy()
        if labels.dtype.kind in "iu":
            keep &= labels[chain, :usable] != IGNORE_INDEX
        else:
            keep &= np.isfinite(labels[chain, :usable])
        return emb, usable, keep

    total = 0
    truncated = 0
    for position, chain in enumerate(indices):
        emb, _, keep = keep_mask(position, chain)
        if keep is None:
            continue
        if emb.shape[0] < int(mask[chain].sum()):
            truncated += 1
        total += int(keep.sum())

    selected = None
    if max_rows is not None and total > max_rows:
        selected = np.sort(
            np.random.default_rng(seed).choice(total, size=max_rows, replace=False))
        _log.info("subsampled %d/%d probe residues", max_rows, total)

    xs, ys, chains = [], [], []
    offset = 0
    for position, chain in enumerate(indices):
        emb, usable, keep = keep_mask(position, chain)
        if keep is None:
            continue
        count = int(keep.sum())
        if count == 0:
            continue
        residue_indices = np.flatnonzero(keep)
        if selected is not None:
            lo = np.searchsorted(selected, offset, side="left")
            hi = np.searchsorted(selected, offset + count, side="left")
            if lo == hi:
                offset += count
                continue
            residue_indices = residue_indices[selected[lo:hi] - offset]
        xs.append(emb[:usable][residue_indices])
        ys.append(labels[chain, :usable][residue_indices])
        chains.append(np.full(residue_indices.size, chain, dtype=np.int64))
        offset += count

    if truncated:
        _log.warning("%d chain(s) had embeddings shorter than their labels; "
                     "the missing tail was excluded", truncated)
    if not xs:
        empty = (np.zeros((0, 0), dtype=np.float32), np.zeros((0,)))
        return empty + (np.zeros((0,), dtype=np.int64),) if return_chains else empty
    stacked = (np.concatenate(xs).astype(np.float32, copy=False),
               np.concatenate(ys))
    if return_chains:
        return stacked + (np.concatenate(chains),)
    return stacked


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           n_classes: int, probs: Optional[np.ndarray] = None,
                           ) -> Dict[str, float]:
    """Accuracy, macro-F1 and the majority-class floor.

    The floor is reported alongside because SS3 is ~43% coil: an accuracy of
    0.55 is a much weaker result than it looks without it.
    """
    out: Dict[str, float] = {}
    out["accuracy"] = float((y_true == y_pred).mean()) if y_true.size else float("nan")

    f1s, per_class = [], {}
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                        "support": int((y_true == c).sum())}
    out["macro_f1"] = float(np.mean(f1s))
    out["per_class"] = per_class

    if y_true.size:
        counts = np.bincount(y_true.astype(np.int64), minlength=n_classes)
        out["majority_floor"] = float(counts.max() / counts.sum())

    if n_classes == 2:
        out["mcc"] = float(_mcc(y_true, y_pred))
        if probs is not None:
            out["auroc"] = float(_auroc(y_true, probs))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"pearson_r": float("nan"), "mae": float("nan")}
    return {"pearson_r": float(_pearson(y_true, y_pred)),
            "mae": float(np.abs(y_true - y_pred).mean()),
            "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean()))}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


# ---------------------------------------------------------------------------
# Chain-level bootstrap
# ---------------------------------------------------------------------------
#
# Residues within a chain are strongly correlated: secondary structure runs in
# segments, so 200 residues of one protein carry nowhere near 200 residues of
# independent evidence. Resampling residues would therefore report an interval
# several times too narrow. The resampling unit here is the chain.
#
# Both estimators below are computed *exactly* rather than by re-scoring each
# replicate. Every metric involved is a function of per-chain sufficient
# statistics that add across chains: a confusion matrix for classification,
# and the moment sums for regression. So a bootstrap replicate is a weighted
# sum of per-chain statistics, which makes 10,000 replicates one matrix
# multiply instead of 10,000 passes over the residues.

#: Replicates per matmul block. Bounds peak memory at roughly
#: ``block x n_chains x 8`` bytes regardless of the replicate count.
_BOOTSTRAP_BLOCK = 2000


def _summary(point: float, replicates: np.ndarray, seed: int) -> Dict:
    """Package a point estimate and its bootstrap interval.

    Keys match ``bootstrap_mean_interval`` in the retrieval benchmark so both
    suites can share one formatter. ``mean`` is the estimate on the observed
    data, not the mean of the replicates.
    """
    finite = replicates[np.isfinite(replicates)]
    if not np.isfinite(point) or finite.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None,
                "bootstrap_replicates": int(finite.size), "bootstrap_seed": seed}
    low, high = np.quantile(finite, [0.025, 0.975])
    return {"mean": float(point), "ci95_low": float(low), "ci95_high": float(high),
            "bootstrap_replicates": int(finite.size), "bootstrap_seed": seed}


def _chain_positions(chain_index: np.ndarray) -> Tuple[np.ndarray, int]:
    """Map arbitrary chain ids onto a contiguous 0..n-1 range."""
    unique, positions = np.unique(np.asarray(chain_index), return_inverse=True)
    return positions, unique.size


def _resample_counts(n_chains: int, replicates: int, seed: int):
    """Yield blocks of multinomial chain counts.

    Drawing ``n_chains`` chains with replacement is exactly a multinomial over
    the chains, so the resampled statistics are a counts-by-statistics matmul.
    """
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_chains, 1.0 / n_chains)
    remaining = replicates
    while remaining > 0:
        size = min(remaining, _BOOTSTRAP_BLOCK)
        yield rng.multinomial(n_chains, probabilities, size=size).astype(np.float64)
        remaining -= size


def _metrics_from_confusions(matrices: np.ndarray) -> Dict[str, np.ndarray]:
    """Accuracy, macro-F1 and (binary only) MCC from ``[R, C, C]`` counts.

    Rows are true classes, columns predicted.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        total = matrices.sum(axis=(1, 2))
        true_positive = np.einsum("rcc->rc", matrices)
        predicted = matrices.sum(axis=1)
        actual = matrices.sum(axis=2)

        precision = np.where(predicted > 0, true_positive / predicted, 0.0)
        recall = np.where(actual > 0, true_positive / actual, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

        out = {
            "accuracy": np.where(total > 0, true_positive.sum(axis=1) / total, np.nan),
            "macro_f1": f1.mean(axis=1),
        }

        if matrices.shape[1] == 2:
            # Rows are true, columns predicted, so [1, 1] is a true positive
            # for the positive class and [0, 0] a true negative.
            tp = matrices[:, 1, 1]
            tn = matrices[:, 0, 0]
            fp = matrices[:, 0, 1]
            fn = matrices[:, 1, 0]
            product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            out["mcc"] = np.where(
                product > 0,
                (tp * tn - fp * fn) / np.sqrt(np.where(product > 0, product, 1.0)),
                0.0,
            )
    return out


def bootstrap_classification_by_chain(y_true: np.ndarray, y_pred: np.ndarray,
                                      chain_index: np.ndarray, n_classes: int,
                                      seed: int = 42, replicates: int = 10000,
                                      ) -> Dict[str, Dict]:
    """Chain-resampled intervals for accuracy, macro-F1 and binary MCC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    positions, n_chains = _chain_positions(chain_index)
    if y_true.size == 0 or n_chains == 0:
        return {}

    # One bin per (chain, true, predicted) triple.
    bins = (positions * n_classes + y_true) * n_classes + y_pred
    per_chain = np.bincount(
        bins, minlength=n_chains * n_classes * n_classes
    ).astype(np.float64).reshape(n_chains, n_classes * n_classes)

    observed = _metrics_from_confusions(
        per_chain.sum(axis=0).reshape(1, n_classes, n_classes))

    collected: Dict[str, List[np.ndarray]] = {}
    for counts in _resample_counts(n_chains, replicates, seed):
        matrices = (counts @ per_chain).reshape(-1, n_classes, n_classes)
        for name, values in _metrics_from_confusions(matrices).items():
            collected.setdefault(name, []).append(values)

    return {
        name: _summary(float(observed[name][0]), np.concatenate(blocks), seed)
        for name, blocks in collected.items()
    }


def bootstrap_regression_by_chain(y_true: np.ndarray, y_pred: np.ndarray,
                                  chain_index: np.ndarray, seed: int = 42,
                                  replicates: int = 10000) -> Dict[str, Dict]:
    """Chain-resampled intervals for Pearson r, MAE and RMSE.

    Pearson r is reconstructed from summed moments, so it is exact under chain
    resampling rather than an approximation.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    positions, n_chains = _chain_positions(chain_index)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any() or n_chains == 0:
        return {}
    y_true, y_pred, positions = y_true[valid], y_pred[valid], positions[valid]

    error = y_true - y_pred
    columns = [
        np.ones_like(y_true), y_true, y_pred,
        y_true * y_true, y_pred * y_pred, y_true * y_pred,
        np.abs(error), error * error,
    ]
    per_chain = np.stack(
        [np.bincount(positions, weights=column, minlength=n_chains)
         for column in columns],
        axis=1,
    )

    def evaluate(totals: np.ndarray) -> Dict[str, np.ndarray]:
        count, sx, sy, sxx, syy, sxy, sae, sse = totals.T
        with np.errstate(divide="ignore", invalid="ignore"):
            covariance = count * sxy - sx * sy
            spread = (count * sxx - sx * sx) * (count * syy - sy * sy)
            pearson = np.where(spread > 0,
                               covariance / np.sqrt(np.where(spread > 0, spread, 1.0)),
                               np.nan)
            return {
                "pearson_r": pearson,
                "mae": np.where(count > 0, sae / count, np.nan),
                "rmse": np.where(count > 0, np.sqrt(sse / np.where(count > 0, count, 1.0)),
                                 np.nan),
            }

    observed = evaluate(per_chain.sum(axis=0)[None, :])
    collected: Dict[str, List[np.ndarray]] = {}
    for counts in _resample_counts(n_chains, replicates, seed):
        for name, values in evaluate(counts @ per_chain).items():
            collected.setdefault(name, []).append(values)

    return {
        name: _summary(float(observed[name][0]), np.concatenate(blocks), seed)
        for name, blocks in collected.items()
    }


# ---------------------------------------------------------------------------
# Paired chain-level bootstrap
# ---------------------------------------------------------------------------
#
# Two arms scored on the *same* chains are a paired design, and comparing their
# two independent intervals throws that pairing away. Overlapping marginal
# intervals routinely hide a difference that is consistent chain by chain,
# because between-chain variance -- some proteins are simply harder -- dominates
# both intervals and cancels exactly in the difference.
#
# So resample the chains once per replicate and evaluate *both* arms on that
# same resample, then take the distribution of the difference. Same exact
# per-chain-sufficient-statistic trick as the unpaired estimators above, so this
# is still one matmul rather than a rescoring loop.


class PairingError(ValueError):
    """Raised when two arms were not scored on the same residues."""


def _paired_summary(point: float, replicates: np.ndarray, seed: int) -> Dict:
    """Delta point estimate, its interval, and a two-sided bootstrap p-value."""
    finite = replicates[np.isfinite(replicates)]
    if not np.isfinite(point) or finite.size == 0:
        return {"delta": None, "ci95_low": None, "ci95_high": None,
                "p_value": None, "bootstrap_replicates": int(finite.size),
                "bootstrap_seed": seed}
    low, high = np.quantile(finite, [0.025, 0.975])
    # Two-sided: how much of the resampled difference sits on the far side of
    # zero from the observed effect, doubled. Floored at 1/replicates because a
    # bootstrap cannot resolve a p-value finer than its own resolution, and
    # reporting p=0 would overstate what 10,000 replicates can show.
    below = float((finite <= 0).mean())
    above = float((finite >= 0).mean())
    p_value = min(1.0, 2.0 * min(below, above))
    p_value = max(p_value, 1.0 / finite.size)
    return {"delta": float(point), "ci95_low": float(low), "ci95_high": float(high),
            "p_value": float(p_value), "significant_at_05": bool(p_value < 0.05),
            "bootstrap_replicates": int(finite.size), "bootstrap_seed": seed}


def _check_pairing(chain_index_a: np.ndarray, chain_index_b: np.ndarray,
                   y_true_a: np.ndarray, y_true_b: np.ndarray) -> np.ndarray:
    """Refuse to pair two arms unless they cover identical residues in order."""
    chain_index_a = np.asarray(chain_index_a)
    chain_index_b = np.asarray(chain_index_b)
    if chain_index_a.shape != chain_index_b.shape:
        raise PairingError(
            f"arms cover different residue counts ({chain_index_a.size} vs "
            f"{chain_index_b.size}); a paired test requires the same residues. "
            "This usually means one arm truncated chains the other did not.")
    if not np.array_equal(chain_index_a, chain_index_b):
        raise PairingError("arms disagree on the chain of at least one residue; "
                           "the two runs used different masking or ordering")
    if not np.array_equal(np.asarray(y_true_a), np.asarray(y_true_b)):
        raise PairingError("arms disagree on the ground-truth labels; "
                           "the two runs are not scoring the same residues")
    return chain_index_a


def paired_bootstrap_classification_by_chain(
        y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
        chain_index: np.ndarray, n_classes: int, seed: int = 42,
        replicates: int = 10000) -> Dict[str, Dict]:
    """Chain-resampled interval on ``b - a`` for accuracy, macro-F1 and MCC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    positions, n_chains = _chain_positions(chain_index)
    if y_true.size == 0 or n_chains == 0:
        return {}

    def per_chain_confusions(predictions: np.ndarray) -> np.ndarray:
        bins = (positions * n_classes + y_true) * n_classes + predictions
        return np.bincount(
            bins, minlength=n_chains * n_classes * n_classes
        ).astype(np.float64).reshape(n_chains, n_classes * n_classes)

    chains_a = per_chain_confusions(pred_a)
    chains_b = per_chain_confusions(pred_b)

    observed_a = _metrics_from_confusions(
        chains_a.sum(axis=0).reshape(1, n_classes, n_classes))
    observed_b = _metrics_from_confusions(
        chains_b.sum(axis=0).reshape(1, n_classes, n_classes))

    collected: Dict[str, List[np.ndarray]] = {}
    for counts in _resample_counts(n_chains, replicates, seed):
        # The same `counts` drives both arms: that is the pairing.
        metrics_a = _metrics_from_confusions(
            (counts @ chains_a).reshape(-1, n_classes, n_classes))
        metrics_b = _metrics_from_confusions(
            (counts @ chains_b).reshape(-1, n_classes, n_classes))
        for name in metrics_a:
            collected.setdefault(name, []).append(metrics_b[name] - metrics_a[name])

    return {
        name: _paired_summary(
            float(observed_b[name][0]) - float(observed_a[name][0]),
            np.concatenate(blocks), seed)
        for name, blocks in collected.items()
    }


def paired_bootstrap_regression_by_chain(
        y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
        chain_index: np.ndarray, seed: int = 42,
        replicates: int = 10000) -> Dict[str, Dict]:
    """Chain-resampled interval on ``b - a`` for Pearson r, MAE and RMSE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    positions, n_chains = _chain_positions(chain_index)
    valid = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    if not valid.any() or n_chains == 0:
        return {}
    y_true = y_true[valid]
    pred_a, pred_b, positions = pred_a[valid], pred_b[valid], positions[valid]

    def per_chain_moments(predictions: np.ndarray) -> np.ndarray:
        error = y_true - predictions
        columns = [
            np.ones_like(y_true), y_true, predictions,
            y_true * y_true, predictions * predictions, y_true * predictions,
            np.abs(error), error * error,
        ]
        return np.stack(
            [np.bincount(positions, weights=column, minlength=n_chains)
             for column in columns],
            axis=1,
        )

    def evaluate(totals: np.ndarray) -> Dict[str, np.ndarray]:
        count, sx, sy, sxx, syy, sxy, sae, sse = totals.T
        with np.errstate(divide="ignore", invalid="ignore"):
            covariance = count * sxy - sx * sy
            spread = (count * sxx - sx * sx) * (count * syy - sy * sy)
            return {
                "pearson_r": np.where(
                    spread > 0,
                    covariance / np.sqrt(np.where(spread > 0, spread, 1.0)),
                    np.nan),
                "mae": np.where(count > 0, sae / count, np.nan),
                "rmse": np.where(count > 0,
                                 np.sqrt(sse / np.where(count > 0, count, 1.0)),
                                 np.nan),
            }

    chains_a = per_chain_moments(pred_a)
    chains_b = per_chain_moments(pred_b)
    observed_a = evaluate(chains_a.sum(axis=0)[None, :])
    observed_b = evaluate(chains_b.sum(axis=0)[None, :])

    collected: Dict[str, List[np.ndarray]] = {}
    for counts in _resample_counts(n_chains, replicates, seed):
        metrics_a = evaluate(counts @ chains_a)
        metrics_b = evaluate(counts @ chains_b)
        for name in metrics_a:
            collected.setdefault(name, []).append(metrics_b[name] - metrics_a[name])

    return {
        name: _paired_summary(
            float(observed_b[name][0]) - float(observed_a[name][0]),
            np.concatenate(blocks), seed)
        for name, blocks in collected.items()
    }


def paired_compare_task(task: str, arm_a: Dict, arm_b: Dict, seed: int = 42,
                        replicates: int = 10000) -> Dict[str, Dict]:
    """Paired comparison from two persisted prediction blocks.

    Each arm is the ``predictions`` payload written by :func:`probe_task`,
    holding ``y_true``, ``predictions`` and ``chains``.
    """
    # Validate the task first: otherwise an unknown task reaches the pairing
    # check and surfaces as a KeyError about the payload, which points at the
    # wrong problem.
    if task not in CLASSIFICATION_TASKS and task not in REGRESSION_TASKS:
        raise ValueError(f"unknown task {task!r}")
    chains = _check_pairing(arm_a["chains"], arm_b["chains"],
                            arm_a["y_true"], arm_b["y_true"])
    if task in CLASSIFICATION_TASKS:
        return paired_bootstrap_classification_by_chain(
            arm_a["y_true"], arm_a["predictions"], arm_b["predictions"],
            chains, CLASSIFICATION_TASKS[task], seed=seed, replicates=replicates)
    return paired_bootstrap_regression_by_chain(
        arm_a["y_true"], arm_a["predictions"], arm_b["predictions"],
        chains, seed=seed, replicates=replicates)


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with ties averaged."""
    positives = y_true == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """Shared probe hyperparameters.

    Held identical across every representation and every baseline: if the probe
    differs, the comparison measures the probe rather than the representation.
    """

    epochs: int = 100
    batch_size: int = 4096
    lr: float = 1e-2
    weight_decay: float = 1e-4
    standardize: bool = True
    class_weight: bool = False
    patience: int = 10
    seed: int = 42
    device: Optional[str] = None
    verbose: bool = False
    # A linear probe that has not converged understates the representation it is
    # measuring, which would look exactly like "the encoder lost the signal".
    # Small splits get one optimizer step per epoch, so the step count -- not the
    # epoch count -- is what has to be guaranteed.
    min_steps: int = 2000
    # Empty means a linear readout. Non-empty builds an MLP with these hidden
    # widths, which is the only thing the readout-capacity ladder varies: a probe
    # reports what a *readout* can extract, not what a representation contains,
    # so a single linear number is a lower bound. Every other hyperparameter is
    # held fixed across the ladder, otherwise the rungs differ by optimisation
    # rather than by capacity.
    hidden_dims: Tuple[int, ...] = ()
    dropout: float = 0.0

    def describe(self) -> str:
        """Short protocol label, so results record which rung produced them."""
        if not self.hidden_dims:
            return "linear"
        return "mlp:" + "-".join(str(width) for width in self.hidden_dims)


class Probe:
    """A readout over frozen features, on GPU when available.

    Linear by default. ``ProbeConfig.hidden_dims`` turns it into an MLP without
    changing anything else, which is what makes the readout-capacity ladder a
    controlled comparison rather than three loosely related experiments.
    """

    def __init__(self, task: str, config: Optional[ProbeConfig] = None):
        if task not in CLASSIFICATION_TASKS and task not in REGRESSION_TASKS:
            raise ValueError(f"unknown task {task!r}")
        self.task = task
        self.config = config or ProbeConfig()
        self.is_classification = task in CLASSIFICATION_TASKS
        self.n_classes = CLASSIFICATION_TASKS.get(task)
        self._weights = None
        self._mean = None
        self._std = None

    # -- internals ---------------------------------------------------------
    def _torch(self):
        import torch
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return torch, torch.device(device)

    def _build(self, torch, n_features: int, out_dim: int):
        widths = tuple(self.config.hidden_dims)
        if not widths:
            return torch.nn.Linear(n_features, out_dim)
        layers: List = []
        previous = n_features
        for width in widths:
            layers.append(torch.nn.Linear(previous, width))
            layers.append(torch.nn.GELU())
            if self.config.dropout > 0:
                layers.append(torch.nn.Dropout(self.config.dropout))
            previous = width
        layers.append(torch.nn.Linear(previous, out_dim))
        return torch.nn.Sequential(*layers)

    def _prepare(self, X, fit: bool):
        X = np.asarray(X, dtype=np.float32)
        if self.config.standardize and fit:
            self._mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True)
            # Constant features would divide by zero; leave them at zero variance.
            self._std = np.where(std < 1e-6, 1.0, std)
        return X

    # -- public ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        torch, device = self._torch()
        torch.manual_seed(self.config.seed)

        X = self._prepare(np.asarray(X, dtype=np.float32), fit=True)
        n_features = X.shape[1]
        out_dim = self.n_classes if self.is_classification else 1

        layer = self._build(torch, n_features, out_dim).to(device)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=self.config.lr,
                                      weight_decay=self.config.weight_decay)

        xt = torch.from_numpy(X)
        mean = (torch.from_numpy(self._mean).to(device)
                if self._mean is not None else None)
        std = (torch.from_numpy(self._std).to(device)
               if self._std is not None else None)
        if self.is_classification:
            yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
            weight = None
            if self.config.class_weight:
                counts = np.bincount(np.asarray(y, dtype=np.int64),
                                     minlength=self.n_classes).astype(np.float64)
                inverse = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
                weight = torch.tensor(inverse / inverse.sum() * self.n_classes,
                                      dtype=torch.float32, device=device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)
            loss_fn = torch.nn.MSELoss()

        n = xt.shape[0]
        steps_per_epoch = max(1, (n + self.config.batch_size - 1) // self.config.batch_size)
        epochs = max(self.config.epochs,
                     -(-self.config.min_steps // steps_per_epoch))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs))
        best_score, best_state, stale = None, None, 0

        for epoch in range(epochs):
            layer.train()
            permutation = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.config.batch_size):
                idx = permutation[start:start + self.config.batch_size]
                xb = xt[idx].to(device, non_blocking=True)
                if mean is not None:
                    xb = (xb - mean) / std
                yb = yt[idx].to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = loss_fn(layer(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            scheduler.step()

            if X_val is not None and len(X_val):
                self._weights = layer
                score = self._val_score(X_val, y_val)
                if best_score is None or score > best_score:
                    best_score, stale = score, 0
                    best_state = {k: v.detach().clone() for k, v in layer.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.config.patience:
                        if self.config.verbose:
                            _log.info("early stop at epoch %d (best %.4f)", epoch, best_score)
                        break
            if self.config.verbose:
                _log.info("epoch %d loss %.4f", epoch, total / max(n, 1))

        if best_state is not None:
            layer.load_state_dict(best_state)
        self._weights = layer
        self.fit_info = {"epochs_run": epoch + 1, "steps_per_epoch": steps_per_epoch,
                         "n_train": int(n), "readout": self.config.describe(),
                         "n_readout_params": int(sum(p.numel()
                                                     for p in layer.parameters()))}
        return self

    def _val_score(self, X_val, y_val) -> float:
        metrics = self.evaluate(X_val, y_val)
        if self.is_classification:
            return metrics["accuracy"]
        return -metrics["mae"]

    def predict(self, X: np.ndarray):
        if self._weights is None:
            raise RuntimeError("probe is not fitted")
        torch, device = self._torch()
        X = self._prepare(X, fit=False)
        mean = (torch.from_numpy(self._mean).to(device)
                if self._mean is not None else None)
        std = (torch.from_numpy(self._std).to(device)
               if self._std is not None else None)
        self._weights.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), self.config.batch_size):
                chunk = torch.from_numpy(X[start:start + self.config.batch_size]).to(device)
                if mean is not None:
                    chunk = (chunk - mean) / std
                outputs.append(self._weights(chunk).float().cpu().numpy())
        logits = np.concatenate(outputs) if outputs else np.zeros((0, 1), dtype=np.float32)
        if self.is_classification:
            return logits.argmax(axis=1), logits
        return logits.squeeze(-1), logits

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        predictions, logits = self.predict(X)
        if self.is_classification:
            probs = None
            if self.n_classes == 2:
                shifted = logits - logits.max(axis=1, keepdims=True)
                exponentiated = np.exp(shifted)
                probs = (exponentiated / exponentiated.sum(axis=1, keepdims=True))[:, 1]
            return classification_metrics(np.asarray(y, dtype=np.int64), predictions,
                                          self.n_classes, probs)
        return regression_metrics(np.asarray(y, dtype=np.float64), predictions)


#: Historical name. The class is linear whenever ``hidden_dims`` is empty, which
#: is the default, so existing callers keep the exact behaviour they had.
LinearProbe = Probe


def probe_task(task: str, features: Dict[str, np.ndarray],
               labels: Dict[str, np.ndarray],
               config: Optional[ProbeConfig] = None,
               test_chains: Optional[np.ndarray] = None,
               bootstrap_replicates: int = 0) -> Dict[str, object]:
    """Fit on ``train``, select on ``val``, report on ``test``.

    Passing ``test_chains`` (from ``flatten(..., return_chains=True)``) with a
    non-zero ``bootstrap_replicates`` adds an ``intervals`` block holding
    chain-resampled 95% confidence intervals. Chains, not residues, are the
    resampling unit: residues within a protein are correlated, so a
    residue-level interval would be far too narrow.
    """
    config = config or ProbeConfig()
    probe = Probe(task, config)
    probe.fit(features["train"], labels["train"],
              features.get("val"), labels.get("val"))
    metrics = probe.evaluate(features["test"], labels["test"])
    metrics["task"] = task
    metrics["n_train"] = int(len(labels["train"]))
    metrics["n_test"] = int(len(labels["test"]))
    metrics["feature_dim"] = int(features["train"].shape[1])
    metrics["readout"] = config.describe()

    if test_chains is not None:
        seed = config.seed
        predictions, _ = probe.predict(features["test"])
        chains = np.asarray(test_chains)
        # Kept under a private key so the JSON writer can drop it: a paired
        # comparison against another arm needs the raw per-residue predictions,
        # and recomputing them later would require re-fitting the probe.
        metrics["_predictions"] = {
            "y_true": np.asarray(labels["test"]),
            "predictions": predictions,
            "chains": chains,
        }
        metrics["n_test_chains"] = int(np.unique(chains).size)

    if test_chains is not None and bootstrap_replicates > 0:
        seed = config.seed
        predictions = metrics["_predictions"]["predictions"]
        chains = metrics["_predictions"]["chains"]
        if probe.is_classification:
            metrics["intervals"] = bootstrap_classification_by_chain(
                labels["test"], predictions, chains, probe.n_classes,
                seed=seed, replicates=bootstrap_replicates)
        else:
            metrics["intervals"] = bootstrap_regression_by_chain(
                labels["test"], predictions, chains,
                seed=seed, replicates=bootstrap_replicates)
        metrics["n_test_chains"] = int(np.unique(chains).size)

    # Train-set score is reported so underfitting cannot be mistaken for a
    # representation that lacks the signal.
    train_metrics = probe.evaluate(features["train"], labels["train"])
    key = "accuracy" if probe.is_classification else "pearson_r"
    metrics[f"train_{key}"] = train_metrics[key]
    metrics["fit_info"] = getattr(probe, "fit_info", {})
    return metrics
