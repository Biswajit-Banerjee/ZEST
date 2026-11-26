"""Tests for the NetSurfP loader (2.0 and 3.0 share one file format).

The official ``.npz`` is not in the repo, so the npz path is exercised against a
synthetic array built in the documented layout. That is the point of these
tests: they prove the layout verifier *accepts* a correct file and *rejects*
shifted or inconsistent ones, so when the real file lands on the cluster a
layout mismatch fails immediately instead of yielding invalid metrics.

Two facts the layout makes easy to get wrong, both covered below:

* there is **no SS3 block** -- Q3 is derived from Q8 via GHIBESTC -> HHHEECCC;
* column 51 flags **ordered** residues, so the disorder target is its
  complement. Reading it as-is would make the rare class the majority.
"""
from pathlib import Path

import numpy as np
import pytest

from zest.data.netsurfp import (COL_ASA, COL_ASA_COMPLEXED, COL_ASA_MAX,
                                COL_EVAL_MASK, COL_ONEHOT, COL_ORDERED, COL_PHI,
                                COL_PROFILE, COL_PSI, COL_RSA,
                                COL_RSA_COMPLEXED, COL_SEQ_MASK, COL_SS8,
                                IGNORE_INDEX, N_FEATURES, NSP2_AA_ORDER,
                                Q8_IDX_TO_Q3_IDX, Q8_TO_Q3, SS3_ALPHABET,
                                SS3_TO_IDX, SS8_ALPHABET, SS8_TO_IDX,
                                check_no_leakage, load_cb513_csv, load_netsurfp,
                                load_npz)

CB513_CSV = Path("data/cb513/CB513.csv")
requires_cb513 = pytest.mark.skipif(not CB513_CSV.exists(),
                                    reason="data/cb513/CB513.csv not present")

# Reference areas roughly matching real ASA_max values, so RSA = ASA / ASA_max
# is a meaningful relation rather than a tautology on constants.
_ASA_MAX = np.linspace(110.0, 230.0, 20).astype(np.float32)


def _synthetic(n=6, width=40, lengths=None, seed=0, unknown_frac=0.0):
    """Build an [N, L, 68] array in the documented NetSurfP layout."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, width, N_FEATURES), dtype=np.float32)
    lengths = lengths if lengths is not None else [width, width - 5, 20, 33, 12, width]
    for i, length in enumerate(lengths):
        data[i, :length, COL_SEQ_MASK] = 1.0

        aa = rng.integers(0, 20, size=length)
        known = rng.random(length) >= unknown_frac
        rows = np.nonzero(known)[0]
        data[i, rows, aa[rows]] = 1.0        # unknown residues stay all-zero

        profile = rng.random((length, 30)).astype(np.float32)
        data[i, :length, COL_PROFILE] = profile / profile.sum(axis=1, keepdims=True)

        q8 = rng.integers(0, 8, size=length)
        data[i, np.arange(length), 57 + q8] = 1.0

        asa_max = _ASA_MAX[aa]
        rsa = rng.random(length).astype(np.float32)
        data[i, :length, COL_ASA_MAX] = asa_max
        data[i, :length, COL_RSA] = rsa
        data[i, :length, COL_ASA] = rsa * asa_max          # RSA == ASA / ASA_max
        data[i, :length, COL_RSA_COMPLEXED] = rsa * 0.9
        data[i, :length, COL_ASA_COMPLEXED] = rsa * 0.9 * asa_max

        data[i, :length, COL_PHI] = rng.uniform(-180, 180, length)
        data[i, :length, COL_PSI] = rng.uniform(-180, 180, length)
        # 1 = ordered, so most residues are 1.
        data[i, :length, COL_ORDERED] = (rng.random(length) > 0.1).astype(np.float32)
        data[i, :length, COL_EVAL_MASK] = (rng.random(length) > 0.3).astype(np.float32)
    return data, np.array(lengths)


def _write(tmp_path, data, name="split.npz", key="data"):
    path = tmp_path / name
    np.savez_compressed(path, **{key: data})
    return path


# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------

def test_q8_to_q3_table_matches_the_published_mapping():
    """The dataset page states GHIBESTC -> HHHEECCC."""
    assert SS8_ALPHABET == "GHIBESTC"
    assert "".join(Q8_TO_Q3[c] for c in SS8_ALPHABET) == "HHHEECCC"
    assert set(Q8_TO_Q3) == set(SS8_ALPHABET)
    assert set(Q8_TO_Q3.values()) == set(SS3_ALPHABET)
    assert Q8_TO_Q3["B"] == "E", "isolated beta-bridge maps to E"


def test_q8_index_mapping_agrees_with_the_letter_mapping():
    for i, letter in enumerate(SS8_ALPHABET):
        assert Q8_IDX_TO_Q3_IDX[i] == SS3_TO_IDX[Q8_TO_Q3[letter]]


# ---------------------------------------------------------------------------
# npz path
# ---------------------------------------------------------------------------

def test_npz_round_trip_recovers_sequence_and_labels(tmp_path):
    data, lengths = _synthetic()
    dataset = load_npz(_write(tmp_path, data), with_profile=True)

    assert len(dataset) == len(lengths)
    assert np.array_equal(dataset.lengths, lengths)
    assert [len(s) for s in dataset.sequences] == list(lengths)

    for i, length in enumerate(lengths):
        expected = "".join(NSP2_AA_ORDER[k]
                           for k in data[i, :length, COL_ONEHOT].argmax(axis=1))
        assert dataset.sequences[i] == expected
        assert np.array_equal(dataset.ss8[i, :length],
                              data[i, :length, COL_SS8].argmax(axis=1))
        assert (dataset.ss8[i, length:] == IGNORE_INDEX).all()
    assert dataset.profile.shape[2] == 30


def test_ss3_is_derived_from_q8_because_the_file_has_no_ss3_block(tmp_path):
    data, _ = _synthetic()
    dataset = load_npz(_write(tmp_path, data))
    valid = dataset.ss8 != IGNORE_INDEX
    assert np.array_equal(dataset.ss3[valid], Q8_IDX_TO_Q3_IDX[dataset.ss8[valid]])
    assert (dataset.ss3[~valid] == IGNORE_INDEX).all()


def test_disorder_is_the_complement_of_the_ordered_column(tmp_path):
    """Column 51 is 1 for ordered residues; the target is disorder."""
    data, lengths = _synthetic()
    dataset = load_npz(_write(tmp_path, data))
    for i, length in enumerate(lengths):
        ordered = data[i, :length, COL_ORDERED] > 0.5
        assert np.array_equal(dataset.disorder[i, :length], (~ordered).astype(np.int64))
    scored = dataset.disorder[dataset.seq_mask]
    assert scored.mean() < 0.5, "disorder must be the rare class"


def test_rsa_and_asa_are_read_from_the_right_columns(tmp_path):
    data, lengths = _synthetic()
    dataset = load_npz(_write(tmp_path, data))
    for i, length in enumerate(lengths):
        assert np.allclose(dataset.rsa[i, :length], data[i, :length, COL_RSA], atol=1e-5)
        assert np.allclose(dataset.asa[i, :length], data[i, :length, COL_ASA], atol=1e-3)
        assert np.allclose(dataset.phi[i, :length], data[i, :length, COL_PHI], atol=1e-3)
        assert np.allclose(dataset.psi[i, :length], data[i, :length, COL_PSI], atol=1e-3)


def test_unknown_residues_decode_as_X(tmp_path):
    """All-zero one-hot rows would otherwise argmax to alanine."""
    data, _ = _synthetic(unknown_frac=0.25, seed=3)
    dataset = load_npz(_write(tmp_path, data))
    assert any("X" in s for s in dataset.sequences)


def test_npz_regression_targets_are_nan_only_on_padding(tmp_path):
    data, _ = _synthetic()
    dataset = load_npz(_write(tmp_path, data))
    assert np.isfinite(dataset.rsa[dataset.seq_mask]).all()
    assert np.isnan(dataset.rsa[~dataset.seq_mask]).all()


def test_eval_mask_is_a_subset_of_real_residues(tmp_path):
    data, _ = _synthetic()
    dataset = load_npz(_write(tmp_path, data))
    assert not (dataset.eval_mask & ~dataset.seq_mask).any()
    assert dataset.n_scored < int(dataset.seq_mask.sum())

    unmasked = load_npz(_write(tmp_path, data, "b.npz"), use_eval_mask=False)
    assert unmasked.n_scored == int(unmasked.seq_mask.sum())


def test_max_length_filter_drops_long_chains(tmp_path):
    data, lengths = _synthetic()
    dataset = load_npz(_write(tmp_path, data), max_length=30)
    assert len(dataset) == int((lengths <= 30).sum())
    assert dataset.lengths.max() <= 30


def test_profile_is_omitted_unless_requested(tmp_path):
    data, _ = _synthetic()
    assert load_npz(_write(tmp_path, data)).profile is None


def test_array_key_is_autodetected(tmp_path):
    data, _ = _synthetic()
    path = tmp_path / "odd.npz"
    np.savez_compressed(path, pdbids=np.zeros((3, 3)), Train_HHblits=data)
    assert len(load_npz(path)) == 6


# ---------------------------------------------------------------------------
# Layout verification must reject bad files
# ---------------------------------------------------------------------------

def test_rejects_wrong_feature_width(tmp_path):
    data, _ = _synthetic()
    narrow = _write(tmp_path, data[:, :, :60])
    with pytest.raises(KeyError, match="no \\[N, L, 68\\] array"):
        load_npz(narrow)
    with pytest.raises(ValueError, match="expected \\[N, L, 68\\]"):
        load_npz(narrow, key="data")


def test_rejects_column_shift(tmp_path):
    data, _ = _synthetic()
    shifted = np.roll(data, shift=1, axis=2)
    with pytest.raises(ValueError, match="sequence mask|layout verification failed"):
        load_npz(_write(tmp_path, shifted))


def test_rejects_shift_that_preserves_the_sequence_mask(tmp_path):
    """Subtler than a whole-array roll: the mask column survives, so rejection
    must come from the one-hot, Q8 and RSA/ASA checks instead."""
    data, _ = _synthetic()
    corrupted = data.copy()
    corrupted[:, :, 51:] = np.roll(data[:, :, 51:], shift=1, axis=2)
    with pytest.raises(ValueError, match="layout verification failed"):
        load_npz(_write(tmp_path, corrupted))


def test_rejects_broken_rsa_asa_relation(tmp_path):
    """RSA == ASA / ASA_max pins columns 53, 55 and 67 simultaneously."""
    data, _ = _synthetic()
    data[:, :, COL_ASA] *= 3.0
    with pytest.raises(ValueError, match="RSA != ASA/ASA_max"):
        load_npz(_write(tmp_path, data))


def test_rejects_swapped_rsa_and_asa_columns(tmp_path):
    """The exact mistake this loader previously made: RSA read one column off."""
    data, _ = _synthetic()
    swapped = data.copy()
    swapped[:, :, COL_RSA] = data[:, :, COL_RSA_COMPLEXED]
    swapped[:, :, COL_RSA_COMPLEXED] = data[:, :, COL_RSA]
    with pytest.raises(ValueError, match="RSA != ASA/ASA_max"):
        load_npz(_write(tmp_path, swapped))


def test_rejects_out_of_range_rsa(tmp_path):
    data, _ = _synthetic()
    data[:, :, COL_RSA] *= 500.0
    with pytest.raises(ValueError, match="RSA"):
        load_npz(_write(tmp_path, data))


def test_rejects_invalid_q8_block(tmp_path):
    data, _ = _synthetic()
    data[:, :, COL_SS8] = 0.0
    with pytest.raises(ValueError, match="SS8 one-hot invalid"):
        load_npz(_write(tmp_path, data))


def test_rejects_all_zero_sequence_mask(tmp_path):
    data, _ = _synthetic()
    data[:, :, COL_SEQ_MASK] = 0.0
    with pytest.raises(ValueError, match="sequence mask"):
        load_npz(_write(tmp_path, data))


def test_rejects_non_binary_ordered_mask(tmp_path):
    data, _ = _synthetic()
    data[:, :, COL_ORDERED] = 0.42
    with pytest.raises(ValueError, match="ordered mask"):
        load_npz(_write(tmp_path, data))


def test_lenient_mode_warns_instead_of_raising(tmp_path):
    data, _ = _synthetic()
    data[:, :, COL_RSA] *= 500.0
    assert len(load_npz(_write(tmp_path, data), strict=False)) == 6


def test_unknown_residues_do_not_trip_the_one_hot_check(tmp_path):
    """All-zero rows are legal for unknown residues."""
    data, _ = _synthetic(unknown_frac=0.2, seed=5)
    assert len(load_npz(_write(tmp_path, data))) == 6


# ---------------------------------------------------------------------------
# CB513 CSV path (real data)
# ---------------------------------------------------------------------------

@requires_cb513
def test_cb513_csv_loads_and_aligns():
    dataset = load_cb513_csv(CB513_CSV)
    assert len(dataset) == 511
    assert int(dataset.seq_mask.sum()) == 144011
    for i in range(len(dataset)):
        assert int(dataset.lengths[i]) == len(dataset.sequences[i])
        assert (dataset.ss3[i, :dataset.lengths[i]] != IGNORE_INDEX).all()


@requires_cb513
def test_cb513_mask_restricts_scoring():
    masked = load_cb513_csv(CB513_CSV, use_eval_mask=True)
    every = load_cb513_csv(CB513_CSV, use_eval_mask=False)
    assert masked.n_scored == 83991
    assert every.n_scored == 144011


@requires_cb513
def test_cb513_q8_q3_consistency_holds_on_real_data():
    dataset = load_cb513_csv(CB513_CSV, strict=True)
    valid = (dataset.ss3 != IGNORE_INDEX) & (dataset.ss8 != IGNORE_INDEX)
    assert np.array_equal(Q8_IDX_TO_Q3_IDX[dataset.ss8[valid]], dataset.ss3[valid])
    assert int(valid.sum()) == 144011


@requires_cb513
def test_cb513_disorder_is_the_rare_class():
    """~97.6% of CB513 residues are ordered, so disorder must be ~2.4%."""
    dataset = load_cb513_csv(CB513_CSV)
    scored = dataset.disorder[dataset.seq_mask]
    assert 0.0 < scored.mean() < 0.10


@requires_cb513
def test_cb513_majority_class_floor_is_reported():
    balance = load_cb513_csv(CB513_CSV).class_balance("ss3")
    assert abs(sum(balance.values()) - 1.0) < 1e-6
    assert 0.42 < max(balance.values()) < 0.44


@requires_cb513
def test_dispatch_by_extension():
    assert len(load_netsurfp(CB513_CSV)) == 511


def test_csv_rejects_misaligned_labels(tmp_path):
    import pandas as pd

    path = tmp_path / "bad.csv"
    pd.DataFrame({"input": ["ACDEF"], "dssp3": ["HHH"], "dssp8": ["HHHHH"]}).to_csv(
        path, index=False)
    with pytest.raises(ValueError, match="length mismatch"):
        load_cb513_csv(path)


def test_csv_rejects_labels_violating_the_mapping(tmp_path):
    import pandas as pd

    path = tmp_path / "bad.csv"
    pd.DataFrame({"input": ["ACDEF"], "dssp3": ["EEEEE"], "dssp8": ["HHHHH"]}).to_csv(
        path, index=False)
    with pytest.raises(ValueError, match="Q8->Q3"):
        load_cb513_csv(path)


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "x.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported extension"):
        load_netsurfp(path)


# ---------------------------------------------------------------------------
# Redundancy and leakage
# ---------------------------------------------------------------------------

def test_leakage_check_detects_shared_sequences():
    assert check_no_leakage(["AAA", "BBB"], ["BBB", "CCC"])["n_exact_duplicates"] == 1
    assert check_no_leakage(["AAA"], ["CCC"])["n_exact_duplicates"] == 0


@requires_cb513
def test_cb513_redundancy_is_measured_not_assumed_away():
    """79 rows repeat an earlier sequence and carry 26% of all residues."""
    report = load_cb513_csv(CB513_CSV).duplicate_report()
    assert report["n_chains"] == 511
    assert report["n_unique"] == 432
    assert report["n_redundant_rows"] == 79
    assert 0.25 < report["redundant_fraction"] < 0.27


@requires_cb513
def test_cb513_duplicates_carry_identical_labels():
    dataset = load_cb513_csv(CB513_CSV)
    first = {}
    for i, seq in enumerate(dataset.sequences):
        if seq in first:
            j, length = first[seq], len(seq)
            assert np.array_equal(dataset.ss3[i, :length], dataset.ss3[j, :length])
        else:
            first[seq] = i


@requires_cb513
def test_deduplicate_yields_distinct_sequences():
    unique = load_cb513_csv(CB513_CSV).deduplicate()
    assert len(unique) == 432 == len(set(unique.sequences))
    for i in range(len(unique)):
        length = int(unique.lengths[i])
        assert length == len(unique.sequences[i])
        assert (unique.ss3[i, length:] == IGNORE_INDEX).all()


@requires_cb513
def test_drop_unscored_chains_removes_the_fully_masked_pair():
    dataset = load_cb513_csv(CB513_CSV)
    kept = dataset.drop_unscored_chains()
    assert len(kept) == len(dataset) - 2
    assert kept.n_scored == dataset.n_scored


def test_select_is_order_preserving_and_retrims(tmp_path):
    data, lengths = _synthetic()
    dataset = load_npz(_write(tmp_path, data), with_profile=True)
    subset = dataset.select([2, 4])
    assert subset.sequences == [dataset.sequences[2], dataset.sequences[4]]
    assert subset.ss3.shape[1] == max(int(lengths[2]), int(lengths[4]))
    assert subset.profile.shape[0] == 2
