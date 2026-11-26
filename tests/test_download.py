"""Tests for automatic dataset fetching.

No test here touches the network. What matters is the logic around the download:
a cached file must be reused, a truncated or HTML-error-page file must be
rejected rather than trusted, and a killed job must never leave a partial file
that later looks valid. Those are the failure modes that waste a reservation.
"""
from pathlib import Path

import numpy as np
import pytest

from zest.data.download import (DATASETS, DatasetUnavailable, MIN_PLAUSIBLE_BYTES,
                                NSP2_BASE, NSP3_BASE, _is_valid_npz,
                                ensure_cb513_csv, ensure_dataset, resolve_split)


def _valid_npz(path, n=20, width=200):
    """An npz shaped right and comfortably over the size floor.

    ``MIN_PLAUSIBLE_BYTES`` is 100 KB, so the fixture has to be a realistic
    size; random data is used because it does not compress away.
    """
    rng = np.random.default_rng(0)
    data = rng.random((n, width, 68)).astype(np.float32)
    np.savez(path, data=data)
    assert path.stat().st_size > MIN_PLAUSIBLE_BYTES
    return path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_points_at_netsurfp_3():
    assert "NetSurfP-3.0" in NSP3_BASE
    assert "NetSurfP-2.0" in NSP2_BASE


def test_registry_covers_the_splits_the_benchmark_needs():
    for name in ("train", "cb513", "ts115", "casp12"):
        assert name in DATASETS
    # HHblits is the variant published accuracies are reported on.
    assert DATASETS["train"]["file"] == "Train_HHblits.npz"
    assert DATASETS["cb513"]["file"] == "CB513_HHblits.npz"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_valid_npz_is_accepted(tmp_path):
    assert _is_valid_npz(_valid_npz(tmp_path / "ok.npz"))


def test_truncated_file_is_rejected(tmp_path):
    path = tmp_path / "small.npz"
    path.write_bytes(b"\x00" * 128)
    assert not _is_valid_npz(path)


def test_html_error_page_is_rejected(tmp_path):
    """A server returning an error page would otherwise be cached as data."""
    path = tmp_path / "err.npz"
    path.write_bytes(b"<html><body>404 Not Found</body></html>" * 5000)
    assert not _is_valid_npz(path)


def test_npz_with_wrong_feature_width_is_rejected(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, data=np.zeros((4, 30, 40), dtype=np.float32))
    assert not _is_valid_npz(path)


def test_missing_file_is_rejected(tmp_path):
    assert not _is_valid_npz(tmp_path / "absent.npz")


def test_size_floor_is_large_enough_to_exclude_error_pages():
    assert MIN_PLAUSIBLE_BYTES >= 100_000


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_cached_file_is_reused_without_downloading(tmp_path):
    """The decisive caching test: a valid cached file must be returned even with
    downloading disabled, so repeat jobs cost nothing."""
    _valid_npz(tmp_path / DATASETS["cb513"]["file"])
    path = ensure_dataset("cb513", root=tmp_path, allow_download=False)
    assert path.exists()
    assert path.name == DATASETS["cb513"]["file"]


def test_missing_file_with_download_disabled_raises_with_the_url(tmp_path):
    with pytest.raises(DatasetUnavailable, match="NetSurfP-3.0"):
        ensure_dataset("cb513", root=tmp_path, allow_download=False)


def test_corrupt_cached_file_is_not_returned(tmp_path):
    """A truncated cache entry must trigger a refetch, not be handed back."""
    target = tmp_path / DATASETS["cb513"]["file"]
    target.write_bytes(b"\x00" * 256)
    with pytest.raises(DatasetUnavailable):
        ensure_dataset("cb513", root=tmp_path, allow_download=False)


def test_unknown_dataset_name_is_rejected(tmp_path):
    with pytest.raises(KeyError, match="unknown dataset"):
        ensure_dataset("cb999", root=tmp_path)


def test_env_var_overrides_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEST_DATA_ROOT", str(tmp_path))
    _valid_npz(tmp_path / DATASETS["cb513"]["file"])
    path = ensure_dataset("cb513", allow_download=False)
    assert path.parent == tmp_path


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_split_accepts_an_existing_path(tmp_path):
    path = _valid_npz(tmp_path / "custom.npz")
    assert resolve_split(str(path)) == path


def test_resolve_split_accepts_a_dataset_name(tmp_path):
    _valid_npz(tmp_path / DATASETS["cb513"]["file"])
    assert resolve_split("cb513", root=tmp_path, allow_download=False).exists()


def test_resolve_split_rejects_a_nonexistent_path(tmp_path):
    with pytest.raises(DatasetUnavailable, match="neither a known dataset"):
        resolve_split(str(tmp_path / "nope.npz"))


def test_resolve_split_prefers_a_real_path_over_a_name(tmp_path, monkeypatch):
    """An explicitly supplied file must never be replaced by a download.

    ``np.savez`` appends ``.npz``, so the collision file is written then renamed.
    """
    monkeypatch.chdir(tmp_path)
    written = _valid_npz(tmp_path / "collide.npz")
    local = tmp_path / "cb513"
    written.rename(local)
    resolved = resolve_split("cb513", root=tmp_path, allow_download=False)
    # Compare resolved paths: the lookup returns the spec as given (relative to
    # the chdir'd cwd), which is the same file as the absolute one.
    assert resolved.resolve() == local.resolve()


# ---------------------------------------------------------------------------
# CB513 CSV
# ---------------------------------------------------------------------------

def test_cb513_csv_helper_returns_none_when_absent(tmp_path):
    assert ensure_cb513_csv(tmp_path / "missing.csv") is None


def test_cb513_csv_helper_finds_the_repo_copy():
    repo_copy = Path("data/cb513/CB513.csv")
    if repo_copy.exists():
        assert ensure_cb513_csv() == repo_copy


# ---------------------------------------------------------------------------
# Integration with the loader
# ---------------------------------------------------------------------------

def test_load_netsurfp_resolves_a_dataset_name(tmp_path):
    """``load_netsurfp('cb513')`` must go through the downloader rather than
    treating the name as a path."""
    from zest.data.netsurfp import load_netsurfp

    with pytest.raises(DatasetUnavailable):
        load_netsurfp("cb513", data_root=tmp_path, allow_download=False)


def test_load_netsurfp_still_accepts_a_plain_path():
    from zest.data.netsurfp import load_netsurfp

    repo_copy = Path("data/cb513/CB513.csv")
    if not repo_copy.exists():
        pytest.skip("CB513 csv missing")
    assert len(load_netsurfp(str(repo_copy))) == 511
