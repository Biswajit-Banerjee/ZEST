"""Automatic dataset fetching for the residue benchmarks.

Nothing here needs manual intervention: callers ask for a split by name and get
back a path, downloading only if the file is absent or unusable. That matters
on the cluster, where a job that stops to ask for a file just wastes the
reservation.

Two properties are deliberate:

* **Atomic.** Downloads land in a ``.part`` file and are renamed only after the
  content validates. A job killed by the wall-time signal can therefore never
  leave a truncated ``.npz`` that looks present and fails later at load time.
* **Validated, not just present.** An existing file is opened and checked for
  the expected ``[N, L, 68]`` array before being accepted. A half-written or
  HTML-error-page download is re-fetched rather than trusted.

Source: NetSurfP-3.0, https://services.healthtech.dtu.dk/services/NetSurfP-3.0/
The 3.0 release ships the same 68-column format as 2.0; only the URLs differ.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

_log = logging.getLogger(__name__)

NSP3_BASE = "https://services.healthtech.dtu.dk/services/NetSurfP-3.0/training_data"
NSP2_BASE = "https://services.healthtech.dtu.dk/services/NetSurfP-2.0/training_data"

# Default to the HHblits profiles: they are the variant the published
# NetSurfP accuracies are reported on.
DATASETS: Dict[str, Dict[str, str]] = {
    "train": {"file": "Train_HHblits.npz", "desc": "NetSurfP-3.0 training split"},
    "cb513": {"file": "CB513_HHblits.npz", "desc": "CB513 test split"},
    "ts115": {"file": "TS115_HHblits.npz", "desc": "TS115 test split"},
    "casp12": {"file": "CASP12_HHblits.npz", "desc": "CASP12 test split"},
    "train_mmseqs": {"file": "Train_MMseqs.npz", "desc": "NetSurfP-3.0 train (MMseqs)"},
    "cb513_mmseqs": {"file": "CB513_MMseqs.npz", "desc": "CB513 test (MMseqs)"},
}

DEFAULT_ROOT = Path("data/netsurfp3")
N_FEATURES = 68
# Anything smaller than this is an error page, not a dataset.
MIN_PLAUSIBLE_BYTES = 100_000


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset could not be fetched and is not already present."""


def _is_valid_npz(path: Path) -> bool:
    """True if *path* is a readable npz holding an ``[N, L, 68]`` array."""
    try:
        if path.stat().st_size < MIN_PLAUSIBLE_BYTES:
            return False
        import numpy as np

        with np.load(path, allow_pickle=False) as handle:
            return any(handle[k].ndim == 3 and handle[k].shape[-1] == N_FEATURES
                       for k in handle.keys())
    except Exception:       # noqa: BLE001 - any failure means "refetch it"
        return False


def _download(url: str, target: Path, timeout: int = 120) -> None:
    """Stream *url* to *target* atomically, with progress logging."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()

    _log.info("downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "zest-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        step = max(total // 10, 20_000_000) if total else 20_000_000
        next_report = step
        with open(partial, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    if total:
                        _log.info("  %.0f%% (%.0f MB)", 100 * done / total, done / 1e6)
                    else:
                        _log.info("  %.0f MB", done / 1e6)
                    next_report += step

    if not _is_valid_npz(partial):
        size = partial.stat().st_size if partial.exists() else 0
        partial.unlink(missing_ok=True)
        raise DatasetUnavailable(
            f"{url} did not yield a valid NetSurfP npz ({size} bytes). "
            "The server may be down or the URL may have moved.")

    partial.replace(target)
    _log.info("saved %s (%.0f MB)", target, target.stat().st_size / 1e6)


def ensure_dataset(name: str, root: Optional[Path] = None, allow_download: bool = True,
                   fallback_v2: bool = True) -> Path:
    """Return a local path to dataset *name*, downloading it if needed.

    Falls back to the NetSurfP-2.0 URL if the 3.0 one is unreachable; the two
    releases share a format, so a 2.0 file still loads correctly. The fallback
    is logged loudly because it changes which dataset the numbers describe.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")

    root = Path(root or os.environ.get("ZEST_DATA_ROOT") or DEFAULT_ROOT)
    filename = DATASETS[name]["file"]
    target = root / filename

    if target.exists():
        if _is_valid_npz(target):
            _log.info("using cached %s", target)
            return target
        _log.warning("%s exists but is not a valid npz; refetching", target)
        target.unlink()

    if not allow_download:
        raise DatasetUnavailable(
            f"{target} is missing and downloading is disabled. "
            f"Fetch it manually from {NSP3_BASE}/{filename}")

    errors = []
    sources = [f"{NSP3_BASE}/{filename}"]
    if fallback_v2:
        sources.append(f"{NSP2_BASE}/{filename}")

    for i, url in enumerate(sources):
        try:
            if i > 0:
                _log.warning("NetSurfP-3.0 unreachable; falling back to 2.0 for %s",
                             filename)
            _download(url, target)
            return target
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                DatasetUnavailable) as exc:
            errors.append(f"{url}: {exc}")

    raise DatasetUnavailable(
        f"could not obtain {filename}. Tried:\n  " + "\n  ".join(errors)
        + f"\nDownload it manually to {target} and rerun.")


def ensure_cb513_csv(path: Optional[Path] = None) -> Optional[Path]:
    """Return the reduced CB513 CSV if it is in the repo, else ``None``.

    This file is checked in; it has no upstream URL, so a caller that needs it
    must handle its absence rather than expect a download.
    """
    candidate = Path(path or "data/cb513/CB513.csv")
    return candidate if candidate.exists() else None


def resolve_split(spec: str, root: Optional[Path] = None,
                  allow_download: bool = True) -> Path:
    """Resolve a CLI argument that may be a dataset name or a filesystem path.

    Lets ``--test-data cb513`` and ``--test-data /path/to/CB513_HHblits.npz``
    both work, so scripts need no special-casing.

    An existing path always wins over a registry name. That ordering matches
    :func:`zest.data.netsurfp.load_netsurfp` and means an explicitly supplied
    file is never silently replaced by a download.
    """
    path = Path(spec)
    if path.exists():
        return path
    if spec in DATASETS:
        return ensure_dataset(spec, root=root, allow_download=allow_download)
    raise DatasetUnavailable(
        f"{spec!r} is neither a known dataset name ({sorted(DATASETS)}) "
        f"nor an existing path.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-fetch NetSurfP-3.0 splits so later jobs never block")
    parser.add_argument("names", nargs="*", default=["train", "cb513"],
                        help=f"datasets to fetch (default: train cb513). "
                             f"Options: {sorted(DATASETS)}")
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-download", action="store_true",
                        help="Only report what is missing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    failures = []
    for name in (args.names or ["train", "cb513"]):
        try:
            path = ensure_dataset(name, root=args.root,
                                  allow_download=not args.no_download)
            size = path.stat().st_size / 1e6
            print(f"  [OK]   {name:14s} {path}  ({size:.0f} MB)")
        except Exception as exc:        # noqa: BLE001 - report all, fail at the end
            failures.append(name)
            print(f"  [FAIL] {name:14s} {exc}")

    csv = ensure_cb513_csv()
    print(f"  [{'OK' if csv else '--'}]   cb513_csv      {csv or 'not in repo'}")

    if failures:
        raise SystemExit(f"{len(failures)} dataset(s) unavailable: {failures}")


if __name__ == "__main__":
    main()
