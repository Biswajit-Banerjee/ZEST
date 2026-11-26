"""NetSurfP-2.0 residue-level dataset loader.

Why this file exists: the earlier secondary-structure numbers came from a CSV of
uncertain provenance with no training split, so nothing could be compared
against published baselines. NetSurfP-2.0 is the primary source that the SS3/SS8
literature reports on, it ships a homology-partitioned training set, and it
carries five more residue-level targets from the same DSSP run (SS8, RSA, ASA,
phi/psi, disorder). Those extra targets cost nothing to load and are what turn
"can ZEST do secondary structure" into "can ZEST recover residue-level geometry".

Two input formats are supported:

* ``.npz`` -- the official release, a dense ``[N, L, 68]`` tensor. This is the
  only format with RSA/ASA/phi/psi/profile and the only one with a train split.
* ``.csv`` -- the reduced CB513 table already in ``data/cb513/CB513.csv``
  (sequence, dssp3, dssp8, disorder, cb513_mask). Test-set evaluation only.

The ``[N, L, 68]`` column layout is *asserted, not assumed*. A silently wrong
slice offset would yield plausible-looking arrays and quietly invalid accuracy,
so :func:`_verify_layout` checks one-hot validity, value ranges and the Q8->Q3
consistency relation before any data is returned.

The Q8->Q3 table below was derived empirically from all 511 CB513 chains
(144,011 residues) and every class maps with purity 1.0000. ``B`` (isolated
beta-bridge) maps to ``E``; sources disagree on this one, hence the check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_log = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Canonical label orders (also the one-hot channel order used by the release).
SS3_ALPHABET = "HEC"
SS8_ALPHABET = "GHIBESTC"

SS3_TO_IDX = {c: i for i, c in enumerate(SS3_ALPHABET)}
SS8_TO_IDX = {c: i for i, c in enumerate(SS8_ALPHABET)}

# Verified against CB513: purity 1.0000 for all eight classes.
Q8_TO_Q3: Dict[str, str] = {
    "G": "H", "H": "H", "I": "H",
    "B": "E", "E": "E",
    "S": "C", "T": "C", "C": "C",
}
# Same relation as channel indices, for validating the dense one-hot arrays.
Q8_IDX_TO_Q3_IDX = np.array(
    [SS3_TO_IDX[Q8_TO_Q3[c]] for c in SS8_ALPHABET], dtype=np.int64)

# Column layout of the official [N, L, 68] arrays, transcribed from
# https://services.healthtech.dtu.dk/services/NetSurfP-2.0/5-Dataset.php
#
#   [0:20]  amino-acid one-hot; UNKNOWN residues are an all-zero vector
#   [20:50] HMM profile
#   [50]    sequence mask    1 = residue, 0 = padding
#   [51]    ordered mask     1 = ordered, 0 = DISORDERED  (note the polarity)
#   [52]    evaluation mask  1 = score this residue (the CB513 protocol)
#   [53]    ASA isolated     [55] RSA isolated
#   [54]    ASA complexed    [56] RSA complexed
#   [57:65] Q8 one-hot, channel order GHIBESTC
#   [65:67] phi, psi
#   [67]    ASA_max, the per-residue reference area
#
# There is no SS3 block in the file. Q3 is *defined* by GHIBESTC -> HHHEECCC,
# so SS3 is derived rather than read, and the derivation is the same table the
# CB513 CSV is checked against.
COL_ONEHOT = slice(0, 20)
COL_PROFILE = slice(20, 50)
COL_SEQ_MASK = 50
COL_ORDERED = 51
COL_EVAL_MASK = 52
COL_ASA = 53
COL_ASA_COMPLEXED = 54
COL_RSA = 55
COL_RSA_COMPLEXED = 56
COL_SS8 = slice(57, 65)
COL_PHI = 65
COL_PSI = 66
COL_ASA_MAX = 67
N_FEATURES = 68

# Channel order of the 20-way residue one-hot in the release.
NSP2_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class ResidueTargets:
    """Residue-level targets, padded to the longest chain in the split.

    Classification targets use ``-100`` at invalid positions so they can go
    straight into ``cross_entropy``; regression targets use ``NaN``, which the
    loss must mask explicitly.
    """

    sequences: List[str]
    lengths: np.ndarray          # [N] int32
    seq_mask: np.ndarray         # [N, L] bool - real residue
    eval_mask: np.ndarray        # [N, L] bool - scored residue (CB513 protocol)
    ss3: np.ndarray              # [N, L] int64, -100 invalid
    ss8: Optional[np.ndarray]    # [N, L] int64, -100 invalid
    disorder: Optional[np.ndarray]   # [N, L] int64, -100 invalid
    rsa: Optional[np.ndarray] = None      # [N, L] float32, NaN invalid
    asa: Optional[np.ndarray] = None      # [N, L] float32
    phi: Optional[np.ndarray] = None      # [N, L] float32, degrees
    psi: Optional[np.ndarray] = None      # [N, L] float32, degrees
    profile: Optional[np.ndarray] = None  # [N, L, 30] float32
    source: str = ""

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def n_scored(self) -> int:
        return int(self.eval_mask.sum())

    def summary(self) -> str:
        parts = [f"{len(self)} chains", f"{int(self.seq_mask.sum())} residues",
                 f"{self.n_scored} scored"]
        available = [name for name in
                     ("ss8", "disorder", "rsa", "asa", "phi", "psi", "profile")
                     if getattr(self, name) is not None]
        parts.append("targets: ss3+" + "+".join(available) if available else "targets: ss3")
        return f"{self.source}: " + ", ".join(parts)

    def select(self, indices: Sequence[int]) -> "ResidueTargets":
        """Return the subset of chains at *indices*, trimmed to their longest."""
        indices = np.asarray(list(indices), dtype=np.int64)
        width = int(self.lengths[indices].max()) if indices.size else 0

        def _sub(array):
            if array is None:
                return None
            return array[indices, :width] if array.ndim >= 2 else array[indices]

        return ResidueTargets(
            sequences=[self.sequences[i] for i in indices],
            lengths=self.lengths[indices],
            seq_mask=_sub(self.seq_mask), eval_mask=_sub(self.eval_mask),
            ss3=_sub(self.ss3), ss8=_sub(self.ss8), disorder=_sub(self.disorder),
            rsa=_sub(self.rsa), asa=_sub(self.asa),
            phi=_sub(self.phi), psi=_sub(self.psi), profile=_sub(self.profile),
            source=self.source,
        )

    def duplicate_report(self) -> Dict[str, object]:
        """Count chains whose sequence appears more than once.

        CB513 as distributed contains 79 redundant rows (60 sequences at
        multiplicity 2-4) carrying 26% of all residues. Per-residue accuracy
        computed over the raw file therefore weights those proteins several
        times. Published numbers use the file as-is, so this is reported rather
        than silently corrected -- see :meth:`deduplicate`.
        """
        first: Dict[str, int] = {}
        redundant: List[int] = []
        for i, seq in enumerate(self.sequences):
            if seq in first:
                redundant.append(i)
            else:
                first[seq] = i
        residues = int(sum(len(self.sequences[i]) for i in redundant))
        total = int(self.seq_mask.sum())
        return {"n_chains": len(self), "n_unique": len(first),
                "n_redundant_rows": len(redundant),
                "redundant_residues": residues,
                "redundant_fraction": (residues / total) if total else 0.0}

    def deduplicate(self) -> "ResidueTargets":
        """Keep the first chain per distinct sequence."""
        seen, keep = set(), []
        for i, seq in enumerate(self.sequences):
            if seq not in seen:
                seen.add(seq)
                keep.append(i)
        return self.select(keep)

    def drop_unscored_chains(self) -> "ResidueTargets":
        """Drop chains with no scored residue.

        Two CB513 chains are fully masked out; keeping them makes any per-chain
        average divide by zero.
        """
        keep = [i for i in range(len(self)) if self.eval_mask[i].any()]
        return self.select(keep)

    def class_balance(self, field: str = "ss3") -> Dict[str, float]:
        """Label frequencies over scored residues.

        Worth printing: SS3 is roughly 35/23/42, so a majority-class predictor
        already reaches ~42% and any accuracy must be read against that floor.
        """
        values = getattr(self, field)
        if values is None:
            return {}
        alphabet = SS3_ALPHABET if field == "ss3" else SS8_ALPHABET
        scored = values[self.eval_mask & (values != IGNORE_INDEX)]
        total = scored.size
        if total == 0:
            return {}
        return {alphabet[i]: float((scored == i).sum()) / total
                for i in range(len(alphabet))}


# ---------------------------------------------------------------------------
# Layout verification
# ---------------------------------------------------------------------------

def _verify_layout(data: np.ndarray, strict: bool = True) -> List[str]:
    """Assert the ``[N, L, 68]`` column layout actually holds.

    Every check keys off a property that only holds when the slices are right,
    so an offset layout cannot pass. Returns the list of confirmations.
    """
    problems: List[str] = []
    notes: List[str] = []

    if data.ndim != 3 or data.shape[2] != N_FEATURES:
        raise ValueError(
            f"expected [N, L, {N_FEATURES}] array, got {list(data.shape)}. "
            "This loader targets the official NetSurfP-2.0 release format.")

    seq_mask = data[:, :, COL_SEQ_MASK] > 0.5
    n_res = int(seq_mask.sum())
    if n_res == 0:
        raise ValueError(f"column {COL_SEQ_MASK} is all zero; it is not the sequence mask")
    notes.append(f"seq mask: {n_res} residues over {seq_mask.shape[0]} chains")

    # Residues must be a left-packed prefix of each row.
    prefix = np.cumsum(seq_mask, axis=1) == np.cumsum(np.ones_like(seq_mask), axis=1)
    if not np.array_equal(seq_mask, seq_mask & prefix):
        problems.append("sequence mask is not a left-aligned prefix per chain")

    aa = data[:, :, COL_ONEHOT]
    aa_sums = aa.sum(axis=2)[seq_mask]
    # Unknown residues are stored as an all-zero vector, so 0 is as valid as 1.
    aa_ok = np.isclose(aa_sums, 1.0, atol=1e-3) | np.isclose(aa_sums, 0.0, atol=1e-3)
    if aa_ok.mean() < 0.99:
        problems.append(f"residue one-hot rows sum to neither 1 nor 0 at "
                        f"{1-aa_ok.mean():.1%} of residues; cols {COL_ONEHOT} are "
                        f"not the amino-acid one-hot")
    else:
        unknown = float(np.mean(np.isclose(aa_sums, 0.0, atol=1e-3)))
        notes.append(f"residue one-hot valid ({unknown:.2%} unknown residues)")

    ss8 = data[:, :, COL_SS8]
    ss8_sums = ss8.sum(axis=2)[seq_mask]
    ss8_ok = np.isclose(ss8_sums, 1.0, atol=1e-3)
    if ss8_ok.mean() < 0.99:
        problems.append(f"SS8 one-hot invalid at {1-ss8_ok.mean():.1%} of residues; "
                        f"cols {COL_SS8} are not the Q8 block")
    else:
        notes.append(f"Q8 one-hot valid at {ss8_ok.mean():.4%} of residues "
                     f"(channel order '{SS8_ALPHABET}')")

    rsa = data[:, :, COL_RSA][seq_mask]
    finite_rsa = rsa[np.isfinite(rsa)]
    if finite_rsa.size and (finite_rsa.min() < -0.01 or finite_rsa.max() > 1.5):
        problems.append(f"RSA outside [0, 1.5]: [{finite_rsa.min():.2f}, "
                        f"{finite_rsa.max():.2f}]; col {COL_RSA} is not RSA")
    elif finite_rsa.size:
        notes.append(f"RSA range [{finite_rsa.min():.2f}, {finite_rsa.max():.2f}]")

    # The decisive check. RSA is ASA divided by the residue's reference area, so
    # this single relation pins columns 53, 55 and 67 at once. A shifted layout
    # cannot satisfy it by accident, which is what makes it worth more than the
    # individual range checks above.
    asa = data[:, :, COL_ASA][seq_mask]
    asa_max = data[:, :, COL_ASA_MAX][seq_mask]
    usable = np.isfinite(asa) & np.isfinite(asa_max) & (asa_max > 1e-6)
    if usable.sum() >= 100:
        implied = asa[usable] / asa_max[usable]
        agree = float(np.mean(np.abs(implied - rsa[usable]) < 0.02))
        if agree < 0.95:
            problems.append(
                f"RSA != ASA/ASA_max at {1-agree:.1%} of residues, so columns "
                f"{COL_ASA} (ASA), {COL_RSA} (RSA) and {COL_ASA_MAX} (ASA_max) "
                f"are not where this loader expects them.")
        else:
            notes.append(f"RSA == ASA/ASA_max at {agree:.2%} of residues "
                         f"(cols {COL_ASA}/{COL_RSA}/{COL_ASA_MAX} confirmed)")

    for col, name in ((COL_PHI, "phi"), (COL_PSI, "psi")):
        ang = data[:, :, col][seq_mask]
        ang = ang[np.isfinite(ang)]
        if ang.size and (ang.min() < -360.1 or ang.max() > 360.1):
            problems.append(f"{name} outside [-360, 360]; col {col} is not {name}")
    notes.append("phi/psi within angular range")

    for col, name in ((COL_ORDERED, "ordered mask"), (COL_EVAL_MASK, "eval mask")):
        vals = np.unique(data[:, :, col])
        if not np.all(np.isin(vals, (0.0, 1.0))):
            problems.append(f"{name} (col {col}) is not binary; found {vals[:6]}")

    if problems:
        message = ("NetSurfP-2.0 layout verification failed:\n  - "
                   + "\n  - ".join(problems)
                   + "\nRefusing to return data that would produce invalid metrics.")
        if strict:
            raise ValueError(message)
        _log.warning(message)
    for note in notes:
        _log.info("layout check: %s", note)
    return notes


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _decode_sequences(data: np.ndarray, seq_mask: np.ndarray) -> List[str]:
    """Recover the amino-acid string from the one-hot block.

    Unknown residues are an all-zero vector, which ``argmax`` would silently
    report as alanine, so they are emitted as ``X`` instead.
    """
    block = data[:, :, COL_ONEHOT]
    idx = block.argmax(axis=2)
    known = block.sum(axis=2) > 0.5
    lookup = np.array(list(NSP2_AA_ORDER))
    out = []
    for i in range(len(idx)):
        length = int(seq_mask[i].sum())
        letters = lookup[idx[i, :length]].copy()
        letters[~known[i, :length]] = "X"
        out.append("".join(letters))
    return out


def load_npz(path, key: Optional[str] = None, strict: bool = True,
             with_profile: bool = False, max_length: Optional[int] = None,
             use_eval_mask: bool = True) -> ResidueTargets:
    """Load an official NetSurfP-2.0 ``.npz`` split.

    ``use_eval_mask`` follows the CB513 protocol: only residues flagged in the
    evaluation-mask column are scored. For training splits that column is
    typically all ones, in which case this is a no-op.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        keys = list(handle.keys())
        if key is None:
            candidates = [k for k in keys if handle[k].ndim == 3
                          and handle[k].shape[-1] == N_FEATURES]
            if not candidates:
                raise KeyError(
                    f"{path.name} has no [N, L, {N_FEATURES}] array. Keys: {keys}")
            key = candidates[0]
        data = np.asarray(handle[key], dtype=np.float32)
    _log.info("loaded %s[%s] shape=%s", path.name, key, list(data.shape))

    _verify_layout(data, strict=strict)

    seq_mask = data[:, :, COL_SEQ_MASK] > 0.5
    lengths = seq_mask.sum(axis=1).astype(np.int32)

    keep = np.ones(len(data), dtype=bool)
    if max_length:
        keep = lengths <= max_length
        if not keep.all():
            _log.info("dropping %d/%d chains longer than %d residues",
                      int((~keep).sum()), len(keep), max_length)
        data, seq_mask, lengths = data[keep], seq_mask[keep], lengths[keep]

    # Trim the padding shared by every chain in the split.
    width = int(lengths.max()) if len(lengths) else 0
    data, seq_mask = data[:, :width], seq_mask[:, :width]

    eval_mask = (data[:, :, COL_EVAL_MASK] > 0.5) & seq_mask if use_eval_mask else seq_mask

    def _labels(block: np.ndarray) -> np.ndarray:
        out = np.full(seq_mask.shape, IGNORE_INDEX, dtype=np.int64)
        out[seq_mask] = block.argmax(axis=2)[seq_mask]
        return out

    def _regression(col: int) -> np.ndarray:
        out = np.full(seq_mask.shape, np.nan, dtype=np.float32)
        out[seq_mask] = data[:, :, col][seq_mask]
        return out

    ss8 = _labels(data[:, :, COL_SS8])
    # The file carries no SS3 block: Q3 is defined as the image of Q8 under
    # GHIBESTC -> HHHEECCC, so it is derived here rather than read.
    ss3 = np.full(seq_mask.shape, IGNORE_INDEX, dtype=np.int64)
    ss3[seq_mask] = Q8_IDX_TO_Q3_IDX[ss8[seq_mask]]

    # Column 51 flags ORDERED residues (1 = ordered). The prediction target is
    # disorder, so the positive class is the complement; using the column as-is
    # would invert the labels and make the rare class the majority.
    disorder = np.full(seq_mask.shape, IGNORE_INDEX, dtype=np.int64)
    disorder[seq_mask] = (data[:, :, COL_ORDERED][seq_mask] <= 0.5).astype(np.int64)

    return ResidueTargets(
        sequences=_decode_sequences(data, seq_mask),
        lengths=lengths,
        seq_mask=seq_mask,
        eval_mask=eval_mask,
        ss3=ss3,
        ss8=ss8,
        disorder=disorder,
        rsa=_regression(COL_RSA),
        asa=_regression(COL_ASA),
        phi=_regression(COL_PHI),
        psi=_regression(COL_PSI),
        profile=np.ascontiguousarray(data[:, :, COL_PROFILE]) if with_profile else None,
        source=f"{path.name}[{key}]",
    )


def _parse_float_field(value, length: int) -> np.ndarray:
    """Parse a whitespace-separated float field, tolerating NaN tokens."""
    if isinstance(value, str):
        parts = value.split()
    else:
        parts = list(np.atleast_1d(value))
    out = np.full(length, np.nan, dtype=np.float32)
    for i, token in enumerate(parts[:length]):
        try:
            out[i] = float(token)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def load_cb513_csv(path, strict: bool = True,
                   use_eval_mask: bool = True) -> ResidueTargets:
    """Load the reduced CB513 CSV (sequence, dssp3, dssp8, disorder, cb513_mask).

    Evaluation only: this file has no training split and no RSA/phi/psi.
    """
    import pandas as pd

    path = Path(path)
    frame = pd.read_csv(path)
    required = {"input", "dssp3", "dssp8"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path.name} is missing columns {sorted(missing)}")

    sequences = [str(s) for s in frame["input"]]
    ragged = [(i, len(s), len(str(frame['dssp3'][i])), len(str(frame['dssp8'][i])))
              for i, s in enumerate(sequences)
              if not len(s) == len(str(frame["dssp3"][i])) == len(str(frame["dssp8"][i]))]
    if ragged and strict:
        raise ValueError(
            f"{len(ragged)} chains have sequence/label length mismatch, e.g. {ragged[:3]}. "
            "Labels cannot be aligned to residues.")

    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    width = int(lengths.max())
    n = len(sequences)

    seq_mask = np.zeros((n, width), dtype=bool)
    ss3 = np.full((n, width), IGNORE_INDEX, dtype=np.int64)
    ss8 = np.full((n, width), IGNORE_INDEX, dtype=np.int64)
    disorder = np.full((n, width), IGNORE_INDEX, dtype=np.int64)
    eval_mask = np.zeros((n, width), dtype=bool)

    unknown3, unknown8, inconsistent = set(), set(), 0
    for i, seq in enumerate(sequences):
        length = len(seq)
        seq_mask[i, :length] = True
        d3, d8 = str(frame["dssp3"][i]), str(frame["dssp8"][i])
        for j in range(length):
            c3, c8 = d3[j], d8[j]
            if c3 in SS3_TO_IDX:
                ss3[i, j] = SS3_TO_IDX[c3]
            else:
                unknown3.add(c3)
            if c8 in SS8_TO_IDX:
                ss8[i, j] = SS8_TO_IDX[c8]
            else:
                unknown8.add(c8)
            # Independent confirmation of the mapping on every residue.
            if c8 in Q8_TO_Q3 and c3 in SS3_TO_IDX and Q8_TO_Q3[c8] != c3:
                inconsistent += 1

        if "disorder" in frame.columns:
            # Same polarity as the npz ordered mask: 1 means ordered, and ~97.6%
            # of CB513 residues are 1. The target is disorder, so invert.
            values = _parse_float_field(frame["disorder"][i], length)
            valid = np.isfinite(values)
            column = disorder[i, :length]
            column[valid] = (values[valid] <= 0.5).astype(np.int64)
            disorder[i, :length] = column

        if use_eval_mask and "cb513_mask" in frame.columns:
            values = _parse_float_field(frame["cb513_mask"][i], length)
            eval_mask[i, :length] = np.nan_to_num(values) > 0.5
        else:
            eval_mask[i, :length] = True

    if unknown3 or unknown8:
        _log.warning("unmapped SS symbols dssp3=%s dssp8=%s (left as ignore)",
                     sorted(unknown3), sorted(unknown8))
    if inconsistent:
        message = (f"{inconsistent} residues violate the Q8->Q3 table "
                   f"{Q8_TO_Q3}; the label conventions in {path.name} differ.")
        if strict:
            raise ValueError(message)
        _log.warning(message)

    ss3[~seq_mask] = IGNORE_INDEX
    eval_mask &= seq_mask & (ss3 != IGNORE_INDEX)

    return ResidueTargets(
        sequences=sequences, lengths=lengths, seq_mask=seq_mask,
        eval_mask=eval_mask, ss3=ss3, ss8=ss8, disorder=disorder,
        source=path.name,
    )


def load_netsurfp(path, allow_download: bool = True, data_root=None,
                  **kwargs) -> ResidueTargets:
    """Load a split, fetching it first if it is not on disk.

    ``path`` may be a filesystem path or a dataset name such as ``"train"`` or
    ``"cb513"``. Names are resolved through :mod:`zest.data.download`, so a job
    never stops to ask for a missing file.
    """
    from zest.data.download import DATASETS, resolve_split

    if not Path(path).exists() and str(path) in DATASETS:
        path = resolve_split(str(path), root=data_root,
                             allow_download=allow_download)

    path = Path(path)
    if path.suffix == ".npz":
        return load_npz(path, **kwargs)
    if path.suffix in (".csv", ".tsv"):
        kwargs.pop("with_profile", None)
        kwargs.pop("max_length", None)
        kwargs.pop("key", None)
        return load_cb513_csv(path, **kwargs)
    raise ValueError(f"unsupported extension {path.suffix!r}; expected .npz or .csv")


def check_no_leakage(train: Sequence[str], test: Sequence[str]) -> Dict[str, object]:
    """Exact-duplicate check between splits.

    NetSurfP-2.0 is already homology-partitioned at 25% identity, so this only
    catches gross contamination such as accidentally evaluating on the training
    file. It is cheap and it makes the claim checkable.
    """
    train_set = set(train)
    overlap = sorted({s for s in test if s in train_set})
    return {"n_train": len(train_set), "n_test": len(set(test)),
            "n_exact_duplicates": len(overlap),
            "examples": [s[:40] for s in overlap[:3]]}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audit a NetSurfP-2.0 split")
    parser.add_argument("path")
    parser.add_argument("--train", default=None, help="Train split for a leakage check")
    parser.add_argument("--no-eval-mask", action="store_true")
    parser.add_argument("--lenient", action="store_true",
                        help="Warn instead of raising on verification failure")
    parser.add_argument("--dedup", action="store_true",
                        help="Keep one chain per distinct sequence")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset = load_netsurfp(args.path, strict=not args.lenient,
                            use_eval_mask=not args.no_eval_mask)
    duplicates = dataset.duplicate_report()
    if args.dedup:
        dataset = dataset.deduplicate()
    print()
    print(dataset.summary())
    print(f"  lengths: min={dataset.lengths.min()} median="
          f"{int(np.median(dataset.lengths))} max={dataset.lengths.max()}")
    print(f"  redundancy: {duplicates['n_redundant_rows']} repeated rows, "
          f"{duplicates['n_unique']} distinct sequences, "
          f"{duplicates['redundant_fraction']:.1%} of residues are repeats"
          + ("  [removed]" if args.dedup else "  [kept: matches published protocol]"))
    unscored = len(dataset) - len(dataset.drop_unscored_chains())
    if unscored:
        print(f"  warning: {unscored} chain(s) have no scored residue")
    balance = dataset.class_balance("ss3")
    print("  SS3 balance: " + " ".join(f"{k}={v:.3f}" for k, v in balance.items()))
    if balance:
        print(f"  majority-class floor: {max(balance.values()):.4f}")
    if dataset.ss8 is not None:
        print("  SS8 balance: " + " ".join(
            f"{k}={v:.3f}" for k, v in dataset.class_balance("ss8").items()))
    if args.train:
        other = load_netsurfp(args.train, strict=not args.lenient)
        print("  leakage: " + str(check_no_leakage(other.sequences, dataset.sequences)))


if __name__ == "__main__":
    main()
