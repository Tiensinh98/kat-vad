"""KAT-VAD v2 decision splits: pure helpers and the sealed-split loader.

The v2 proposal (``core/docs/v2/KAT_VAD_PROPOSAL_v2.md`` §7.2) decides every
check on **T2-val** and **DoTA-dev** and opens **DoTA-eval** once, for the final
report. The split files are written once by ``core.tools.freeze_splits`` and
committed; this module holds what both the writer and every reader need:

* :func:`share_bin` -- the accident-share bins the CRN rule is read per.
* :func:`dota_group` -- DoTA clips cut from the same YouTube video are
  correlated (1,397 clips come from 179 videos, up to 19 each), so the split and
  every bootstrap over it group by source video, never by clip.
* :func:`grouped_split` -- a seeded, stratified, group-disjoint split that aims
  at a clip fraction, not a group fraction.
* :func:`load_split` -- the only reader. It checks the file against the
  manifest's sha1 and **refuses a sealed split** (DoTA-eval) unless the caller
  passes ``final=True``, so no harness can print a DoTA-eval number by accident.

Derived split sets (DoTA-CAP, addendum §11 L7) are frozen later than the base
splits, by their own tool, into their own manifest in the same directory
(``V2_DERIVED_MANIFEST_FILENAMES``). :func:`frozen_splits` merges them, so the
base manifest never changes and ``freeze_splits --check`` keeps passing.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from core import constants

LOGGER = logging.getLogger(__name__)

# DoTA clip ids are "{youtube_id}_{start_frame:06d}"; the YouTube id may itself
# contain underscores, so split on the LAST one.
_DOTA_ID_SEPARATOR = "_"


class SealedSplitError(RuntimeError):
    """A sealed split (DoTA-eval) was requested before the final report."""


def share_bin(share: float) -> str:
    """Label of the accident-share bin ``share`` (a fraction in [0, 1]) falls in.

    Bins are half-open on the right edge: 0.30 is in ``30-50``, not ``<30``.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"accident share must be in [0, 1], got {share}")
    index = bisect.bisect_right(constants.V2_SHARE_BIN_EDGES, share)
    return constants.V2_SHARE_BIN_LABELS[index]


def dota_group(clip_id: str) -> str:
    """Source YouTube video of a DoTA clip id (``abc_def_000387`` -> ``abc_def``)."""
    group, sep, start = clip_id.rpartition(_DOTA_ID_SEPARATOR)
    if not sep or not group or not start.isdigit():
        raise ValueError(f"not a DoTA clip id (expected '<video>_<frame>'): {clip_id!r}")
    return group


def grouped_split(
    items_by_group: dict[str, list[str]],
    stratum_of_group: dict[str, str],
    fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split whole groups into (selected, rest) so ~``fraction`` of ITEMS are selected.

    Per stratum, groups are shuffled with ``seed`` and taken greedily while taking
    the next one moves the stratum's selected item count closer to
    ``fraction`` of its items. A group is never split, so the two sides share no
    source video.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    missing = sorted(set(items_by_group) - set(stratum_of_group))
    if missing:
        raise ValueError(f"{len(missing)} groups have no stratum: {missing[:3]}")

    groups_by_stratum: dict[str, list[str]] = {}
    for group in sorted(items_by_group):
        groups_by_stratum.setdefault(stratum_of_group[group], []).append(group)

    # deterministic split shuffling, not security-sensitive
    rng = random.Random(f"{seed}:{fraction}")  # nosec B311
    selected_groups: set[str] = set()
    for stratum in sorted(groups_by_stratum):
        groups = groups_by_stratum[stratum]
        rng.shuffle(groups)
        target = fraction * sum(len(items_by_group[g]) for g in groups)
        taken = 0
        for group in groups:
            size = len(items_by_group[group])
            if abs(taken + size - target) < abs(taken - target):
                selected_groups.add(group)
                taken += size

    selected = {i for g in selected_groups for i in items_by_group[g]}
    rest = {i for g in items_by_group if g not in selected_groups for i in items_by_group[g]}
    return selected, rest


def lines_sha1(ids: list[str]) -> str:
    """Identity fingerprint of a split: sha1 of its sorted ids, one per line."""
    payload = "".join(f"{i}\n" for i in sorted(ids)).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def split_path(name: str, split_dir: Path = constants.V2_SPLITS_DIR) -> Path:
    """Path of split ``name``'s id file."""
    return split_dir / f"{name}{constants.V2_SPLIT_FILE_SUFFIX}"


def read_manifest(split_dir: Path = constants.V2_SPLITS_DIR) -> dict[str, Any]:
    """The frozen splits' manifest; raises if the splits were never frozen."""
    path = split_dir / constants.V2_SPLITS_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; freeze the splits first: python -m core.tools.freeze_splits"
        )
    with path.open("r", encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)
    return manifest


def frozen_splits(split_dir: Path = constants.V2_SPLITS_DIR) -> dict[str, dict[str, Any]]:
    """``{name: {count, sha1}}`` over the base manifest and every derived one present.

    A name frozen in two manifests raises: one id file cannot have two fingerprints.
    """
    splits: dict[str, dict[str, Any]] = dict(read_manifest(split_dir)["splits"])
    for filename in constants.V2_DERIVED_MANIFEST_FILENAMES:
        path = split_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            derived: dict[str, dict[str, Any]] = json.load(fh)["splits"]
        clash = sorted(set(splits) & set(derived))
        if clash:
            raise ValueError(f"{path} re-freezes splits {clash} already frozen elsewhere")
        splits |= derived
    return splits


def load_split(
    name: str,
    split_dir: Path = constants.V2_SPLITS_DIR,
    final: bool = False,
) -> list[str]:
    """Ids of a frozen v2 split, sorted; verified against the manifest's sha1.

    ``final=True`` is required for a sealed split (DoTA-eval, DoTA-CAP-eval). Pass
    it only from the final-report step, once, after E3 has been decided on DoTA-dev.
    """
    if name in constants.V2_SEALED_SPLITS and not final:
        raise SealedSplitError(
            f"split {name!r} is sealed until the final report; pass final=True only there"
        )
    splits = frozen_splits(split_dir)
    if name not in splits:
        raise KeyError(f"unknown v2 split {name!r}; frozen splits: {sorted(splits)}")
    ids = [
        line.strip()
        for line in split_path(name, split_dir).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = splits[name]["sha1"]
    if lines_sha1(ids) != expected:
        raise ValueError(
            f"split {name!r} does not match its frozen sha1 {expected}; "
            "the file was edited after freezing"
        )
    return sorted(ids)
