"""Per-video captions for ``loss.cap_contrast_mode=caption`` (``KIM_RUN.md`` §30 C3).

Reads the ``captions.json`` manifest ``core.tools.nexar_caption`` writes and refuses one that is
not bound to the current ``nexar_train`` split. A manifest naming ``train_split=dota_train``
(§65, ``core.tools.dota_label_captions``) is bound to the §55 DoTA train clips instead.
Captions are kept verbatim (type prefix included, §30 C2). §29.3's QC gate was waived by the
user (2026-11-17); the manifest is still checked for its split binding and completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

from core import constants
from core.data.v2_splits import lines_sha1, load_split

MANIFEST_CAPTIONS_KEY = "captions"
MANIFEST_SHA_KEY = "train_ids_sha1"


def dota_train_ids() -> list[str]:
    """The §55 DoTA train clips: the frozen §52 map minus ``DOTA_TRAIN_EXCLUDED``, sorted."""
    path = constants.DOTA_TRAIN_SPLITS_DIR / constants.DOTA_TRAIN_MAP_FILENAME
    clips = json.loads(path.read_text(encoding="utf-8"))["clips"]
    return sorted(set(clips) - set(constants.DOTA_TRAIN_EXCLUDED))


def split_ids(split: str) -> list[str]:
    """Sorted ids of a caption split (``constants.CAPTION_SPLITS``); another name raises."""
    if split == constants.CAPTION_SPLIT_NEXAR:
        return sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN))
    if split == constants.CAPTION_SPLIT_DOTA:
        return dota_train_ids()
    raise ValueError(f"caption split must be one of {constants.CAPTION_SPLITS}, got {split!r}")


def load_caption_manifest(path: Path, dataset: str | None = None) -> dict[str, str]:
    """``{source video id: caption}``; a manifest from another split or with blanks raises.

    ``dataset`` (``data.dataset``) must match the manifest's split when given: DoTA train for
    ``dota_train``, anything else for ``nexar_train``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"caption manifest {path} does not exist (KIM_RUN.md §30 C3)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    captions = manifest.get(MANIFEST_CAPTIONS_KEY)
    if not isinstance(captions, dict) or not captions:
        raise ValueError(f"{path}: no '{MANIFEST_CAPTIONS_KEY}' mapping")
    split = manifest.get(constants.CAPTION_MANIFEST_SPLIT_KEY, constants.CAPTION_SPLIT_NEXAR)
    want = lines_sha1(split_ids(split))
    if manifest.get(MANIFEST_SHA_KEY) != want:
        raise ValueError(f"{path}: bound to train split {manifest.get(MANIFEST_SHA_KEY)!r}, "
                         f"the current {split} is {want!r}")
    if dataset is not None and (dataset == constants.DOTA_TRAIN_DATASET) != (
            split == constants.CAPTION_SPLIT_DOTA):
        raise ValueError(f"{path}: captions of {split} cannot train data.dataset={dataset}")
    blank = sorted(v for v, c in captions.items() if not str(c).strip())
    if blank:
        raise ValueError(f"{path}: {len(blank)} empty captions: {blank[:5]}")
    return {str(v): str(c) for v, c in captions.items()}
