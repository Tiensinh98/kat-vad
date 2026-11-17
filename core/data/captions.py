"""Per-video captions for ``loss.cap_contrast_mode=caption`` (``KIM_RUN.md`` §30 C3).

Reads the ``captions.json`` manifest ``core.tools.nexar_caption`` writes and refuses one that is
not bound to the current ``nexar_train`` split. Captions are kept verbatim (type prefix
included, §30 C2). §29.3's QC gate was waived by the user (2026-11-17); the manifest is still
checked for its split binding and completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

from core import constants
from core.data.v2_splits import lines_sha1, load_split

MANIFEST_CAPTIONS_KEY = "captions"
MANIFEST_SHA_KEY = "train_ids_sha1"


def load_caption_manifest(path: Path) -> dict[str, str]:
    """``{source video id: caption}``; a manifest from another split or with blanks raises."""
    if not path.is_file():
        raise FileNotFoundError(f"caption manifest {path} does not exist (KIM_RUN.md §30 C3)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    captions = manifest.get(MANIFEST_CAPTIONS_KEY)
    if not isinstance(captions, dict) or not captions:
        raise ValueError(f"{path}: no '{MANIFEST_CAPTIONS_KEY}' mapping")
    want = lines_sha1(sorted(load_split(constants.V2_SPLIT_NEXAR_TRAIN)))
    if manifest.get(MANIFEST_SHA_KEY) != want:
        raise ValueError(f"{path}: bound to train split {manifest.get(MANIFEST_SHA_KEY)!r}, "
                         f"the current nexar_train is {want!r}")
    blank = sorted(v for v, c in captions.items() if not str(c).strip())
    if blank:
        raise ValueError(f"{path}: {len(blank)} empty captions: {blank[:5]}")
    return {str(v): str(c) for v, c in captions.items()}
