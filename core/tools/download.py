"""Download CLI for external artifacts (Phase-6 prerequisites).

Subcommands (datasets themselves are user-provided — MSAD is already on disk):

    ckpt   LaGoVAD ``best.ckpt`` from Google Drive (gdown, resumable)
    clip   HF CLIP ViT-B/16 snapshot prefetch (pinned revision)
    raft   torchvision RAFT-Large weights prefetch
    all    everything above

Each step is skip-if-exists; ``--dry-run`` only logs what would happen. An
optional ``--ckpt-sha256`` verifies the checkpoint after download. Transfer
functions are module-level so tests monkeypatch them (no network in CI).

CLI::

    python -m core.tools.download all [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import zipfile
from pathlib import Path

from core import constants

LOGGER = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20
_CKPT_SUFFIXES = (".ckpt", ".pt", ".pth")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gdown_fetch(file_id: str, output: Path) -> None:
    # gdown's __init__ re-export is not typed as public; use the submodule
    from gdown.download import download as gdown_download

    gdown_download(id=file_id, output=str(output), resume=True, quiet=False)


def _unwrap_zip_wrapper(target: Path) -> None:
    """Replace a zip-wrapped checkpoint download with the inner file, in place.

    The LaGoVAD Drive release is ``best.ckpt`` inside a plain zip; loading the
    wrapper fails with ``file in archive is not in a subdirectory``. A torch
    checkpoint is itself a zip, but its entries live under a subdirectory
    (``*/data.pkl``) — a root-level ``*.ckpt``/``*.pt`` entry means wrapper.
    """
    if not zipfile.is_zipfile(target):
        return
    with zipfile.ZipFile(target) as zf:
        inner = [n for n in zf.namelist() if n.lower().endswith(_CKPT_SUFFIXES)]
        if not inner:
            return  # a torch archive, not a wrapper
        if len(inner) > 1:
            raise RuntimeError(f"{target} wraps multiple checkpoints: {inner}")
        LOGGER.info("ckpt: unwrapping %s from zip download", inner[0])
        tmp = target.with_name(target.name + ".unwrapped")
        with zf.open(inner[0]) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    tmp.replace(target)


def _hf_snapshot_fetch(repo_id: str, revision: str) -> None:
    from huggingface_hub import snapshot_download

    # .bin/.msgpack are redundant (we load model.safetensors) and ~600 MB each.
    snapshot_download(
        repo_id=repo_id, revision=revision, ignore_patterns=["*.bin", "*.msgpack"]
    )


def _raft_weights_fetch(weights_name: str) -> None:
    from torchvision.models.optical_flow import Raft_Large_Weights

    Raft_Large_Weights[weights_name].get_state_dict(progress=True)


def download_ckpt(dry_run: bool = False, sha256: str | None = None, force: bool = False) -> Path:
    """LaGoVAD reference checkpoint -> ``ckpts/lagovad_best.ckpt`` (gate a)."""
    target = constants.CKPT_ROOT / constants.LAGOVAD_BEST_CKPT_FILENAME
    if target.exists() and not force:
        LOGGER.info("ckpt: %s already exists, skipping", target)
        return target
    LOGGER.info(
        "ckpt: GDrive id %s -> %s", constants.LAGOVAD_BEST_CKPT_GDRIVE_ID, target
    )
    if dry_run:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _gdown_fetch(constants.LAGOVAD_BEST_CKPT_GDRIVE_ID, target)
    if not target.exists():
        raise RuntimeError(f"gdown reported success but {target} is missing")
    _unwrap_zip_wrapper(target)
    if sha256 is not None:
        actual = sha256_of(target)
        if actual != sha256.lower():
            raise RuntimeError(
                f"Checksum mismatch for {target}: expected {sha256}, got {actual}"
            )
        LOGGER.info("ckpt: sha256 verified")
    return target


def download_clip(dry_run: bool = False) -> None:
    """Prefetch the pinned HF CLIP snapshot into the local HF cache."""
    LOGGER.info(
        "clip: snapshot %s @ %s", constants.CLIP_MODEL_NAME, constants.CLIP_MODEL_REVISION
    )
    if dry_run:
        return
    _hf_snapshot_fetch(constants.CLIP_MODEL_NAME, constants.CLIP_MODEL_REVISION)


def download_raft(dry_run: bool = False) -> None:
    """Prefetch torchvision RAFT-Large weights into the torch hub cache."""
    LOGGER.info("raft: Raft_Large_Weights.%s", constants.RAFT_WEIGHTS_NAME)
    if dry_run:
        return
    _raft_weights_fetch(constants.RAFT_WEIGHTS_NAME)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "target", choices=("ckpt", "clip", "raft", "all"), help="what to download"
    )
    parser.add_argument("--dry-run", action="store_true", help="log actions only")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--ckpt-sha256", default=None, help="verify best.ckpt checksum")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.target in ("ckpt", "all"):
        download_ckpt(dry_run=args.dry_run, sha256=args.ckpt_sha256, force=args.force)
    if args.target in ("clip", "all"):
        download_clip(dry_run=args.dry_run)
    if args.target in ("raft", "all"):
        download_raft(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
