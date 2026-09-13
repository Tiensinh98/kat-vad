"""DVS KNN filler cache builder (A4) — our own; the baseline ships none.

For every *abnormal* training video, find the ``K`` most similar *normal*
training videos by cosine similarity (faiss inner product on L2-normalized
keys). Key = central-frame CLIP feature; with ``motion_key=True`` a coarse
motion descriptor (mean magnitude mean/std + mean direction cos/sin, from the
cached raw flow statistics) is L2-normalized and appended (spec §6.1 —
ego-centric datasets only; leave off for fixed-camera MSAD).

Saved as ``cache/knn/{DATASET}/knn_cache.npz`` with arrays ``anomaly_ids``,
``neighbor_ids (N, K)``, ``k`` and ``motion_key``.

CLI::

    python -m core.data.knn_cache --data-dir data/MSAD --dataset MSAD
        [--k 10] [--motion-key] [--clip-dir ...] [--flow-dir ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import faiss
import numpy as np

from core import constants
from core.data.windows import FeatureSlicer, load_windows

LOGGER = logging.getLogger(__name__)

# torch and faiss-cpu each bundle libomp on macOS arm64; running faiss
# multi-threaded after torch is imported segfaults in IndexFlat.search.
# Single-threaded faiss is plenty at our index sizes.
faiss.omp_set_num_threads(1)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def central_frame_key(
    clip_dir: Path, video_id: str, slicer: FeatureSlicer | None = None
) -> np.ndarray:
    """L2-normalized central-frame CLIP feature ``(D,)`` for ``video_id``.

    On a windowed corpus (lesson **C28**) the id is a window id and the central
    frame must be the *window's* centre, not the source clip's -- two windows of
    one clip are different fillers and must get different keys.
    """
    slicer = slicer if slicer is not None else FeatureSlicer()
    features = slicer.load(clip_dir, video_id)
    return _l2_normalize(features[len(features) // 2].astype(np.float32))


def motion_descriptor(
    flow_dir: Path, video_id: str, slicer: FeatureSlicer | None = None
) -> np.ndarray:
    """Coarse motion descriptor (4,) from the cached raw flow statistics.

    [mean magnitude-mean, mean magnitude-std, mean-direction cos, sin],
    L2-normalized. Requires the ``.stats.npy`` files written by raft_extract.
    """
    slicer = slicer if slicer is not None else FeatureSlicer()
    stats_path = flow_dir / f"{slicer.source_of(video_id)}{constants.FLOW_STATS_SUFFIX}"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Motion-aware KNN key needs {stats_path}; run raft_extract first"
        )
    stats = slicer.load(flow_dir, video_id, constants.FLOW_STATS_SUFFIX)  # (L, FLOW_STATS_DIM)
    histogram = stats[:, 7:].mean(axis=0)  # mean angle histogram
    bin_centers = (np.arange(len(histogram)) + 0.5) / len(histogram) * 2 * math.pi - math.pi
    direction = np.array(
        [float(histogram @ np.cos(bin_centers)), float(histogram @ np.sin(bin_centers))]
    )
    descriptor = np.array(
        [stats[:, 0].mean(), stats[:, 1].mean(), direction[0], direction[1]],
        dtype=np.float32,
    )
    return _l2_normalize(descriptor)


def build_key(
    clip_dir: Path,
    video_id: str,
    motion_key: bool,
    flow_dir: Path | None,
    slicer: FeatureSlicer | None = None,
) -> np.ndarray:
    key = central_frame_key(clip_dir, video_id, slicer)
    if motion_key:
        if flow_dir is None:
            raise ValueError("motion_key=True requires flow_dir")
        key = _l2_normalize(
            np.concatenate([key, motion_descriptor(flow_dir, video_id, slicer)])
        )
    return key


def build_knn_cache(
    labels: dict[str, int],
    clip_dir: Path,
    output_path: Path,
    k: int = constants.KNN_TOP_K,
    motion_key: bool = False,
    flow_dir: Path | None = None,
    slicer: FeatureSlicer | None = None,
) -> dict[str, list[str]]:
    """Build and save the abnormal->normal-neighbors mapping; returns it."""
    normal_ids = sorted(vid for vid, label in labels.items() if label == 0)
    anomaly_ids = sorted(vid for vid, label in labels.items() if label == 1)
    if not normal_ids or not anomaly_ids:
        raise ValueError(
            f"Need both classes: {len(normal_ids)} normal, {len(anomaly_ids)} abnormal"
        )
    k = min(k, len(normal_ids))

    normal_keys = np.stack(
        [build_key(clip_dir, vid, motion_key, flow_dir, slicer) for vid in normal_ids]
    )
    anomaly_keys = np.stack(
        [build_key(clip_dir, vid, motion_key, flow_dir, slicer) for vid in anomaly_ids]
    )

    index = faiss.IndexFlatIP(normal_keys.shape[1])
    index.add(np.ascontiguousarray(normal_keys))
    _scores, neighbor_rows = index.search(np.ascontiguousarray(anomaly_keys), k)

    neighbor_ids = np.array(
        [[normal_ids[j] for j in row] for row in neighbor_rows], dtype=object
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        anomaly_ids=np.array(anomaly_ids, dtype=object),
        neighbor_ids=neighbor_ids,
        k=k,
        motion_key=motion_key,
    )
    LOGGER.info(
        "KNN cache: %d abnormal x top-%d of %d normal -> %s",
        len(anomaly_ids),
        k,
        len(normal_ids),
        output_path,
    )
    return {vid: list(row) for vid, row in zip(anomaly_ids, neighbor_ids, strict=True)}


def load_knn_cache(path: Path) -> dict[str, list[str]]:
    """Load the mapping ``{abnormal_id: [normal neighbor ids]}``."""
    payload = np.load(path, allow_pickle=True)
    return {
        str(vid): [str(n) for n in row]
        for vid, row in zip(payload["anomaly_ids"], payload["neighbor_ids"], strict=True)
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="dataset dir holding labels_train.json (default: data/{dataset})")
    parser.add_argument("--dataset", default=constants.MSAD_DATASET)
    parser.add_argument("--clip-dir", type=Path, default=None,
                        help="default: cache/clip/{dataset}")
    parser.add_argument("--flow-dir", type=Path, default=None,
                        help="default: cache/flow/v1/{dataset} (motion key only)")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: cache/knn/{dataset}/knn_cache.npz")
    parser.add_argument("--k", type=int, default=constants.KNN_TOP_K)
    parser.add_argument("--motion-key", action="store_true",
                        help="append the coarse motion descriptor (ego-centric sets)")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    data_dir = args.data_dir or constants.DATA_ROOT / args.dataset
    with (data_dir / constants.LABELS_TRAIN_FILENAME).open("r", encoding="utf-8") as fh:
        labels: dict[str, int] = json.load(fh)
    build_knn_cache(
        labels=labels,
        clip_dir=args.clip_dir or constants.CLIP_CACHE_DIR / args.dataset,
        output_path=args.output
        or constants.KNN_CACHE_DIR / args.dataset / constants.KNN_CACHE_FILENAME,
        k=args.k,
        motion_key=args.motion_key,
        flow_dir=args.flow_dir or constants.FLOW_CACHE_DIR / args.dataset,
        # A windowed corpus keys labels_train.json by window id, so the keys must
        # be the windows' own central frames (lesson C28).
        slicer=FeatureSlicer(load_windows(data_dir)),
    )


if __name__ == "__main__":
    main()
