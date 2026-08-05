"""Deterministic synthetic MSAD-style fixture (Phase 4/5 tests, no downloads).

Builds, under a given root, a miniature dataset in the exact on-disk contract:
an MSAD 5-column annotation table, standardized label files (produced by the
*real* :mod:`core.data.msad` preprocessor), random CLIP features, random flow
embeddings + stats (with a persisted A10 projection), a KNN cache built by the
real builder, and (optionally) tiny real ``.mp4`` videos for extractor tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import constants
from core.data import msad
from core.data.knn_cache import build_knn_cache
from core.flow.raft_extract import make_projection, save_projection

SCENARIOS = ("highway", "road", "parking lot")
FIXTURE_SEED = 7
MIN_SAMPLED_LEN = 12  # keep videos comfortably longer than top-k denominators
MAX_SAMPLED_LEN = 40
VIDEO_SIZE = 48  # tiny square frames for optional real .mp4 generation


@dataclass
class FixtureLayout:
    """Paths of one generated fixture dataset."""

    root: Path
    data_dir: Path
    videos_dir: Path
    clip_dir: Path
    flow_dir: Path
    flow_cache_root: Path
    knn_cache_path: Path
    annotation_path: Path
    train_ids: list[str]
    test_ids: list[str]


def _annotation_rows(
    rng: np.random.Generator, num_abnormal: int, num_normal: int, stride: int
) -> list[tuple[str, str, int, int | None, int | None]]:
    rows: list[tuple[str, str, int, int | None, int | None]] = []
    for i in range(num_abnormal):
        total = int(rng.integers(MIN_SAMPLED_LEN, MAX_SAMPLED_LEN)) * stride
        start = int(rng.integers(0, total // 2))
        end = int(rng.integers(start + stride, total - 1))
        rows.append(
            (f"Traffic_accident_{i:03d}", SCENARIOS[i % len(SCENARIOS)], total, start, end)
        )
    for i in range(num_normal):
        total = int(rng.integers(MIN_SAMPLED_LEN, MAX_SAMPLED_LEN)) * stride
        rows.append((f"Normal_{i:03d}", SCENARIOS[i % len(SCENARIOS)], total, None, None))
    return rows


def build_fixture(
    root: Path,
    num_abnormal: int = 6,
    num_normal: int = 8,
    stride: int = constants.FRAME_STRIDE,
    seed: int = FIXTURE_SEED,
    with_videos: bool = False,
) -> FixtureLayout:
    """Generate the full fixture; deterministic for a given ``seed``."""
    rng = np.random.default_rng(seed)
    data_dir = root / "data" / constants.MSAD_DATASET
    videos_dir = data_dir / constants.VIDEOS_DIRNAME
    clip_dir = root / "cache" / "clip" / constants.MSAD_DATASET
    flow_cache_root = root / "cache" / "flow" / constants.FLOW_CACHE_VERSION
    flow_dir = flow_cache_root / constants.MSAD_DATASET
    knn_cache_path = (
        root / "cache" / "knn" / constants.MSAD_DATASET / constants.KNN_CACHE_FILENAME
    )
    for directory in (data_dir, clip_dir, flow_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. annotation table (tab-separated, with header) + real preprocessing
    rows = _annotation_rows(rng, num_abnormal, num_normal, stride)
    annotation_path = data_dir / constants.ANNOTATIONS_DIRNAME / "anomaly_annotation.tsv"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["name\tscenario\ttotal frames\tstarting frame of anomaly\tending frame of anomaly"]
    for name, scenario, total, start, end in rows:
        start_cell = "" if start is None else str(start)
        end_cell = "" if end is None else str(end)
        lines.append(f"{name}.mp4\t{scenario}\t{total}\t{start_cell}\t{end_cell}")
    annotation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    train, test = msad.preprocess(
        annotation=annotation_path, out_dir=data_dir, stride=stride, seed=seed
    )

    # 2. per-video CLIP features + flow embeddings/stats (random, deterministic).
    # Flow stats are a fixed function of the features (+ noise) so the PMG head
    # has a learnable feature→flow mapping (stage-1 E2E asserts L_KIP_rec drops).
    projection = make_projection()
    save_projection(
        flow_cache_root / constants.FLOW_PROJECTION_FILENAME,
        projection,
        constants.FLOW_PROJECTION_SEED,
    )
    feat_to_stats = rng.standard_normal(
        (constants.CLIP_FEATURE_DIM, constants.FLOW_STATS_DIM)
    ) / np.sqrt(constants.CLIP_FEATURE_DIM)
    for record in train + test:
        length = msad.num_sampled_frames(record.total_frames, stride)
        features = rng.standard_normal((length, constants.CLIP_FEATURE_DIM))
        np.save(clip_dir / f"{record.video_id}.npy", features.astype(np.float32))
        stats = np.abs(
            features @ feat_to_stats
            + 0.05 * rng.standard_normal((length, constants.FLOW_STATS_DIM))
        )
        np.save(
            flow_dir / f"{record.video_id}{constants.FLOW_STATS_SUFFIX}",
            stats.astype(np.float32),
        )
        np.save(
            flow_dir / f"{record.video_id}.npy", (stats @ projection).astype(np.float32)
        )

    # 3. KNN filler cache over the train split (real builder)
    labels_train = {r.video_id: int(r.is_abnormal) for r in train}
    build_knn_cache(
        labels=labels_train,
        clip_dir=clip_dir,
        output_path=knn_cache_path,
        k=min(constants.KNN_TOP_K, 5),
    )

    # 4. optional tiny real videos (for CLIP/RAFT extractor tests)
    if with_videos:
        _write_videos(videos_dir, rows, seed)

    return FixtureLayout(
        root=root,
        data_dir=data_dir,
        videos_dir=videos_dir,
        clip_dir=clip_dir,
        flow_dir=flow_dir,
        flow_cache_root=flow_cache_root,
        knn_cache_path=knn_cache_path,
        annotation_path=annotation_path,
        train_ids=sorted(r.video_id for r in train),
        test_ids=sorted(r.video_id for r in test),
    )


def _write_videos(
    videos_dir: Path,
    rows: list[tuple[str, str, int, int | None, int | None]],
    seed: int,
) -> None:
    """Write tiny mp4s (moving-square content) with cv2; exact frame counts."""
    import cv2

    videos_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    for name, _scenario, total, _start, _end in rows:
        writer = cv2.VideoWriter(
            str(videos_dir / f"{name}.mp4"), fourcc, 10.0, (VIDEO_SIZE, VIDEO_SIZE)
        )
        try:
            base = rng.integers(0, 128, size=3)
            for frame_index in range(total):
                frame = np.full(
                    (VIDEO_SIZE, VIDEO_SIZE, 3), base, dtype=np.uint8
                )
                offset = (frame_index * 2) % (VIDEO_SIZE - 8)
                frame[offset : offset + 8, offset : offset + 8] = 255
                writer.write(frame)
        finally:
            writer.release()


__all__ = ["FixtureLayout", "build_fixture"]
