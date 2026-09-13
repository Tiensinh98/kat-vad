"""Training / eval datasets over cached CLIP features (MSAD-first, generic).

``DVSFeatureDataset`` generalizes the baseline ``PreVADDatasetOnline``:
``__len__ = 2 x num_anomaly`` (A9 balance; first half anomaly anchors, second
half normal anchors), θ-gated dynamic video synthesis with 50% KNN / 50%
random-normal fillers, and per-item ``e_O`` flow targets for the KIP losses.
Samples are variable-length (truncated to ``vis_max_len``, never padded);
batch padding + masks come from :func:`core.data.collate.collate_variable_length`.

``FeatureEvalDataset`` serves frame-labeled test videos at full length.

Both datasets read their feature rows through :class:`core.data.windows.FeatureSlicer`,
so a corpus rebuilt into fixed-length windows (``windows.json`` present, lesson
**C28**) resolves each item id to a slice of its source clip's cached ``.npy``.
Without that file the slicer is the identity and nothing changes.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from core import constants
from core.config import DVSConfig
from core.data.msad import ABNORMAL_CLASS, NORMAL_CLASS
from core.data.synthesis import choose_filler_ids, compose_sequence, truncate_sample
from core.data.windows import FeatureSlicer, load_windows

LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class DVSFeatureDataset(Dataset[dict[str, Any]]):
    """Weakly-supervised training set with θ-gated dynamic video synthesis.

    Yields per item: ``v_feat (T,512)``, ``e_o (T,256)``, ``pseudo_frame_label
    (T,)``, scalar ``label``/``is_synthesized`` tensors, ``cls_label`` string
    and ``video_id``. Flow rows are zeros when ``require_flow=False`` (KIP-off).
    """

    def __init__(
        self,
        data_dir: Path,
        clip_dir: Path,
        flow_dir: Path | None = None,
        dvs: DVSConfig | None = None,
        vis_max_len: int = constants.MAX_VIS_LEN,
        is_egocentric: bool = False,
        require_flow: bool = True,
        knn_cache: dict[str, list[str]] | None = None,
        seed: int = constants.SEED,
    ) -> None:
        self.clip_dir = clip_dir
        self.flow_dir = flow_dir
        self.slicer = FeatureSlicer(load_windows(data_dir))
        self.dvs = dvs if dvs is not None else DVSConfig()
        self.vis_max_len = vis_max_len
        self.require_flow = require_flow
        self.knn_cache = knn_cache or {}
        # DVS sampling is training augmentation, not security-sensitive
        self._rng = random.Random(seed)  # nosec B311

        labels: dict[str, int] = _load_json(data_dir / constants.LABELS_TRAIN_FILENAME)
        self.anomaly_ids = sorted(vid for vid, lab in labels.items() if lab == 1)
        self.normal_ids = sorted(vid for vid, lab in labels.items() if lab == 0)
        if not self.anomaly_ids or not self.normal_ids:
            raise ValueError(
                f"Training set needs both classes: {len(self.normal_ids)} normal, "
                f"{len(self.anomaly_ids)} abnormal in {data_dir}"
            )

        meta_path = data_dir / constants.META_FILENAME
        self._classes: dict[str, str] = {}
        if meta_path.exists():
            meta = _load_json(meta_path)
            self._classes = {vid: info["class_name"] for vid, info in meta.items()}

        if is_egocentric:
            self.theta = self.dvs.theta_ego
            self.max_clips = min(self.dvs.syn_max_num_clips, self.dvs.delta_m_ego)
        else:
            self.theta = self.dvs.theta
            self.max_clips = min(self.dvs.syn_max_num_clips, self.dvs.delta_m)

    def __len__(self) -> int:
        return 2 * len(self.anomaly_ids)  # A9: anomaly half + normal half

    def _class_name(self, video_id: str, is_abnormal: bool) -> str:
        fallback = ABNORMAL_CLASS if is_abnormal else NORMAL_CLASS
        return self._classes.get(video_id, fallback)

    def _load_features(self, video_id: str) -> Tensor:
        return torch.from_numpy(
            self.slicer.load(self.clip_dir, video_id).astype(np.float32)
        )

    def _load_flow(self, video_id: str, length: int) -> Tensor:
        if not self.require_flow:
            return torch.zeros(length, constants.FLOW_DIM)
        if self.flow_dir is None:
            raise ValueError("require_flow=True but no flow_dir configured")
        path = self.flow_dir / f"{self.slicer.source_of(video_id)}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing flow cache {path}; run raft_extract or set require_flow=False"
            )
        # Sliced by the same window as the appearance rows -- an offset between
        # the two branches is lesson C13 in miniature.
        flow = torch.from_numpy(self.slicer.load(self.flow_dir, video_id).astype(np.float32))
        if len(flow) != length:
            raise ValueError(
                f"Flow/feature length mismatch for {video_id}: {len(flow)} vs {length}"
            )
        return flow

    def _load_clip(self, video_id: str) -> tuple[Tensor, Tensor]:
        features = self._load_features(video_id)
        return features, self._load_flow(video_id, len(features))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        is_anomaly_half = index < len(self.anomaly_ids)
        anchor_id = (
            self.anomaly_ids[index]
            if is_anomaly_half
            else self._rng.choice(self.normal_ids)
        )
        anchor_feats, anchor_flow = self._load_clip(anchor_id)

        # θ = NO-synthesis probability; the synthesis branch always splices
        # at least one filler (num_clips uniform in [2, max_clips], anchor incl.)
        if self._rng.random() < self.theta or self.max_clips <= 1:
            num_fillers = 0
        else:
            num_fillers = self._rng.randint(1, self.max_clips - 1)

        filler_ids = choose_filler_ids(
            self._rng,
            num_fillers,
            self.normal_ids,
            self.knn_cache.get(anchor_id),
            self.dvs.knn_filler_ratio,
        )
        fillers = [(vid, *self._load_clip(vid)) for vid in filler_ids]
        sample = compose_sequence(
            anchor_id=anchor_id,
            anchor_features=anchor_feats,
            anchor_flow=anchor_flow,
            anchor_is_abnormal=is_anomaly_half,
            fillers=fillers,
            insert_index=self._rng.randint(0, len(fillers)),
        )
        sample = truncate_sample(sample, self.vis_max_len)

        return {
            "v_feat": sample.features,
            "e_o": sample.flow,
            "pseudo_frame_label": sample.pseudo_label,
            "label": torch.tensor(int(is_anomaly_half)),
            "is_synthesized": sample.is_synthesized,
            "cls_label": self._class_name(anchor_id, is_anomaly_half),
            "video_id": anchor_id,
        }


class FeatureEvalDataset(Dataset[dict[str, Any]]):
    """Frame-labeled test videos at full length (sliding-window eval in §10)."""

    def __init__(self, data_dir: Path, clip_dir: Path) -> None:
        self.clip_dir = clip_dir
        self.slicer = FeatureSlicer(load_windows(data_dir))
        frame_labels: dict[str, list[int]] = _load_json(
            data_dir / constants.FRAME_LABELS_TEST_FILENAME
        )
        self.video_ids = sorted(frame_labels)
        self.frame_labels = frame_labels
        if not self.video_ids:
            raise ValueError(f"No test videos in {data_dir}")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        video_id = self.video_ids[index]
        features = torch.from_numpy(
            self.slicer.load(self.clip_dir, video_id).astype(np.float32)
        )
        labels = torch.tensor(self.frame_labels[video_id], dtype=torch.float32)
        length = min(len(features), len(labels))
        if len(features) != len(labels):
            LOGGER.warning(
                "%s: feature length %d != label length %d; truncating to %d",
                video_id,
                len(features),
                len(labels),
                length,
            )
        return {
            "v_feat": features[:length],
            "frame_label": labels[:length],
            "label": torch.tensor(int(bool(labels.max() > 0))),
            "video_id": video_id,
        }


__all__ = ["DVSFeatureDataset", "FeatureEvalDataset"]
