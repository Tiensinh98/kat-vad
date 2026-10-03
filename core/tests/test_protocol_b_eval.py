"""Protocol-B scoring of any v2 arm (E1's protocol on the arm's own input cache).

The failures that matter are silent: a baked stride-3 cache subsampled a second time, a plain
cache scored by a CRN checkpoint, or a curve placed at the wrong native frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.tools import protocol_b_eval as pb
from core.tools import rate_matched_eval as e1

DIM = 4


def _pointwise_scores(
    _model: Any, feats: torch.Tensor, class_feats_fn: Any, max_vis_len: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    class_feats_fn()
    return feats[:, 0].clone(), torch.zeros(len(feats), 1)


def _baked(path: Path, stride: int, crn: str = "R2") -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {"crn": crn, "motion": "none", "stride_over_clip_dir": stride}
    (path / constants.V2_INPUT_MANIFEST_FILENAME).write_text(json.dumps(manifest))


class TestInputRows:
    def test_plain_cache_is_subsampled_here(self, tmp_path: Path) -> None:
        rows = np.arange(10 * DIM, dtype=np.float32).reshape(10, DIM)
        np.save(tmp_path / "v_000001.npy", rows)
        out = pb.input_rows(tmp_path, "v_000001", 10, 3, baked=False)
        np.testing.assert_array_equal(out, rows[::3])

    def test_baked_cache_is_read_as_is_and_its_length_checked(self, tmp_path: Path) -> None:
        np.save(tmp_path / "v_000001.npy", np.ones((4, DIM), dtype=np.float32))
        assert pb.input_rows(tmp_path, "v_000001", 10, 3, baked=True).shape == (4, DIM)
        with pytest.raises(ValueError, match="protocol needs 5"):
            pb.input_rows(tmp_path, "v_000001", 13, 3, baked=True)

    def test_a_double_subsampled_cache_is_refused(self, tmp_path: Path) -> None:
        np.save(tmp_path / "v_000001.npy", np.ones((2, DIM), dtype=np.float32))
        with pytest.raises(ValueError, match="protocol needs 4"):
            pb.input_rows(tmp_path, "v_000001", 10, 3, baked=True)

    def test_plain_cache_must_be_stride_one(self, tmp_path: Path) -> None:
        np.save(tmp_path / "v_000001.npy", np.ones((4, DIM), dtype=np.float32))
        with pytest.raises(ValueError, match="plain cache"):
            pb.input_rows(tmp_path, "v_000001", 10, 3, baked=False)


class TestBakedStride:
    def test_plain_is_false_and_matching_stride_is_true(self, tmp_path: Path) -> None:
        assert not pb.check_baked_stride(tmp_path, 3)
        _baked(tmp_path / "b", 3)
        assert pb.check_baked_stride(tmp_path / "b", 3)

    def test_other_stride_is_refused(self, tmp_path: Path) -> None:
        _baked(tmp_path, 1)
        with pytest.raises(ValueError, match="baked at stride 1"):
            pb.check_baked_stride(tmp_path, 3)


def _fake_loader(crn: str) -> Any:
    def load(_run: e1.Run, _device: torch.device) -> tuple[Any, int, Any, Any]:
        def text_fn(batch: list[str]) -> torch.Tensor:
            return torch.zeros(len(batch), DIM)

        cfg = SimpleNamespace(v2=SimpleNamespace(crn=crn, motion="none"))
        return None, 2040, text_fn, cfg

    return load


class TestScoreCheckpoint:
    @pytest.fixture(autouse=True)
    def _pointwise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(e1, "sliding_window_scores", _pointwise_scores)
        monkeypatch.setattr(pb, "load_class_names", lambda _d: ["Normal", "CarAccident"])

    def _args(self, input_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(input_dir=input_dir, stride=3, data_dir=Path("unused"))

    def test_baked_steps_land_on_native_frames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pb, "load_finished_model", _fake_loader("R2"))
        _baked(tmp_path, 3)
        steps = np.array([[0.0], [3.0], [6.0], [9.0]], dtype=np.float32).repeat(DIM, axis=1)
        np.save(tmp_path / "v_000001.npy", steps)
        native, step, arm = pb.score_checkpoint(
            e1.Run("s", tmp_path / "ckpt.pt"), ["v_000001"], {"v_000001": 11},
            self._args(tmp_path), torch.device("cpu"),
        )
        assert step == 2040 and arm == {"crn": "R2", "motion": "none"}
        np.testing.assert_allclose(native["v_000001"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])

    def test_crn_checkpoint_on_a_plain_cache_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pb, "load_finished_model", _fake_loader("R2"))
        np.save(tmp_path / "v_000001.npy", np.ones((10, DIM), dtype=np.float32))
        with pytest.raises(ValueError, match="plain CLIP cache"):
            pb.score_checkpoint(
                e1.Run("s", tmp_path / "ckpt.pt"), ["v_000001"], {"v_000001": 10},
                self._args(tmp_path), torch.device("cpu"),
            )


class TestClipAucs:
    def test_single_class_clips_are_skipped(self) -> None:
        labels = {"a": np.array([0, 0, 1, 1]), "b": np.array([1, 1, 1, 1])}
        scores = {"a": np.array([0.1, 0.2, 0.8, 0.9]), "b": np.ones(4)}
        assert pb.clip_aucs(scores, labels) == {"a": 1.0}
