"""D6 / N11 shuffle-control read-out (core.tools.e2d_shuffle) on synthetic caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.data.dota import DotaRecord
from core.data.v2_splits import lines_sha1
from core.tools import dota_cap
from core.tools import e2d_shuffle as shuf

S = constants.VIDEOMAE_ENCODER_S
DIM_X, DIM_U = 6, 4
ALIGNMENT = "a" * 64
SEED = constants.V2_D6_SHUFFLE_SEED


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``dota_cap_dev`` whose ordered ``u`` carries the label and whose shuffled ``u`` is noise."""
    rng = np.random.default_rng(3)
    cap = [f"vid{g:02d}_{i:06d}" for g in range(10) for i in range(3)]
    split_dir, s1_dir = tmp_path / "splits", tmp_path / "s1"
    split_dir.mkdir()
    s1_dir.mkdir()
    (split_dir / "dota_cap_dev.txt").write_text("".join(f"{v}\n" for v in cap))
    (split_dir / constants.V2_SPLITS_MANIFEST_FILENAME).write_text(json.dumps({"splits": {}}))
    (split_dir / constants.V2_DOTA_CAP_MANIFEST_FILENAME).write_text(json.dumps({
        "source": {"alignment_sha256": ALIGNMENT},
        "splits": {"dota_cap_dev": {"count": len(cap), "sha1": lines_sha1(cap)}},
    }))
    dirs = shuf.shuffle_dirs(tmp_path / "video", S, SEED)
    for side, d in dirs.items():
        d.mkdir(parents=True)
        manifest = dota_cap.video_manifest(S, ALIGNMENT, SEED if side == shuf.SHUFFLED else None)
        (d / constants.VIDEO_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    records = {}
    for v in cap:
        n = int(rng.integers(30, 60))
        np.save(s1_dir / f"{v}.npy", rng.normal(size=(n, DIM_X)).astype(np.float32))
        records[v] = DotaRecord(v, "ego: turning", n, (0.4, 0.7))
        t = (np.arange(n) + 0.5) / n
        signal = ((t >= 0.4) & (t < 0.7)).astype(np.float32)[:, None]
        ordered = rng.normal(size=(n, DIM_U)).astype(np.float32) * 0.3 + 2.0 * signal
        np.save(dirs[shuf.ORDERED] / f"{v}.npy", ordered)
        np.save(dirs[shuf.SHUFFLED] / f"{v}.npy", rng.normal(size=(n, DIM_U)).astype(np.float32))
    monkeypatch.setattr(shuf, "parse_metadata", lambda *_a: list(records.values()))
    monkeypatch.setattr(shuf, "read_split_ids", lambda *_a: set(cap))
    return [
        "--dota-s1-dir", str(s1_dir), "--metadata", "m", "--split-file", "s",
        "--video-root", str(tmp_path / "video"), "--split-dir", str(split_dir),
        "--resamples", "50", "--folds", "3",
    ]


class TestShuffleReadout:
    def test_ordered_signal_beats_shuffled_noise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = _setup(tmp_path, monkeypatch)
        out = tmp_path / "out"
        shuf.main([*args, "--out-dir", str(out)])
        readout = json.loads((out / shuf.READOUT_JSON).read_text())
        assert readout["frame_order"] == dota_cap.video_manifest(S, ALIGNMENT, SEED)["frame_order"]
        for arm in ("A2", "A3"):
            delta = readout["per_arm"][arm]["ordered_minus_shuffled"]
            assert delta["u"]["mean"] > 0.2 and delta["with"]["mean"] > 0.1
        assert "ordered - shuffled" in (out / shuf.READOUT_MD).read_text()

    def test_an_ordered_cache_in_the_shuffle_slot_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = _setup(tmp_path, monkeypatch)
        slot = shuf.shuffle_dirs(tmp_path / "video", S, SEED)[shuf.SHUFFLED]
        (slot / constants.VIDEO_MANIFEST_FILENAME).write_text(
            json.dumps(dota_cap.video_manifest(S, ALIGNMENT))
        )
        with pytest.raises(ValueError, match="frame_order"):
            shuf.main([*args, "--out-dir", str(tmp_path / "out")])
