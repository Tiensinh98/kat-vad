"""v2 TAD clip-level gate (addendum §27): the gate, the fusion and the read-out as pre-registered.

The failures that matter are silent: a gate that moves a within-video ranking (macro would change
under a "clip-level" repair), a sentence written when one interval misses 0, a test read after the
dev gate failed, and a dev split that overlaps the test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import constants
from core.tools import tad_clip_gate as tg
from core.tools import tadz_readout as tz
from core.tools.e3_readout import REPORTS

SEEDS = list(constants.V2_E3_SEEDS)
FRAMES, ONSET, END = 40, 15, 25
DIM = constants.CLIP_FEATURE_DIM


def _interval(low: float, high: float) -> dict[str, Any]:
    return {"mean": (low + high) / 2, "low": low, "high": high}


class TestSentence:
    def test_both_above_zero_restores(self) -> None:
        assert "restores" in tg.gate_sentence(_interval(0.01, 0.05), _interval(0.002, 0.04))

    @pytest.mark.parametrize(
        ("seeds", "videos"), [((0.01, 0.05), (-0.01, 0.04)), ((-0.01, 0.05), (0.01, 0.04))]
    )
    def test_one_interval_is_no_gain(
        self, seeds: tuple[float, float], videos: tuple[float, float]
    ) -> None:
        assert tg.gate_sentence(_interval(*seeds), _interval(*videos)).startswith(
            "no clip-level gain"
        )

    def test_both_below_zero_costs(self) -> None:
        assert "costs" in tg.gate_sentence(_interval(-0.05, -0.01), _interval(-0.04, -0.002))


class TestFusion:
    def test_gate_never_moves_a_within_video_ranking(self) -> None:
        rng = np.random.default_rng(0)
        labels = {f"v{i}": (np.arange(FRAMES) >= ONSET).astype(np.int64) for i in range(6)}
        scores = {v: rng.random(FRAMES) for v in labels}
        fused = tg.fuse(scores, {v: float(rng.uniform(0.01, 1.0)) for v in labels})
        a, b = tz.clip_aucs(fused, labels), tz.clip_aucs(scores, labels)
        assert all(abs(a[v] - b[v]) < constants.V2_TADG_MACRO_TOL for v in a)

    def test_gate_sets_each_video_peak(self) -> None:
        fused = tg.fuse({"v": np.array([0.2, 0.9, 0.5])}, {"v": 0.3})
        assert fused["v"].max() == pytest.approx(0.3)
        assert fused["v"].min() == pytest.approx(0.0)


def _write_features(
    clip_dir: Path, labels: dict[str, int], rng: np.random.Generator, shift: float
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for v, y in labels.items():
        x = rng.normal(size=(int(rng.integers(5, 30)), DIM)) + shift * y
        np.save(clip_dir / f"{v}.npy", x.astype(np.float32))


class TestFitScore:
    def _fit(self, tmp_path: Path, shift: float = 0.5) -> tuple[Path, dict[str, int]]:
        rng = np.random.default_rng(1)
        labels = {f"m{i:03d}": int(i < 20) for i in range(60)}
        _write_features(tmp_path / "msad", labels, rng, shift)
        (tmp_path / "labels.json").write_text(json.dumps(labels))
        args = argparse.Namespace(
            clip_dir=tmp_path / "msad",
            labels=tmp_path / "labels.json",
            out=tmp_path / "gate.npz",
            expect_counts=[20, 40],
            seed=0,
        )
        summary = tg.cmd_fit(args)
        assert summary["train_clip_auc"] > 0.95
        return tmp_path / "gate.npz", labels

    def test_round_trip_scores_without_pickle(self, tmp_path: Path) -> None:
        gate, labels = self._fit(tmp_path)
        with np.load(gate, allow_pickle=False) as data:
            assert set(data.files) == {"mean", "scale", "coef", "intercept"}
        (tmp_path / "ids.txt").write_text("\n".join(labels) + "\n")
        out = tg.cmd_score(
            argparse.Namespace(
                gate=gate,
                clip_dir=tmp_path / "msad",
                ids_file=tmp_path / "ids.txt",
                out=tmp_path / "c.json",
            )
        )
        assert tg.clip_level({v: g["c"] for v, g in out.items()}, labels) > 0.95
        assert all(0.0 < g["c"] < 1.0 and g["rows"] >= 5 for g in out.values())

    def test_wrong_split_counts_raise(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2)
        labels = {f"m{i}": int(i < 5) for i in range(12)}
        _write_features(tmp_path / "msad", labels, rng, 1.0)
        (tmp_path / "labels.json").write_text(json.dumps(labels))
        with pytest.raises(ValueError, match="G9"):
            tg.cmd_fit(
                argparse.Namespace(
                    clip_dir=tmp_path / "msad",
                    labels=tmp_path / "labels.json",
                    out=tmp_path / "g.npz",
                    expect_counts=[120, 360],
                    seed=0,
                )
            )

    def test_wrong_feature_width_raises(self, tmp_path: Path) -> None:
        (tmp_path / "c").mkdir()
        np.save(tmp_path / "c" / "v.npy", np.zeros((4, 768), dtype=np.float32))
        with pytest.raises(ValueError, match="G9"):
            tg.mean_features(tmp_path / "c", ["v"])

    def test_frame_labels_reduce_to_video_labels(self, tmp_path: Path) -> None:
        (tmp_path / "f.json").write_text(json.dumps({"a": [0, 1, 0], "b": [0, 0], "c": 1}))
        assert tg.read_labels(tmp_path / "f.json") == {"a": 1, "b": 0, "c": 1}


# ---------------------------------------------------------------------------
# end-to-end read-out on a synthetic §26 layout
# ---------------------------------------------------------------------------
def _test_ids() -> dict[str, int]:
    ids = {f"01_Accident_{i:03d}": 1 for i in range(constants.V2_TADZ_ABNORMAL)}
    ids.update(
        {f"Normal_{i:03d}": 0 for i in range(constants.V2_TADZ_VIDEOS - constants.V2_TADZ_ABNORMAL)}
    )
    return ids


def _gt(abnormal: int) -> np.ndarray:
    gt = np.zeros(FRAMES, dtype=np.int64)
    if abnormal:
        gt[ONSET:END] = 1
    return gt


def _bump(abnormal: int) -> np.ndarray:
    """The accident span on accident videos; an equally high span elsewhere on normals (§26)."""
    return _gt(1) if abnormal else np.roll(_gt(1), FRAMES // 2)


def _write_run(run_dir: Path, scores: dict[str, np.ndarray]) -> None:
    out = run_dir / tz.SCORES_DIRNAME
    out.mkdir(parents=True)
    for v, s in scores.items():
        np.savez(
            out / f"{v}.npz", score=s.astype(np.float32), gt=_gt(_test_ids()[v]).astype(np.float32)
        )


def _layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate_signal: float
) -> argparse.Namespace:
    """A3-ign localizes inside every video but peaks at 1.0 on normals too (§26's failure)."""
    rng = np.random.default_rng(3)
    ids = _test_ids()
    root = tmp_path / "tadz"
    for k, s in enumerate(SEEDS):
        _write_run(
            root / constants.V2_TADG_ARM / tz.seed_name(s),
            {
                v: np.clip(_bump(y) * 0.6 + rng.random(FRAMES) * 0.4 + 0.01 * k, 0, 1)
                for v, y in ids.items()
            },
        )
    t0 = {v: _gt(y) * 0.3 + rng.random(FRAMES) * 0.5 + 0.2 * y for v, y in ids.items()}
    _write_run(root / constants.V2_TADZ_REFERENCE, t0)
    t0_labels = {v: _gt(y) for v, y in ids.items()}
    monkeypatch.setattr(constants, "V2_TADZ_T0_MICRO", tz.micro(t0, t0_labels))
    monkeypatch.setattr(constants, "V2_E1_BOOTSTRAP", 200)

    def gate(names: dict[str, int]) -> dict[str, dict[str, float]]:
        return {
            v: {
                "c": float(
                    np.clip(0.5 + gate_signal * (y - 0.5) + rng.normal(0, 0.05), 0.01, 0.99)
                ),
                "rows": float(rng.integers(20, 60)),
            }
            for v, y in names.items()
        }

    dev_labels = {f"dev_{i:03d}": int(i % 3 == 0) for i in range(constants.V2_TADG_DEV_MIN_VIDEOS)}
    for name, payload in (
        ("dev.json", gate(dev_labels)),
        ("dev_labels.json", dev_labels),
        ("test.json", gate(ids)),
    ):
        (tmp_path / name).write_text(json.dumps(payload))
    return argparse.Namespace(
        tadz_root=root,
        dev=tmp_path / "dev.json",
        dev_labels=tmp_path / "dev_labels.json",
        test=tmp_path / "test.json",
        msad_test=None,
        msad_test_labels=None,
        out_dir=tmp_path / "out",
        seeds=SEEDS,
        seed=0,
    )


class TestReadout:
    def test_informative_gate_restores_and_keeps_macro(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        r = tg.cmd_readout(_layout(tmp_path, monkeypatch, gate_signal=0.8))
        assert r["dev"]["passed"]
        assert "restores" in r["sentence"]
        assert r["levels"]["c x minmax(A3ign)"]["clip_auc"] > r["levels"]["A3ign"]["clip_auc"]
        md = (tmp_path / "out" / REPORTS / tg.READOUT_MD).read_text()
        assert "0.9226" in md and "89.56" in md and "form seen on this set" in md

    def test_failed_dev_gate_never_reads_the_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = _layout(tmp_path, monkeypatch, gate_signal=0.0)
        args.test.unlink()  # a test read would raise FileNotFoundError
        r = tg.cmd_readout(args)
        assert not r["dev"]["passed"]
        assert "endpoint" not in r and "not read" in r["sentence"]

    def test_dev_overlapping_test_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = _layout(tmp_path, monkeypatch, gate_signal=0.8)
        dev = json.loads(args.dev.read_text())
        labels = json.loads(args.dev_labels.read_text())
        dev["Normal_000"], labels["Normal_000"] = {"c": 0.2, "rows": 30.0}, 0
        args.dev.write_text(json.dumps(dev))
        args.dev_labels.write_text(json.dumps(labels))
        with pytest.raises(ValueError, match="G9"):
            tg.cmd_readout(args)
