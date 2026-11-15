"""core.tools.zs_curves: picks, onset alignment, the refusals and the CLI. Data-free, CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import zs_curves as zc

LENGTH = 20
ONSET = 8


def _curves(seed: int, n: int = 12, normal_videos: int = 0):
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(n):
        v = f"v{i:02d}"
        y = np.zeros(LENGTH, dtype=np.int64)
        if i >= normal_videos:
            y[ONSET:ONSET + 6] = 1
        scores[v] = y * rng.uniform(0.0, 1.0) + rng.normal(0.0, 0.5, LENGTH)
        labels[v] = y
    return scores, labels


def _write_npz(path: Path, scores, labels) -> Path:
    arrays = {f"score/{v}": s for v, s in scores.items()}
    arrays.update({f"label/{v}": labels[v] for v in labels})
    np.savez(path, **arrays)
    return path


def _write_run(run_dir: Path, scores, labels) -> Path:
    (run_dir / "scores").mkdir(parents=True)
    for v in labels:
        np.savez(run_dir / "scores" / f"{v}.npz", score=scores[v], gt=labels[v])
    return run_dir


def test_picks_span_the_auc_rank_and_add_one_normal() -> None:
    scores, labels = _curves(1, normal_videos=2)
    aucs = zc.video_aucs(scores, labels)
    assert set(aucs) == {v for v in labels if labels[v].any()}
    picks = zc.pick_videos(aucs, labels, 4)
    ranked = sorted(aucs, key=lambda v: (aucs[v], v))
    assert picks[0] == ranked[0] and picks[2] == ranked[-1]
    assert picks[-1] == "v00" and not labels["v00"].any()
    assert len(picks) == 4


def test_picks_without_normals_use_every_slot() -> None:
    scores, labels = _curves(2)
    picks = zc.pick_videos(zc.video_aucs(scores, labels), labels, 5)
    assert len(picks) == 5 and all(labels[v].any() for v in picks)
    with pytest.raises(ValueError):
        zc.pick_videos({}, labels, 0)


def test_onset_curve_aligns_at_the_first_gt_row() -> None:
    # a step at the onset: every bin before 0 reads 0, every bin after reads 1
    labels = {f"v{i}": np.r_[np.zeros(ONSET + i), np.ones(6)].astype(np.int64) for i in range(3)}
    scores = {v: g.astype(np.float64) for v, g in labels.items()}
    c = zc.onset_curve(scores, labels, step_s=1.0, use_minmax=False, window_s=4.0, bin_rows=1,
                       min_videos=1)
    centers, mean = np.asarray(c["centers_s"]), c["mean"]
    assert c["videos"] == 3 and c["bin_s"] == 1.0
    assert all(m == 0.0 for x, m in zip(centers, mean, strict=True) if x < 0)
    assert all(m == 1.0 for x, m in zip(centers, mean, strict=True) if 0 < x < 6)


def test_onset_curve_hides_thin_bins_and_skips_one_class_videos() -> None:
    scores, labels = _curves(3, n=4, normal_videos=1)
    c = zc.onset_curve(scores, labels, step_s=0.5, use_minmax=True, window_s=6.0, bin_rows=2,
                       min_videos=3)
    assert c["videos"] == 3
    for m, k in zip(c["mean"], c["n_per_bin"], strict=True):
        assert (m is None) == (k < 3)


def test_load_reads_refuses_other_labels_and_lengths(tmp_path: Path) -> None:
    scores, labels = _curves(4)
    a = _write_npz(tmp_path / "a.npz", scores, labels)
    other = {v: np.roll(g, 1) for v, g in labels.items()}
    b = _write_npz(tmp_path / "b.npz", scores, other)
    with pytest.raises(ValueError, match="other labels"):
        zc.load_reads([("dota", "K5", a), ("dota", "LaGoVAD", b)])
    short = {v: s[:-1] for v, s in scores.items()}
    c = _write_npz(tmp_path / "c.npz", short, labels)
    with pytest.raises(ValueError, match="lengths differ"):
        zc.load_reads([("dota", "K5", c)])
    with pytest.raises(ValueError, match="unknown set"):
        zc.load_reads([("nexar", "K5", a)])
    with pytest.raises(FileNotFoundError):
        zc.load_reads([("dota", "K5", tmp_path / "missing.npz")])


def test_build_refuses_a_missing_rank_model(tmp_path: Path) -> None:
    scores, labels = _curves(5)
    data = zc.load_reads([("dada", "LaGoVAD", _write_npz(tmp_path / "a.npz", scores, labels))])
    with pytest.raises(ValueError, match="rank model"):
        zc.build(data, "K5", 3, None)


def test_cli_writes_grids_onset_and_json(tmp_path: Path) -> None:
    k5, labels = _curves(6)
    lago, _ = _curves(7)
    tk5, tlab = _curves(8, normal_videos=3)
    tlago, _ = _curves(9, normal_videos=3)
    out = tmp_path / "out"
    table = zc.run(zc.build_arg_parser().parse_args([
        "--rank-by", "K5", "--n", "4", "--name", "t", "--out-dir", str(out),
        "--read", "dota", "K5", str(_write_npz(tmp_path / "k5.npz", k5, labels)),
        "--read", "dota", "LaGoVAD", str(_write_npz(tmp_path / "lg.npz", lago, labels)),
        "--read", "tad", "K5", str(_write_run(tmp_path / "tk5", tk5, tlab)),
        "--read", "tad", "LaGoVAD", str(_write_run(tmp_path / "tlg", tlago, tlab)),
    ]))
    assert list(table) == ["dota", "tad"]
    assert table["dota"]["pooling"] == "min-max" and table["tad"]["pooling"] == "raw"
    assert table["tad"]["two_class"] == 9 and not tlab[table["tad"]["picks"][-1]].any()
    assert table["dota"]["step_s"] == constants.ZS_CURVES_STEP_SECONDS["dota"]
    for name in ("grid_dota.png", "grid_tad.png", "onset_aligned.png"):
        assert (out / name).stat().st_size > 0
    saved = json.loads((out / f"{constants.ZS_CURVES_STEM}_t.json").read_text())
    assert saved["rank_by"] == "K5" and set(saved["sets"]["dota"]["onset"]) == {"K5", "LaGoVAD"}


def test_clips_override_the_picks(tmp_path: Path) -> None:
    scores, labels = _curves(10)
    data = zc.load_reads([("dada", "K5", _write_npz(tmp_path / "a.npz", scores, labels))])
    table = zc.build(data, "K5", 3, ["v03", "absent", "v07"])
    assert table["dada"]["picks"] == ["v03", "v07"]
