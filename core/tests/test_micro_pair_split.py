"""Tests for ``core.tools.micro_pair_split`` (KIM_RUN §62)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.metrics import frame_auc
from core.tools import micro_pair_split as mp
from core.tools.protocol_b_eval import CLIP_SCORES_NPZ, write_clip_scores

Curves = dict[str, np.ndarray]


def _curves(seed: int, pre_lift: float = 0.0) -> tuple[Curves, Curves]:
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(20):
        n = 30 + i
        y = np.zeros(n, dtype=np.int64)
        y[12:20] = 1
        s = 0.4 * y + 0.3 * rng.random(n)
        s[:12] += pre_lift * rng.random()  # a clip-dependent lift of the pre-onset frames
        scores[f"vid{i // 2}_{i:06d}"], labels[f"vid{i // 2}_{i:06d}"] = s, y
    return scores, labels


def test_frame_kinds_places_negatives() -> None:
    y = np.array([0, 0, 1, 0, 1, 0])
    want = [mp.PRE, mp.PRE, mp.POS, mp.GAP, mp.POS, mp.POST]
    np.testing.assert_array_equal(mp.frame_kinds(y), want)


def test_parts_rebuild_pooled_minmax_micro() -> None:
    scores, labels = _curves(0, pre_lift=0.5)
    ids = sorted(labels)
    row = mp.Row(scores, labels, ids)
    out = row.parts(np.ones(len(ids)))
    pooled = frame_auc(np.concatenate([mp.minmax(scores[v]) for v in ids]),
                       np.concatenate([labels[v] for v in ids]))
    assert out["micro"] == pytest.approx(pooled, abs=1e-9)
    assert out["macro"] == pytest.approx(np.mean([frame_auc(scores[v], labels[v]) for v in ids]))


def test_cross_matches_brute_force() -> None:
    scores, labels = _curves(1, pre_lift=0.3)
    ids = sorted(labels)
    out = mp.Row(scores, labels, ids).parts(np.ones(len(ids)))
    mm = {v: mp.minmax(scores[v]) for v in ids}
    wins = pairs = 0.0
    for a in ids:
        for b in ids:
            if a == b:
                continue
            p = mm[a][labels[a] > 0][:, None]
            n = mm[b][(labels[b] == 0) & (mp.frame_kinds(labels[b]) == mp.PRE)][None, :]
            wins += float((p > n).sum() + 0.5 * (p == n).sum())
            pairs += p.size * n.size
    assert out["x_pre"] == pytest.approx(wins / pairs, abs=1e-9)


def test_duplicate_clip_weight_counts_as_one_clip() -> None:
    scores, labels = _curves(2)
    ids = sorted(labels)
    row = mp.Row(scores, labels, ids)
    w = np.ones(len(ids))
    w[0] = 2.0
    out = row.parts(w)
    assert np.isfinite(out["x_all"]) and 0.0 <= out["w_within"] < 1.0


def test_pre_lift_is_read_as_cross_clip_pre(tmp_path: Path) -> None:
    s_off, y = _curves(3)
    s_on, _ = _curves(3, pre_lift=0.6)
    for name, s in (("on", s_on), ("off", s_off)):
        (tmp_path / name).mkdir()
        write_clip_scores(tmp_path / name / CLIP_SCORES_NPZ, s, y)
    r = mp.read_pair("t", "dota", [tmp_path / "on"], [tmp_path / "off"])
    assert r["delta"]["x_pre"]["mean"] < 0
    assert r["level"]["on"]["pre"] > r["level"]["off"]["pre"]
    assert r["reading"].startswith(("cross-clip", "within-clip"))


def test_mismatched_labels_raise(tmp_path: Path) -> None:
    s, y = _curves(4)
    y2 = {k: v.copy() for k, v in y.items()}
    first = sorted(y2)[0]
    y2[first][5] = 1
    for name, lab in (("on", y), ("off", y2)):
        (tmp_path / name).mkdir()
        write_clip_scores(tmp_path / name / CLIP_SCORES_NPZ, s, lab)
    with pytest.raises(ValueError, match="labels differ"):
        mp.read_pair("t", "dota", [tmp_path / "on"], [tmp_path / "off"])


def test_parse_pair_rejects_bad_set() -> None:
    with pytest.raises(ValueError):
        mp.parse_pair("a|tad|x|y")
    label, set_name, on, off = mp.parse_pair("a|dota|x+y|z")
    assert (label, set_name, on, off) == ("a", "dota", [Path("x"), Path("y")], [Path("z")])
