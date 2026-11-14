"""core.tools.zs_arm_table: pooling per set, paired deltas, and the refusals. Data-free, CPU."""

from __future__ import annotations

import numpy as np
import pytest

from core import constants
from core.metrics import frame_auc
from core.tools import zs_arm_table as zt
from core.tools.zs_lagovad_readout import minmax_micro

DRAWS = 50
CONTROL, ARM = "K5", "K5ln"


def _curves(seed: int, n: int = 12, length: int = 20, normal_videos: int = 0):
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(n):
        v = f"v{i:02d}"
        y = np.zeros(length, dtype=np.int64)
        if i >= normal_videos:
            y[8:14] = 1
        level = rng.uniform(-2.0, 2.0)  # per-video offset: min-max and raw pool differently
        scores[v] = level + y * rng.uniform(0.2, 1.0) + rng.normal(0.0, 0.5, length)
        labels[v] = y
    return scores, labels


Curves = dict[str, np.ndarray]


def _read(scores: Curves, labels: Curves, set_name: str) -> dict:
    items = {v: frame_auc(scores[v], labels[v]) for v in labels
             if 0 < labels[v].sum() < len(labels[v])}
    raw = frame_auc(np.concatenate([scores[v] for v in sorted(labels)]),
                    np.concatenate([labels[v] for v in sorted(labels)]))
    # as each reader records it: DoTA after min-max, every other set raw
    micro = minmax_micro(scores, labels) if set_name == zt.DOTA else raw
    mean = float(np.mean(list(items.values())))
    return {"items": items, "groups": {v: v for v in items},
            "macro": {"mean": mean, "low": mean - 0.1, "high": mean + 0.1},
            "micro": micro, "clip_auc": None, "extra": {}}


def _set(set_name, normal_videos=0):
    loaded = {CONTROL: _curves(1, normal_videos=normal_videos)}
    s2, _ = _curves(2, normal_videos=normal_videos)
    loaded[ARM] = (s2, loaded[CONTROL][1])
    rows = {a: _read(loaded[a][0], loaded[a][1], set_name) for a in loaded}
    return rows, loaded


@pytest.mark.parametrize("set_name", [zt.DOTA, zt.DADA])
def test_all_abnormal_sets_pool_after_minmax(set_name: str) -> None:
    rows, loaded = _set(set_name)
    out = zt.set_table(set_name, rows, loaded, CONTROL, DRAWS)
    for arm in (CONTROL, ARM):
        assert out["cells"][arm]["micro"] == pytest.approx(minmax_micro(*loaded[arm]), abs=1e-12)


@pytest.mark.parametrize("set_name", [zt.TAD, zt.DAD])
def test_sets_with_normal_videos_pool_raw(set_name: str) -> None:
    rows, loaded = _set(set_name, normal_videos=4)
    out = zt.set_table(set_name, rows, loaded, CONTROL, DRAWS)
    scores, labels = loaded[CONTROL]
    raw = frame_auc(np.concatenate([scores[v] for v in sorted(labels)]),
                    np.concatenate([labels[v] for v in sorted(labels)]))
    assert out["cells"][CONTROL]["micro"] == pytest.approx(raw, abs=1e-12)
    # the fixture can tell the two poolings apart
    assert raw != pytest.approx(minmax_micro(scores, labels), abs=1e-6)


def test_deltas_are_arm_minus_control_with_an_interval() -> None:
    rows, loaded = _set(zt.DADA)
    out = zt.set_table(zt.DADA, rows, loaded, CONTROL, DRAWS)
    d = out["deltas"][ARM]["micro"]
    assert d["mean"] == pytest.approx(out["cells"][ARM]["micro"] - out["cells"][CONTROL]["micro"])
    assert d["low"] <= d["mean"] <= d["high"]
    assert set(out["deltas"]) == {ARM}  # no control-minus-control row
    assert out["deltas"][ARM]["micro_sentence"] in ("above", "below", "not separated")


def test_minmax_leaves_macro_alone() -> None:
    scores, labels = _curves(3)
    normed = zt.table_curves(scores, zt.DADA)
    for v in labels:
        assert frame_auc(normed[v], labels[v]) == pytest.approx(frame_auc(scores[v], labels[v]))
    assert zt.table_curves(scores, zt.TAD) is scores


def test_other_labels_raise() -> None:
    rows, loaded = _set(zt.DADA)
    s, y = loaded[ARM]
    y = {v: a.copy() for v, a in y.items()}
    y["v00"][0] = 1
    loaded[ARM] = (s, y)
    with pytest.raises(ValueError, match="other labels"):
        zt.set_table(zt.DADA, rows, loaded, CONTROL, DRAWS)


def test_a_score_file_that_is_not_the_read_raises() -> None:
    rows, loaded = _set(zt.DADA)
    rows[ARM]["micro"] += 0.01
    with pytest.raises(ValueError, match="not the same scores"):
        zt.set_table(zt.DADA, rows, loaded, CONTROL, DRAWS)


def test_missing_control_raises() -> None:
    rows, loaded = _set(zt.DADA)
    with pytest.raises(ValueError, match="control"):
        zt.set_table(zt.DADA, {ARM: rows[ARM]}, {ARM: loaded[ARM]}, CONTROL, DRAWS)


def test_render_has_a_summary_cell_per_arm_and_set() -> None:
    table = {}
    for set_name in (zt.DOTA, zt.TAD):
        rows, loaded = _set(set_name, normal_videos=0 if set_name == zt.DOTA else 4)
        table[set_name] = zt.set_table(set_name, rows, loaded, CONTROL, DRAWS)
    md = zt.render(table, CONTROL, constants.KIM_SEED, "k5ln_s2024")
    assert "| K5ln - K5 |" in md and "## Summary (macro / micro)" in md
    summary = md.split("## Summary")[1].split("\n## ")[0]
    assert summary.count(" / ") >= 4  # 2 arms x 2 sets
