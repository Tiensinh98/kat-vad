"""``core.tools.pre_onset_read`` (``KIM_RUN.md`` §43): event-locked reads of nexar_val curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.tools import pre_onset_read as po

FPS = 30.0
LENGTH = 600  # 20 s
ONSET = 360  # 12 s: 7 s of "far" before the 5 s zone
SPAN_END = 420
NEG_LEVEL, SPAN_LEVEL = 0.1, 0.9


def _gt() -> np.ndarray:
    gt = np.zeros(LENGTH, dtype=np.int64)
    gt[ONSET:SPAN_END] = 1
    return gt


def _rising(lead_s: float) -> np.ndarray:
    """Far level, a linear climb starting ``lead_s`` before onset, span level inside the span."""
    score = np.full(LENGTH, NEG_LEVEL)
    start = ONSET - round(lead_s * FPS)
    if lead_s > 0:
        score[start:ONSET] = np.linspace(NEG_LEVEL, SPAN_LEVEL, ONSET - start, endpoint=False)
    score[ONSET:SPAN_END] = SPAN_LEVEL
    return score


def _write(folder: Path, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    scores = folder / "scores"
    scores.mkdir(parents=True)
    for v, (score, gt) in curves.items():
        np.savez(scores / f"{v}.npz", score=score.astype(np.float32), gt=gt.astype(np.float32))
    return folder


def _row(tmp_path: Path, name: str, lead_s: float, n: int = 6) -> Path:
    rng = np.random.default_rng(0)
    curves = {f"p{i}": (_rising(lead_s) + rng.normal(0, 0.01, LENGTH), _gt()) for i in range(n)}
    curves |= {f"n{i}": (np.full(LENGTH, NEG_LEVEL) + rng.normal(0, 0.01, LENGTH),
                         np.zeros(LENGTH, dtype=np.int64)) for i in range(n)}
    return _write(tmp_path / name, curves)


def test_a_climb_before_the_alert_is_a_rise(tmp_path: Path) -> None:
    rows = [("early", _row(tmp_path, "early", 3.0)), ("flat", _row(tmp_path, "flat", 0.0))]
    report = po.build(rows, "flat", FPS)
    early, flat = report["rows"]["early"], report["rows"]["flat"]
    assert flat["rise"]["mean"] == pytest.approx(0.0, abs=0.02)
    # a 3 s linear climb: over [-2 s, 0) the mean is 2/3 of the way + half of the last third
    assert early["rise"]["mean"] == pytest.approx(2 / 3, abs=0.03)
    last = po._zone_key(*constants.PRE_ONSET_ZONES_S[-1])
    assert early["zone_auc"][last] > 0.99
    assert 0.3 < flat["zone_auc"][last] < 0.7       # noise only: like a video with no accident
    assert flat["zone_auc"][po.SPAN] > 0.99
    assert report["paired_rise"]["early"]["mean"] == pytest.approx(2 / 3, abs=0.03)
    first_span_bin = next(c for c in early["curve_scaled"] if c["from_s"] == 0.0)
    assert first_span_bin["ci"]["mean"] == pytest.approx(1.0, abs=0.05)   # the span's level
    text = po.render(report)
    assert "## 2. Zone AUC" in text and "decides nothing" in text


def test_rows_must_share_videos_and_labels(tmp_path: Path) -> None:
    a = _row(tmp_path, "a", 0.0)
    b = _row(tmp_path, "b", 0.0, n=5)
    with pytest.raises(ValueError, match="other videos"):
        po.build([("a", a), ("b", b)], "a", FPS)


def test_control_must_be_a_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="control"):
        po.build([("a", _row(tmp_path, "a", 0.0))], "K5f", FPS)


def test_main_writes_report_and_figure(tmp_path: Path) -> None:
    out = tmp_path / "out"
    po.main(["--row", f"K5f={_row(tmp_path, 'k', 1.0)}", "--out-dir", str(out), "--plots"])
    stem = constants.PRE_ONSET_REPORT_STEM
    assert (out / f"{stem}.md").is_file() and (out / f"{stem}.json").is_file()
    assert (out / po.FIGURE_NAME).read_bytes()[:4] == b"\x89PNG"


DOTA_STEP = 0.1  # s per row (DoTA native 10 fps)
PEAK = 1.0  # the climb ends above SPAN_LEVEL, so its last frames outrank the span
D_LEN, D_ONSET, D_END = 120, 70, 100  # 7 s before onset: in the fixed composition


def _clip_scores(path: Path, lead_s: float, n: int = 5) -> Path:
    """A ``clip_scores.npz`` of two-class clips (no negative video): far level, a climb that
    starts ``lead_s`` before onset and overshoots the span level, span level inside the span."""
    rng = np.random.default_rng(1)
    arrays = {}
    for i in range(n):
        s = np.full(D_LEN, NEG_LEVEL) + rng.normal(0, 0.01, D_LEN)
        start = D_ONSET - round(lead_s / DOTA_STEP)
        if lead_s > 0:
            s[start:D_ONSET] = np.linspace(NEG_LEVEL, PEAK, D_ONSET - start, endpoint=False)
        s[D_ONSET:D_END] = SPAN_LEVEL
        g = np.zeros(D_LEN)
        g[D_ONSET:D_END] = 1
        arrays[f"score/c{i}"], arrays[f"label/c{i}"] = s, g
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def test_clip_scores_set_without_negatives(tmp_path: Path) -> None:
    early = _clip_scores(tmp_path / "e" / "clip_scores.npz", 2.0)
    flat = _clip_scores(tmp_path / "f" / "clip_scores.npz", 0.0)
    report = po.build([("early", early), ("flat", flat)], "flat", 1 / DOTA_STEP, "dota")
    e, f = report["rows"]["early"], report["rows"]["flat"]
    assert all(v is None for v in e["zone_auc"].values()) and e["negative_mean"] is None
    zone = po._zone_key(-1.0, 0.0)
    assert e["fixed"]["videos"]["fixed"] == 5
    # a 2 s linear climb to PEAK: over [-1, 0) the mean is 3/4 of the way to PEAK
    want = 0.75 * (PEAK - NEG_LEVEL) / (SPAN_LEVEL - NEG_LEVEL)
    assert e["fixed"]["rescaled"][zone]["mean"] == pytest.approx(want, abs=0.03)
    assert f["fixed"]["rescaled"][zone]["mean"] == pytest.approx(0.0, abs=0.03)
    # dropping the climbing normal frames raises the early row's macro, not the flat one's
    assert e["fixed"]["macro_without_zone_minus_macro"]["mean"] > 0.01
    assert f["fixed"]["macro_without_zone_minus_macro"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert "## 4. Fixed composition" in po.render(report)


def test_main_names_non_nexar_files_by_set(tmp_path: Path) -> None:
    out = tmp_path / "out"
    src = _clip_scores(tmp_path / "k" / "clip_scores.npz", 1.0)
    po.main(["--set", "dada", "--row", f"K5f={src}", "--out-dir", str(out), "--plots"])
    stem, figure = po.report_names("dada")
    assert (out / f"{stem}.md").is_file() and (out / figure).is_file()
    assert stem.endswith("_dada")
