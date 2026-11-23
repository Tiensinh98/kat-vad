"""``core.tools.nexar_eda``: the rebuilt labels equal the corpus's, and the tables read them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import constants
from core.tools import nexar_build, nexar_eda

DURATION = 40.0
FRAMES = 1200  # 150 rows at s8


def _row(label: int, alert: float | None = None, event: float | None = None) -> dict[str, Any]:
    return {
        "label": label, "duration_s": DURATION, "time_of_alert": alert, "time_of_event": event,
        "issues": [], "fps": 30.0, "meta_scene": "Urban" if label else "Highway",
        "meta_weather": "Clear", "frames_header": FRAMES,
    }


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    rows = {
        "p1": _row(1, 18.0, 19.5), "p2": _row(1, 20.0, 21.0), "pv": _row(1, 19.0, 20.0),
        "n1": _row(0), "nv": _row(0),
    }
    nexar_build.build_window(tmp_path, rows, ["n1", "p1", "p2"], ["nv", "pv"],
                             dict.fromkeys(rows, FRAMES))
    return tmp_path


def test_rebuilt_labels_equal_the_corpus_meta(corpus: Path) -> None:
    labels, meta, _ = nexar_eda.load_corpus(corpus)
    rows, mismatched = nexar_eda.window_rows(labels, meta, constants.NEXAR_TRAIN_STRIDE)
    assert mismatched == [] and len(rows) == len(labels)
    assert all(r["label"] == int(r["positive"] >= 1) for r in rows)
    assert {r["source"] for r in rows} == {"p1", "p2"}


def test_wrong_stride_is_refused(corpus: Path) -> None:
    with pytest.raises(ValueError, match="rebuilt positive rows differ"):
        nexar_eda.build_report(corpus, constants.NEXAR_TRAIN_STRIDE * 2)


def test_report_reads_geometry_and_renders(corpus: Path) -> None:
    report = nexar_eda.build_report(corpus, constants.NEXAR_TRAIN_STRIDE)
    counts = report["corpus"]
    assert counts["train_windows"] == counts["train_positive"] + counts["train_normal"]
    assert counts["train_sources"] == 2 and counts["sources_without_a_positive_window"] == 0
    # p1's span is [18.0, 20.9] s = 2.9 s ~ 10-11 rows at s8 / 30 fps: windows are 20 rows
    assert 1 <= report["positive_windows"]["abnormal_rows"]["percentiles"]["p50"] <= 11
    assert len(report["positive_windows"]["row_profile"]) == constants.NEXAR_WINDOW_LENGTH
    normal = report["normal_windows"]
    assert normal["normal_windows_with_a_span"] == normal["before_span"] + normal["after_span"]
    assert 0.0 <= report["rulers"]["center_hump"] <= 1.0
    assert {r["scene"] for r in report["scene"]} == {"Urban"}      # normals share their video
    text = nexar_eda.render_markdown(report)
    assert "## 6. Val windows" in text and "AUC over train windows" in text
    json.dumps(report)


def test_main_writes_both_files(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    nexar_eda.main(["--data-dir", str(corpus), "--output-dir", str(out)])
    stem = constants.NEXAR_EDA_REPORT_STEM
    assert (out / f"{stem}.md").is_file() and (out / f"{stem}.json").is_file()


def test_plots_write_six_figures(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    nexar_eda.main(["--data-dir", str(corpus), "--output-dir", str(out), "--plots"])
    figures = sorted((out / constants.NEXAR_EDA_FIGURES_DIRNAME).glob("*.png"))
    assert [f.name for f in figures] == sorted([
        "abnormal_rows_per_positive_window.png", "row_profile.png", "normal_gap_to_span.png",
        "window_center_by_class.png", "scene_weather_positive_rate.png", "val_frame_share.png"])
    assert all(f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for f in figures)


def test_no_plots_without_the_flag(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    nexar_eda.main(["--data-dir", str(corpus), "--output-dir", str(out)])
    assert not (out / constants.NEXAR_EDA_FIGURES_DIRNAME).exists()
