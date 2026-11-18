"""``core.tools.k5_a7_report`` (``KIM_RUN.md`` §33): R4 verdict and the R5 same-batch guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import constants
from core.tools import k5_a7_report as a7
from core.tools.zs_lagovad_readout import ABOVE, BELOW, NOT_SEPARATED

SETS = constants.KIM_LNEG_CAP_SETS


def _map(**sentences: str) -> dict[str, str]:
    out: dict[str, str] = dict.fromkeys((*SETS, "tad"), NOT_SEPARATED)
    out.update(sentences)
    return out


@pytest.mark.parametrize(("vs_ref", "candidate"), [
    (_map(dada=ABOVE), True),
    (_map(dada=ABOVE, dad=BELOW), False),
    (_map(), False),
    (_map(tad=ABOVE), False),  # TAD never decides
])
def test_candidate_needs_one_above_and_none_below(vs_ref: dict[str, str],
                                                  candidate: bool) -> None:
    got = a7.verdict(_map(), vs_ref)
    assert got["candidate"] is candidate
    assert got["line"] == (a7.LINE_CANDIDATE if candidate else a7.LINE_CLOSED)


def test_primary_contrast_decides_nothing() -> None:
    got = a7.verdict(_map(nexar=ABOVE, dota=ABOVE), _map())
    assert got["candidate"] is False and got["line"] == a7.LINE_CLOSED
    assert got["normal_caption_above"] == ["nexar", "dota"]


def _metrics(run: Path, values: list[float]) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    rows = [{"epoch": 0, "global_step": i + 1, a7.GUARD_KEY: v} for i, v in enumerate(values)]
    rows.append({"epoch": 0, "global_step": len(values) + 1, "mil": 0.5})  # no guard key
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def test_same_batches_pass(tmp_path: Path) -> None:
    a = _metrics(tmp_path / "a", [2.0, 3.0, 1.0])
    b = _metrics(tmp_path / "b", [2.0, 3.0, 1.0])
    assert a7.check_same_batches(a, b) == 3


def test_different_batches_raise(tmp_path: Path) -> None:
    a = _metrics(tmp_path / "a", [2.0, 3.0, 1.0])
    b = _metrics(tmp_path / "b", [2.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="differs at 1 steps"):
        a7.check_same_batches(a, b)
    c = _metrics(tmp_path / "c", [2.0, 3.0])
    with pytest.raises(ValueError, match="steps differ"):
        a7.check_same_batches(a, c)


def test_rows_and_sentence() -> None:
    assert a7.ROWS == ("K5f", "K5cf", "K5cfn")
    assert constants.NORMAL_PROBE_DASHCAM_DEFS.index(constants.NORMAL_CAPTION_DASHCAM_TEXT) == 3
    assert "surveillance" not in constants.NORMAL_CAPTION_DASHCAM_TEXT


def test_parser_requires_the_roots() -> None:
    with pytest.raises(SystemExit):
        a7.build_arg_parser().parse_args(["--kim-root", "a", "--out-dir", "o"])
