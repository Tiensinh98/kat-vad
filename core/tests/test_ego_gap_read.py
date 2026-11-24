"""Tests for ``core.tools.ego_gap_read`` (KIM_RUN §46)."""

from __future__ import annotations

import numpy as np
import pytest

from core.tools import ego_gap_read as eg


def _diff(**kw: tuple[float, float]) -> dict[str, dict[str, float]]:
    base = {"G": (-0.2, -0.1), "G_A": (-0.2, -0.1), "low_uninvolved": (-0.2, -0.1)} | kw
    return {k: {"mean": sum(v) / 2, "low": v[0], "high": v[1]} for k, v in base.items()}


def _row(low_u: float, low_i: float, n: int = 200) -> eg.Row:
    low = np.array([float(k < low_u * n / 2) for k in range(n // 2)]
                   + [float(k < low_i * n / 2) for k in range(n // 2)])
    return {"low": low, "A": 1.0 - low, "auc": np.full(n, 0.8)}


INVOLVED = np.array([False] * 100 + [True] * 100)
DRAWS = [np.random.default_rng(k).integers(0, 200, 200) for k in range(200)]


def test_gaps_and_signs() -> None:
    g = eg.gaps(_row(0.4, 0.1), INVOLVED)
    assert g["G"] == pytest.approx(0.3) and g["G_A"] == pytest.approx(0.3)


def test_paired_difference_is_signed_t2_minus_nexar() -> None:
    d = eg.paired(_row(0.4, 0.1), _row(0.1, 0.1), INVOLVED, DRAWS)
    assert d["G"]["mean"] == pytest.approx(-0.3)
    assert d["G"]["high"] < 0 and d["low_uninvolved"]["high"] < 0
    assert eg.verdict(d) == eg.SUPPORT


def test_identical_rows_show_no_shrink() -> None:
    d = eg.paired(_row(0.4, 0.1), _row(0.4, 0.1), INVOLVED, DRAWS)
    assert d["G"]["mean"] == 0 and eg.verdict(d) == eg.NO_SHRINK_EXCLUDED


@pytest.mark.parametrize(("diff", "want"), [
    (_diff(), eg.SUPPORT),
    (_diff(low_uninvolved=(-0.1, 0.02)), eg.OTHER_SIDE),
    (_diff(G_A=(-0.1, 0.02)), eg.MIXED),
    (_diff(G=(0.01, 0.1)), eg.WIDENED),
    (_diff(G=(-0.05, 0.05)), eg.NO_SHRINK_EXCLUDED),
    (_diff(G=(-0.20, 0.05)), eg.NO_SHRINK_OPEN),
])
def test_verdicts(diff: dict[str, dict[str, float]], want: str) -> None:
    assert eg.verdict(diff) == want


def test_gate_refuses_a_row_that_does_not_reproduce_the_reference() -> None:
    low = np.array([1.0] * 50 + [0.0] * 150)
    with pytest.raises(ValueError, match="EG2 gate"):
        eg.gate_reference("K5f", low, INVOLVED)
    eg.gate_reference("K5fT2", low, INVOLVED)
