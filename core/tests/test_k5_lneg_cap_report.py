"""``core.tools.k5_lneg_cap_report`` (``KIM_RUN.md`` §30): C6 / C7 and the interaction."""

from __future__ import annotations

import pytest

from core import constants
from core.tools import k5_lneg_cap_report as cr
from core.tools.zs_lagovad_readout import ABOVE, BELOW, NOT_SEPARATED

SETS = constants.KIM_LNEG_CAP_SETS


def _map(**sentences: str) -> dict[str, str]:
    out = dict.fromkeys((*SETS, "tad"), NOT_SEPARATED)
    out.update(sentences)
    return out


def test_complementary_needs_two_above_and_none_below() -> None:
    got = cr.verdict(_map(nexar=ABOVE, dada=ABOVE), _map())
    assert got["verdict"] == constants.KIM_LNEG_CAP_COMPLEMENTARY
    assert cr.verdict(_map(nexar=ABOVE, dada=ABOVE, dad=BELOW), _map())["verdict"] == (
        constants.KIM_LNEG_CAP_NO_CLAIM)


def test_antagonistic_and_no_claim() -> None:
    assert cr.verdict(_map(dota=BELOW, dad=BELOW), _map())["verdict"] == (
        constants.KIM_LNEG_CAP_ANTAGONISTIC)
    assert cr.verdict(_map(nexar=ABOVE), _map())["verdict"] == constants.KIM_LNEG_CAP_NO_CLAIM


def test_tad_never_decides() -> None:
    assert cr.verdict(_map(tad=ABOVE, nexar=ABOVE), _map())["verdict"] == (
        constants.KIM_LNEG_CAP_NO_CLAIM)


@pytest.mark.parametrize(("both", "candidate"), [
    (_map(dada=ABOVE), True), (_map(dada=ABOVE, dota=BELOW), False), (_map(), False),
])
def test_k5dc_candidate(both: dict[str, str], candidate: bool) -> None:
    assert cr.verdict(_map(), both)["k5dc_candidate"] is candidate


def test_interaction_is_the_difference_of_differences() -> None:
    k5 = {f"v{i}": 0.70 for i in range(30)}
    items = {cr.K5: k5, cr.K5C: {v: a + 0.01 for v, a in k5.items()},
             cr.K5D: {v: a + 0.02 for v, a in k5.items()},
             cr.K5DC: {v: a + 0.08 for v, a in k5.items()}}
    got = cr.interaction(items, {})
    assert got["items"] == 30
    assert got["delta"]["mean"] == pytest.approx(0.05)  # (0.08 - 0.02) - (0.01 - 0.0)
    assert got["sentence"] == ABOVE


def test_interaction_uses_common_items_only() -> None:
    items = {r: {"a": 0.5, "b": 0.6} for r in (cr.K5, cr.K5C, cr.K5D)}
    items[cr.K5DC] = {"a": 0.5}
    assert cr.interaction(items, {})["items"] == 1
    items[cr.K5DC] = {"z": 0.5}
    with pytest.raises(ValueError, match="share no scored item"):
        cr.interaction(items, {})


def test_parser_requires_the_roots() -> None:
    with pytest.raises(SystemExit):
        cr.build_arg_parser().parse_args(["--kim-root", "a", "--out-dir", "o"])
