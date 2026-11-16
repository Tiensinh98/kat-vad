"""``core.tools.k5_lneg_uniq_report`` (``KIM_RUN.md`` §28.3): the U2 verdict and the parser."""

from __future__ import annotations

import pytest

from core import constants
from core.tools import k5_lneg_uniq_report as ur
from core.tools.zs_lagovad_readout import ABOVE, BELOW, NOT_SEPARATED

SETS = constants.KIM_LNEG_UNIQ_SETS


def _map(**sentences: str) -> dict[str, str]:
    out = dict.fromkeys(SETS, NOT_SEPARATED)
    out.update(sentences)
    return out


def test_no_set_below_k5_is_a() -> None:
    got = ur.verdict(_map(nexar=ABOVE), _map(nexar=ABOVE, dada=ABOVE))
    assert got["verdict"] == constants.KIM_LNEG_UNIQ_A and got["below_k5"] == []


def test_still_below_and_not_recovered_is_b() -> None:
    below = dict.fromkeys(SETS, BELOW)
    got = ur.verdict(below, _map(dada=ABOVE))
    assert got["verdict"] == constants.KIM_LNEG_UNIQ_B and got["above_lneg"] == ["dada"]
    three = _map(nexar=BELOW, dota=BELOW, dada=BELOW)
    assert ur.verdict(three, _map())["verdict"] == constants.KIM_LNEG_UNIQ_B


@pytest.mark.parametrize(("vs_k5", "vs_lneg"), [
    (_map(nexar=BELOW), _map()),                                        # one set below
    (dict.fromkeys(SETS, BELOW), _map(nexar=ABOVE, dota=ABOVE)),        # cost partly recovered
])
def test_otherwise_mixed(vs_k5: dict[str, str], vs_lneg: dict[str, str]) -> None:
    assert ur.verdict(vs_k5, vs_lneg)["verdict"] == constants.KIM_LNEG_UNIQ_MIXED


def test_tad_does_not_decide() -> None:
    vs_k5 = {**_map(), "tad": BELOW}
    assert ur.verdict(vs_k5, _map())["verdict"] == constants.KIM_LNEG_UNIQ_A


def test_parser_requires_the_three_roots() -> None:
    with pytest.raises(SystemExit):
        ur.build_arg_parser().parse_args(["--kim-root", "a", "--out-dir", "o"])
