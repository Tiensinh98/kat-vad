"""MM-AU Phase 0 provenance matcher (plan .project/plans/katvad-mmau-phase0.md §3-§4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.data.v2_splits import load_split
from core.tools import mmau_match

DIM = 64  # random unit vectors in 64-d have |cos| ~ 0.1, far below NEAR


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    return mmau_match.normalize_rows(rng.normal(size=(n, DIM)))


def _at_cosine(rows: np.ndarray, cos: float, rng: np.random.Generator) -> np.ndarray:
    """Rows whose cosine to ``rows`` is exactly ``cos`` (a unit component orthogonal to each)."""
    noise = rng.normal(size=rows.shape)
    noise -= (noise * rows).sum(axis=1, keepdims=True) * rows
    noise = mmau_match.normalize_rows(noise)
    mixed: np.ndarray = cos * rows + np.sqrt(1.0 - cos**2) * noise
    return mixed


class TestContainment:
    def test_subsampled_clip_is_exact_with_rate_and_offset(self) -> None:
        rng = np.random.default_rng(0)
        cap = {f"c{i}": _unit(rng, 60) for i in range(6)}
        query = {"q": cap["c3"][5::3][:15]}
        (m,) = mmau_match.match_queries(query, cap)
        assert (m.cap_id, m.grade) == ("c3", mmau_match.GRADE_EXACT)
        assert m.kappa == pytest.approx(1.0, abs=1e-5)
        assert m.rate == pytest.approx(3.0) and m.offset == pytest.approx(5.0)
        assert m.null_kappa < constants.MMAU_MATCH_NEAR
        assert m.hits == ["c3"]

    def test_reprocessed_frames_grade_near(self) -> None:
        rng = np.random.default_rng(1)
        cap = {f"c{i}": _unit(rng, 40) for i in range(6)}
        query = {"q": _at_cosine(cap["c2"][:20], 0.97, rng)}
        (m,) = mmau_match.match_queries(query, cap)
        assert (m.cap_id, m.grade) == ("c2", mmau_match.GRADE_NEAR)
        assert m.kappa == pytest.approx(0.97, abs=1e-4)
        assert m.rate == pytest.approx(1.0)

    def test_unrelated_clip_grades_none_without_alignment(self) -> None:
        rng = np.random.default_rng(2)
        cap = {f"c{i}": _unit(rng, 40) for i in range(6)}
        (m,) = mmau_match.match_queries({"q": _unit(rng, 20)}, cap)
        assert m.grade == mmau_match.GRADE_NONE
        assert m.rate is None and m.hits == []

    def test_alignment_needs_two_distinct_frames(self) -> None:
        best = np.array([0.99, 0.1, 0.1])
        assert mmau_match.fit_alignment(best, np.array([4, 0, 0])) is None


class TestBranch:
    @staticmethod
    def _block(exact: float, near_or_exact: float, null: float = 0.0) -> dict[str, float]:
        return {
            "exact_coverage": exact,
            "near_or_exact_coverage": near_or_exact,
            "null_at_near_share": null,
        }

    @pytest.mark.parametrize(
        ("all_", "dev", "partial", "expected"),
        [
            ((1.0, 1.0), (1.0, 1.0), True, mmau_match.BRANCH_PARTIAL),
            ((0.96, 0.99), (0.95, 0.99), False, mmau_match.BRANCH_P),
            ((0.96, 0.99), (0.90, 0.99), False, mmau_match.BRANCH_P_PRIME),
            ((0.50, 0.60), (0.50, 0.60), False, mmau_match.BRANCH_C),
        ],
    )
    def test_rule(
        self,
        all_: tuple[float, float],
        dev: tuple[float, float],
        partial: bool,
        expected: str,
    ) -> None:
        got = mmau_match.decide_branch(self._block(*all_), self._block(*dev), partial)
        assert got == expected

    def test_null_flag_blocks_every_branch(self) -> None:
        good = self._block(1.0, 1.0)
        assert (
            mmau_match.decide_branch(good, self._block(1.0, 1.0, null=0.02), partial=False)
            == mmau_match.BRANCH_UNRELIABLE
        )


class TestCli:
    def test_readout_counts_coverage_and_clean(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        dev_ids = load_split(constants.V2_SPLIT_DOTA_DEV)[:3]
        cap = {f"{i:06d}": _unit(rng, 50) for i in range(8)}
        dota = {
            dev_ids[0]: cap["000000"][::3],  # exact, dev
            dev_ids[1]: cap["000001"][2::2],  # exact, dev
            dev_ids[2]: _unit(rng, 12),  # none, dev
            "not_in_dev": cap["000002"][::4],  # exact, outside dev
        }
        dada = {"t01_v001": cap["000003"][::8]}
        dirs = {name: tmp_path / name for name in ("cap", "dota", "dada")}
        for name, clips in (("cap", cap), ("dota", dota), ("dada", dada)):
            dirs[name].mkdir()
            for vid, rows in clips.items():
                np.save(dirs[name] / f"{vid}.npy", rows.astype(np.float32))
        out = tmp_path / "out"
        base = [
            "--cap-clip-dir", str(dirs["cap"]), "--dota-clip-dir", str(dirs["dota"]),
            "--dada-clip-dir", str(dirs["dada"]), "--out-dir", str(out),
        ]
        mmau_match.main([*base, "--expected-cap", "8"])
        readout = json.loads((out / mmau_match.READOUT_JSON).read_text())
        assert readout["dota_all"]["counts"] == {"exact": 3, "near": 0, "none": 1}
        assert readout["dota_dev"]["n"] == 3
        assert readout["dota_dev"]["exact_coverage"] == pytest.approx(2 / 3)
        assert readout["dada"]["counts"]["exact"] == 1
        assert readout["cap"] | {"clean_frames": 0} == {
            "clips": 8, "expected": 8, "partial": False, "hit_by_dota": 3,
            "hit_by_dada": 1, "clean": 4, "clean_frames": 0,
        }
        assert readout["branch"] == mmau_match.BRANCH_C
        matches = json.loads((out / mmau_match.MATCHES_JSON).read_text())
        by_query = {m["query"]: m for m in matches["dota"]}
        assert by_query[dev_ids[0]]["rate"] == pytest.approx(3.0)
        assert "Branch (plan §4): **C**" in (out / mmau_match.READOUT_MD).read_text()

        mmau_match.main(base)  # default expected = the full CAP count
        partial = json.loads((out / mmau_match.READOUT_JSON).read_text())
        assert partial["branch"] == mmau_match.BRANCH_PARTIAL
