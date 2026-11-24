"""core.tools.kim_ablation_report (KIM_RUN §53): contrasts, guards and verdicts. Data-free, CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.config import Config, validate_kim
from core.metrics import frame_auc
from core.tools import kim_ablation_report as ka
from core.tools.tad_clip_gate import micro_bootstrap_pairs
from core.tools.v5_stage1 import _flat

DRAWS = 40
SEED = 7


def _curves(seed: int, n: int = 12, length: int = 20) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(n):
        y = np.zeros(length, dtype=np.int64)
        if i % 3:
            y[8:14] = 1
        labels[f"v{i:02d}"] = y
        scores[f"v{i:02d}"] = np.round(rng.normal(size=length) + y, 1)  # rounded: has ties
    return scores, labels


def test_pooled_auc_equals_frame_auc_with_ties() -> None:
    scores, labels = _curves(1)
    p = ka.PooledAuc(scores, labels, {v: i for i, v in enumerate(sorted(labels))})
    ids = sorted(labels)
    want = frame_auc(np.concatenate([scores[v] for v in ids]),
                     np.concatenate([labels[v] for v in ids]))
    assert p.auc(np.ones(len(ids))) == pytest.approx(want, abs=1e-12)


def test_pooled_auc_is_nan_with_one_class() -> None:
    scores, labels = _curves(1)
    p = ka.PooledAuc(scores, labels, {v: i for i, v in enumerate(sorted(labels))})
    counts = np.zeros(len(labels))
    counts[0] = 1  # v00 has no positive frame
    assert np.isnan(p.auc(counts))


def test_micro_contrast_equals_the_tables_bootstrap() -> None:
    a, labels = _curves(1)
    b, _ = _curves(2)
    groups = {v: f"g{i // 2}" for i, v in enumerate(sorted(labels))}  # two videos per cluster
    want = micro_bootstrap_pairs({"a": a, "b": b}, {"d": ("a", "b")}, labels, SEED,
                                 groups=groups, draws=DRAWS)["d"]
    got = ka.micro_contrasts({"a": a, "b": b}, labels, groups, {"d": {"a": 1, "b": -1}}, DRAWS,
                             SEED)["d"]
    for key in ("mean", "low", "high"):
        assert got[key] == pytest.approx(want[key], abs=1e-12)


def test_difference_of_differences_terms() -> None:
    terms = ka.contrast_terms(["K5f", "K5dfi"])
    assert terms["K5f"] == {"K5f:on": 1, "K5f:off": -1}
    assert terms["K5dfi vs K5f"] == {"K5dfi:on": 1, "K5dfi:off": -1, "K5f:on": -1, "K5f:off": 1}
    assert "K5f vs K5f" not in terms


def test_macro_contrasts_are_paired_per_item() -> None:
    items = {"A:on": {"x": 0.9, "y": 0.8}, "A:off": {"x": 0.7, "y": 0.7},
             "B:on": {"x": 0.6, "y": 0.6}, "B:off": {"x": 0.6, "y": 0.5}}
    out = ka.macro_contrasts(items, {"x": "x", "y": "y"}, ["A", "B"])
    a, b = out["A"], out["B vs A"]
    assert a is not None and b is not None
    assert a["mean"] == pytest.approx(0.15)
    assert b["mean"] == pytest.approx(0.05 - 0.15)


def _labels(**kw: str) -> dict[str, str]:
    return {**dict.fromkeys(ka.ALL_SETS, ka.NO_DIFFERENCE), **kw}


@pytest.mark.parametrize("labels,want", [
    (_labels(), ka.NO_DIFFERENCE),
    (_labels(dada=ka.GAIN), ka.BENEFITS),
    (_labels(dada=ka.COST), ka.HURT),
    (_labels(dada=ka.GAIN, nexar=ka.COST), ka.MIXED),
    (_labels(tad=ka.GAIN), ka.NO_DIFFERENCE),  # TAD is printed, never decides
])
def test_metric_verdict(labels: dict[str, str], want: str) -> None:
    assert ka.metric_verdict(labels) == want


def test_headline_flags_a_micro_only_benefit() -> None:
    assert "micro only" in ka.headline(ka.BENEFITS, ka.HURT)
    assert "micro only" not in ka.headline(ka.BENEFITS, ka.BENEFITS)


def _cfg(**flags: bool) -> dict:
    cfg = _flat(Config().to_dict())
    return {**cfg, **{f"kim.{k}": v for k, v in flags.items()}}


def test_kim_only_accepts_the_on_rows_flags() -> None:
    on = _cfg(enabled=True, prototypes=True, edge_mask=True)
    assert ka.check_kim_only(on, _cfg(), "K5f") == ["kim.edge_mask", "kim.enabled",
                                                    "kim.prototypes"]


def test_kim_only_refuses_a_second_factor_and_a_kim_off_on_row() -> None:
    on = _cfg(enabled=True, prototypes=True, edge_mask=True)
    with pytest.raises(ValueError, match="config diff"):
        ka.check_kim_only(on, {**_cfg(), "v2.motion": "other"}, "K5f")
    with pytest.raises(ValueError, match=r"kim\.enabled=false"):
        ka.check_kim_only(_cfg(), _cfg(), "K5f")
    with pytest.raises(ValueError, match="config diff"):
        ka.check_kim_only(on, _cfg(enabled=True, prototypes=True), "K5f")


def test_config_diff_reads_a_missing_key_at_its_default() -> None:
    base = _flat(Config().to_dict())
    old = {k: v for k, v in base.items() if k != "loss.normal_caption"}
    assert ka.config_diff(old, base) == {}


def test_off_config_is_one_the_code_accepts() -> None:
    cfg = Config()
    assert not (cfg.kim.enabled or cfg.kim.prototypes or cfg.kim.edge_mask or cfg.kim.seam_mask)
    validate_kim(cfg)


def test_edge_row_that_starts_like_k0f_is_refused() -> None:
    same = {"K5f": {"new": 0.6878}, "K5fepre": {"new": 0.6878}}
    with pytest.raises(ValueError, match="forced_normals"):
        ka.check_edge_read(same)
    ka.check_edge_read({"K5f": {"new": 0.6878}, "K5fepre": {"new": 0.6973}})


def test_rerun_replaces_the_saved_on_row(tmp_path: Path) -> None:
    arm = ka.arm_paths("K5fepre", tmp_path)
    assert arm.on_row == "K5fepre" and arm.on_root == tmp_path / "kim_edge" / "K5fepre"
    assert arm.off_root == tmp_path / constants.KIM_ABLATION_OUTPUT_DIRNAME / "K0fepre"
    rerun = (tmp_path / constants.KIM_ABLATION_OUTPUT_DIRNAME
             / f"K5fepre{constants.KIM_ABLATION_RERUN_SUFFIX}")
    rerun.mkdir(parents=True)
    assert ka.arm_paths("K5fepre", tmp_path).on_root == rerun


def test_six_arms_have_distinct_off_rows() -> None:
    off = [row for row, _ in constants.KIM_ABLATION_ARMS.values()]
    assert len(off) == len(set(off)) == 6
    assert next(iter(constants.KIM_ABLATION_ARMS)) == constants.KIM_REFERENCE_ARM
