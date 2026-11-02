"""LaGoVAD-STP first-run diagnostics (``core.tools.stp_diag``, rules ``STP_BUILD.md`` §5)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

from core import constants
from core.tools import stp_diag
from core.train import cosine_warmup_lambda

LR, WARMUP, STEPS = 5e-3, 5, 200


def _train(grad_fn: object, steps: int = STEPS, frozen: bool = True) -> tuple[
        torch.nn.Sequential, dict[str, torch.Tensor], dict[str, Any]]:
    """A toy model trained by AdamW the way Trainer builds it; returns (model, init, payload)."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 1))
    if frozen:
        for p in model.get_submodule("0").parameters():
            p.requires_grad = False
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=LR, weight_decay=0.0)
    sched = LambdaLR(opt, cosine_warmup_lambda(WARMUP, steps))
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        for p in trainable:
            p.grad = grad_fn(p, gen)  # type: ignore[operator]
        opt.step()
        sched.step()
    return model, init, {"optimizer": opt.state_dict(), "global_step": steps}


def _coherent(p: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.ones_like(p)


def _noise(p: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(p.shape, generator=gen)


def _dead(p: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.full_like(p, 1e-12)


def _row(grad_fn: object) -> dict[str, Any]:
    model, init, payload = _train(grad_fn)
    names = stp_diag.trainable_names(model)
    opt = stp_diag.optimizer_state(payload, names)
    budget = stp_diag.lr_budget(LR, WARMUP, STEPS, STEPS)
    row = stp_diag.merged_stats(names, model.state_dict(), init, opt, budget)
    assert row is not None
    return row


class TestBudget:
    def test_lr_budget_is_the_scheduler_sum(self) -> None:
        p = torch.nn.Parameter(torch.zeros(1))
        opt = AdamW([p], lr=LR)
        sched = LambdaLR(opt, cosine_warmup_lambda(WARMUP, STEPS))
        total = 0.0
        for _ in range(STEPS):
            total += opt.param_groups[0]["lr"]
            opt.step()
            sched.step()
        assert stp_diag.lr_budget(LR, WARMUP, STEPS, STEPS) == pytest.approx(total)

    def test_noise_snr_matches_adam_emas_of_noise(self) -> None:
        b1, b2 = 0.9, 0.999
        rng = np.random.default_rng(0)
        g = rng.standard_normal((5000, 2000))
        m = np.zeros(2000)
        v = np.zeros(2000)
        for t in range(len(g)):
            m = b1 * m + (1 - b1) * g[t]
            v = b2 * v + (1 - b2) * g[t] ** 2
        assert np.abs(m).mean() / np.sqrt(v).mean() == pytest.approx(
            stp_diag.noise_snr(b1), rel=0.05)


class TestOptimizerState:
    def test_names_follow_trainer_order_and_skip_frozen(self) -> None:
        model, _, payload = _train(_coherent, steps=3)
        names = stp_diag.trainable_names(model)
        assert names == ["1.weight", "1.bias"]
        opt = stp_diag.optimizer_state(payload, names)
        head = model.get_submodule("1")
        assert opt["1.weight"]["exp_avg"].shape == head.weight.shape
        assert opt["1.bias"]["exp_avg"].shape == head.bias.shape

    def test_mismatched_architecture_raises(self) -> None:
        model, _, payload = _train(_coherent, steps=3)
        with pytest.raises(ValueError, match="does not match"):
            stp_diag.optimizer_state(payload, [*stp_diag.trainable_names(model), "extra"])


class TestParamStats:
    def test_coherent_gradient_uses_the_budget(self) -> None:
        row = _row(_coherent)
        assert row["snr"] == pytest.approx(1.0, abs=0.01)
        assert row["budget_used"] == pytest.approx(1.0, abs=0.02)
        assert row["share_below_10eps"] == 0.0

    def test_noise_gradient_is_incoherent(self) -> None:
        row = _row(_noise)
        assert row["snr"] < constants.STP_DIAG_NOISE_FACTOR * row["noise_snr"]
        assert row["budget_used"] < constants.STP_DIAG_MOVED * 2

    def test_dead_gradient_sits_below_eps(self) -> None:
        row = _row(_dead)
        assert row["share_below_10eps"] == 1.0
        assert row["budget_used"] < 1e-2

    def test_zero_init_has_no_cosine(self) -> None:
        final = torch.ones(3)
        state = {"group": {"betas": (0.9, 0.999), "eps": 1e-8}}
        row = stp_diag.param_stats(final, torch.zeros(3), state, 1.0)
        assert row["cos_final_init"] is None
        assert row["adam_state"] is False
        assert row["displacement"] == pytest.approx(math.sqrt(3))


class TestInitFingerprint:
    def test_first_logged_prompt_norm(self, tmp_path: Path) -> None:
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text("\n".join(json.dumps(r) for r in (
            {"step": 0, "mil": 1.0}, {"step": 0, "prompt_norm": 1.25}, {"prompt_norm": 1.3})))
        assert stp_diag.logged_init_norm(metrics) == 1.25
        assert stp_diag.logged_init_norm(tmp_path / "absent.jsonl") is None

    def test_tolerance(self) -> None:
        assert stp_diag.init_reproduced(1.2612254, 1.2612255) is True
        assert stp_diag.init_reproduced(1.27, 1.2612255) is False
        assert stp_diag.init_reproduced(1.0, None) is None


class TestClassify:
    BASE: ClassVar[dict[str, Any]] = {
        "adam_state": True, "init_reproduced": True, "share_below_10eps": 0.0,
        "snr": 0.9, "noise_snr": 0.184, "budget_used": 0.8,
    }

    @pytest.mark.parametrize(("change", "prefix"), [
        ({}, "D1-moved"),
        ({"budget_used": 0.01}, "D1-lr"),
        ({"snr": 0.2, "budget_used": 0.01}, "D1-noise"),
        ({"share_below_10eps": 0.9}, "D1-dead"),
        ({"init_reproduced": False}, "INIT NOT REPRODUCED"),
    ])
    def test_readings(self, change: dict[str, float], prefix: str) -> None:
        assert stp_diag.classify({**self.BASE, **change}).startswith(prefix)

    def test_missing_row(self) -> None:
        assert stp_diag.classify(None).startswith("no prompt")


def _box_tiles(box: list[float]) -> np.ndarray:
    return np.flatnonzero(stp_diag.tile_weights([box], constants.STP_GRID) > 0)


BOX = [0.0, 0.0, 300.0, 200.0]  # top-left corner of a 1280 x 720 frame


def _clip(
    rows: int = 20, start: int = 30
) -> tuple[np.ndarray, np.ndarray, dict[int, list[list[float]]]]:
    steps = np.arange(rows) * constants.STP_DOTA_STRIDE
    labels = np.zeros(rows * constants.STP_DOTA_STRIDE, dtype=np.int64)
    labels[start:] = 1
    boxes = {int(f): [BOX] for f in steps if f >= start}
    return steps, labels, boxes


class TestSelectionRead:
    def test_planted_selection_points(self) -> None:
        steps, labels, boxes = _clip()
        hit = _box_tiles(BOX)
        rng = np.random.default_rng(0)
        index = np.stack([rng.permutation(constants.STP_PATCHES)[:constants.STP_TOP_K]
                          for _ in steps])
        anom = steps >= 30
        index[anom, : len(hit)] = hit
        read = stp_diag.selection_read(index, steps, labels, boxes)
        assert read.lift is not None and read.lift > 0.2
        assert read.lift_vs_pre is not None and read.lift_vs_pre > 0.2

    def test_static_selection_on_the_box_is_not_pointing(self) -> None:
        steps, labels, boxes = _clip()
        hit = _box_tiles(BOX)
        rest = np.setdiff1d(np.arange(constants.STP_PATCHES), hit)
        row = np.concatenate([hit, rest])[: constants.STP_TOP_K]
        index = np.tile(row, (len(steps), 1))
        read = stp_diag.selection_read(index, steps, labels, boxes)
        assert read.lift is not None and read.lift > 0
        assert read.lift_vs_pre == pytest.approx(0.0)

    def test_regions(self) -> None:
        steps, labels, boxes = _clip()
        border = np.flatnonzero(stp_diag.BORDER)[: constants.STP_TOP_K]
        index = np.tile(border, (len(steps), 1))
        read = stp_diag.selection_read(index, steps, labels, boxes)
        assert read.regions["border"] == pytest.approx(1 - 24 / 49)
        assert stp_diag.BORDER.sum() == 24
        assert stp_diag.BOTTOM.sum() == 7
        assert stp_diag.CENTRE.sum() == 9

    def test_too_few_anomaly_rows(self) -> None:
        steps, labels, boxes = _clip(rows=20, start=59)
        index = np.zeros((len(steps), constants.STP_TOP_K), dtype=np.int64)
        read = stp_diag.selection_read(index, steps, labels, boxes)
        assert read.lift is None and read.lift_vs_pre is None


class TestVariantReads:
    def test_end_to_end_and_render(self) -> None:
        rng = np.random.default_rng(0)
        clips, mo, index = {}, {}, {}
        for i in range(6):
            v = f"src{i % 3}_{i:06d}"
            steps, labels, boxes = _clip()
            m = rng.random((len(steps), constants.STP_PATCHES))
            m[steps >= 30][:, _box_tiles(BOX)] += 0.0
            m[np.ix_(steps >= 30, _box_tiles(BOX))] += 2.0
            mo[v] = m
            index[v] = select_top = np.argsort(-m, axis=1, kind="stable")[:, : constants.STP_TOP_K]
            assert select_top.shape == (len(steps), constants.STP_TOP_K)
            clips[v] = stp_diag.ClipInputs(labels, steps, boxes, ego=bool(i % 2))
        out = stp_diag.variant_reads(clips, mo, index)
        assert out["clips"] == 6
        assert out["topk_lift"]["all"]["mean"] > 0.2
        assert out["topk_lift_vs_pre"]["all"]["mean"] > 0.2
        assert out["mo"]["hit_minus_chance"]["mean"] > 0
        md = stp_diag.render_pointing({"rule": stp_diag.RULE, "ego_count": {"ego": 3, "non_ego": 3},
                                       "variants": {"raw": out}})
        assert "| raw | non_ego |" in md

    def test_render_prompt(self) -> None:
        row = _row(_coherent)
        out = {"rule": stp_diag.RULE, "runs": {"R6": {
            "rows": {"prompt": row, "sa2": None}, "text_prompt": "prefix", "global_step": STEPS,
            "lr_budget": 1.0, "reading": stp_diag.classify(row)}}}
        md = stp_diag.render_prompt(out)
        assert "| R6 | prompt |" in md and "D1-moved" in md
