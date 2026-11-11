"""Tests for ``core.tools.pos_neg_gap``: the pooling, the floor, the gate and the CLI.

The silent failures: a ``-logit`` pool that is really ``+logit``, a floor that reuses frames
across its two halves, a decision flipped by an empty N-thr, or ratio arithmetic against the
wrong mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import constants
from core.tools import pos_neg_gap as pg

TAU = constants.CONTRASTIVE_TEMP
DIM = 8
LENGTH = 40
ABNORMAL = slice(15, 25)


def separated_video(distinct: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, logits, labels. ``distinct``: the abnormal frames lie along another axis."""
    rng = np.random.default_rng(0)
    feats = np.tile(np.eye(DIM)[0], (LENGTH, 1)) + 0.05 * rng.standard_normal((LENGTH, DIM))
    labels = np.zeros(LENGTH, dtype=np.int64)
    labels[ABNORMAL] = 1
    if distinct:
        feats[ABNORMAL] = np.eye(DIM)[1] + 0.05 * rng.standard_normal((10, DIM))
    logits = np.where(labels == 1, 3.0, -3.0) + 0.1 * rng.standard_normal(LENGTH)
    return feats, logits, labels


def test_attention_weights_sign_and_support() -> None:
    logits = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.array([True, True, False, True])
    plus = pg.attention_weights(logits, 1.0, mask, TAU)
    minus = pg.attention_weights(logits, -1.0, mask, TAU)
    assert plus.sum() == pytest.approx(1.0) and minus.sum() == pytest.approx(1.0)
    assert plus[2] == 0.0 and minus[2] == 0.0
    assert plus.argmax() == 3 and minus.argmax() == 0  # +logit picks the top, -logit the bottom


def test_attention_over_empty_set_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        pg.attention_weights(np.zeros(3), 1.0, np.zeros(3, dtype=bool), TAU)


def test_cosine_distance_basics() -> None:
    a = np.array([1.0, 0.0])
    assert pg.cosine_distance(a, a) == pytest.approx(0.0)
    assert pg.cosine_distance(a, np.array([0.0, 1.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero vector"):
        pg.cosine_distance(a, np.zeros(2))


def test_separated_video_has_gap_far_above_floor() -> None:
    feats, logits, labels = separated_video(distinct=True)
    rec = pg.video_metrics(feats, logits, labels, TAU, 10, np.random.default_rng(1))
    assert isinstance(rec, dict) and rec["gap_label"] is not None and rec["floor"] is not None
    assert rec["gap_label"] > 0.9 and rec["floor"] < 0.05
    assert rec["pos_mass_abnormal"] == pytest.approx(1.0, abs=1e-6)
    assert rec["thr_precision"] == pytest.approx(1.0) and rec["thr_recall"] == pytest.approx(1.0)


def test_indistinct_video_gap_is_at_floor() -> None:
    feats, logits, labels = separated_video(distinct=False)
    rec = pg.video_metrics(feats, logits, labels, TAU, 10, np.random.default_rng(1))
    assert isinstance(rec, dict) and rec["gap_label"] is not None and rec["floor"] is not None
    assert rec["gap_label"] < 0.05 and rec["floor"] < 0.05


def test_skips_are_named() -> None:
    feats, logits, _ = separated_video(distinct=True)
    rng = np.random.default_rng(0)
    no_abnormal = np.zeros(LENGTH, int)
    assert pg.video_metrics(feats, logits, no_abnormal, TAU, 3, rng) == pg.SKIP_NO_ABNORMAL
    one_normal = np.ones(LENGTH, int)
    one_normal[0] = 0
    assert pg.video_metrics(feats, logits, one_normal, TAU, 3, rng) == pg.SKIP_NO_NORMAL


def test_length_mismatch_raises() -> None:
    feats, logits, labels = separated_video(distinct=True)
    with pytest.raises(ValueError, match="lengths differ"):
        pg.video_metrics(feats, logits, labels[:-1], TAU, 3, np.random.default_rng(0))


def test_empty_thr_is_recorded_not_fatal() -> None:
    feats, _, labels = separated_video(distinct=True)
    logits = np.full(LENGTH, 4.0)  # every sigmoid above 0.5
    rec = pg.video_metrics(feats, logits, labels, TAU, 3, np.random.default_rng(0))
    assert isinstance(rec, dict) and rec["gap_thr"] is None and rec["gap_label"] is not None


def record(gap: float, floor: float) -> dict[str, float | None]:
    rec: dict[str, float | None] = dict.fromkeys(pg.METRICS, 0.5)
    rec |= {"gap_label": gap, "floor": floor}
    return rec


@pytest.mark.parametrize(
    ("gap", "floor", "decision"),
    [(0.10, 0.10, pg.STOP), (0.149, 0.10, pg.STOP),
     (0.151, 0.10, pg.CONTINUE), (0.9, 0.1, pg.CONTINUE)],
)
def test_decision_uses_the_means_and_the_bar(gap: float, floor: float, decision: str) -> None:
    recs = {f"v{i}": record(gap, floor) for i in range(5)}
    out = pg.summarize(recs, {}, seed=0)
    assert out["decision"] == decision
    assert out["ratio_gap_label_over_floor"] == pytest.approx(gap / floor)


def test_no_videos_gives_no_decision() -> None:
    out = pg.summarize({}, {"v0": pg.SKIP_NO_NORMAL}, seed=0)
    assert out["decision"] is None and out["ratio_gap_label_over_floor"] is None


def test_parser_requires_checkpoint_and_out_dir() -> None:
    with pytest.raises(SystemExit):
        pg.build_arg_parser().parse_args(["--name", "K5"])
