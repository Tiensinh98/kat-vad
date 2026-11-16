"""``core.tools.feature_diversity`` (``KIM_RUN.md`` §28.2): the pure math and the verdict."""

from __future__ import annotations

import numpy as np
import pytest

from core import constants
from core.tools import feature_diversity as fd

DIM = 16


def _video(spread: float, offset: float, n_norm: int = 30, n_abn: int = 10,
           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Normal rows around e0, abnormal rows around e0 + offset * e1, isotropic noise."""
    rng = np.random.default_rng(seed)
    centre_n = np.eye(DIM)[0] * 3.0
    centre_a = centre_n + offset * np.eye(DIM)[1]
    feats = np.vstack([centre_n + spread * rng.standard_normal((n_norm, DIM)),
                       centre_a + spread * rng.standard_normal((n_abn, DIM))])
    labels = np.r_[np.zeros(n_norm, int), np.ones(n_abn, int)]
    return feats, labels


def test_row_cos_matches_the_pairwise_mean() -> None:
    z = fd.normalize_rows(np.random.default_rng(1).standard_normal((7, DIM)))
    gram = z @ z.T
    expected = (gram.sum() - np.trace(gram)) / (7 * 6)
    assert fd.row_cos(z) == pytest.approx(expected)
    assert fd.row_cos(fd.normalize_rows(np.ones((5, DIM)))) == pytest.approx(1.0)


def test_eff_rank_counts_directions() -> None:
    assert fd.eff_rank(np.eye(DIM)[:4]) == pytest.approx(3.0)  # 4 points span 3 centred dims
    line = np.outer(np.arange(6.0), np.eye(DIM)[0])
    assert fd.eff_rank(line) == pytest.approx(1.0)


def test_separation_grows_with_the_offset_and_is_scale_free() -> None:
    near, labels = _video(spread=0.3, offset=0.2)
    far, _ = _video(spread=0.3, offset=3.0)
    s_near = fd.separation(fd.normalize_rows(near), labels == 1)
    s_far = fd.separation(fd.normalize_rows(far), labels == 1)
    assert s_far > 10 * s_near
    assert fd.separation(fd.normalize_rows(10 * far), labels == 1) == pytest.approx(s_far)


def test_collapsed_video_reads_as_predicted() -> None:
    """The §28.1 hypothesis in miniature: shrink the spread and the offset together."""
    base = fd.video_metrics(*_video(spread=0.5, offset=3.0))
    collapsed = fd.video_metrics(*_video(spread=0.1, offset=0.1))
    assert isinstance(base, dict) and isinstance(collapsed, dict)
    assert collapsed["row_cos"] > base["row_cos"] and collapsed["sep"] < base["sep"]


def test_too_few_rows_are_skipped_and_lengths_checked() -> None:
    feats, labels = _video(spread=0.3, offset=1.0, n_abn=1)
    assert fd.video_metrics(feats, labels) == fd.SKIP_FEW_ROWS
    with pytest.raises(ValueError, match="lengths differ"):
        fd.video_metrics(feats, labels[:-1])
    with pytest.raises(ValueError, match="all zero"):
        fd.normalize_rows(np.zeros((3, DIM)))


def _records(spreads: list[float], offsets: list[float]) -> dict[str, dict[str, float]]:
    out = {}
    for k, (s, o) in enumerate(zip(spreads, offsets, strict=True)):
        result = fd.video_metrics(*_video(spread=s, offset=o, seed=k))
        assert isinstance(result, dict)
        out[f"{k:05d}"] = result
    return out


def test_verdict_supported_opposite_and_not_supported() -> None:
    n = 12
    sharp = _records([0.5] * n, [3.0] * n)
    blurred = _records([0.1] * n, [0.1] * n)
    assert fd.paired_summary(sharp, blurred, seed=0)["verdict"] == constants.FEATDIV_SUPPORTED
    assert fd.paired_summary(blurred, sharp, seed=0)["verdict"] == constants.FEATDIV_OPPOSITE
    same = fd.paired_summary(sharp, sharp, seed=0)
    assert same["verdict"] == constants.FEATDIV_NOT_SUPPORTED
    assert same["metrics"]["sep"]["delta"] == 0.0


def test_paired_summary_refuses_mismatched_videos() -> None:
    a = _records([0.5, 0.5], [3.0, 3.0])
    with pytest.raises(ValueError, match="different videos"):
        fd.paired_summary(a, {"00000": a["00000"]}, seed=0)


def test_parser_requires_both_checkpoints() -> None:
    with pytest.raises(SystemExit):
        fd.build_arg_parser().parse_args(["--base-name", "K5", "--base-ckpt", "a",
                                          "--census", "c", "--clip-s1-dir", "x",
                                          "--data-dir", "d", "--out-dir", "o"])
