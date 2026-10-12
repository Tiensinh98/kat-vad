"""v2.2 temporal probe (architecture §5).

The silent failures: padding inside a scored training window (a position cue), a receptive field
other than ±4, a position feature that does not follow the step's original index, and seeds that
do not reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core import constants
from core.tools import order_probe as op
from core.tools.kill_switch_probe import position_features

CPU = torch.device("cpu")
C = constants.V22_PROBE_CONTEXT


def _seq(n: int, dim: int = 3, seed: int = 0) -> op.Sequence:
    rng = np.random.default_rng(seed)
    return op.Sequence(rng.normal(size=(n, dim)), np.arange(n), (np.arange(n) % 4 == 0).astype(int))


class TestRows:
    def test_cubic_is_the_harness_position_feature(self) -> None:
        n = 17
        assert np.allclose(op.cubic((np.arange(n) + 0.5) / n), position_features(n))

    def test_training_windows_carry_real_context(self) -> None:
        seq = _seq(40)
        rows = op.window_rows(seq, 10, 30, position=False)
        assert rows.shape == (20 + 2 * C, 3)
        assert np.array_equal(rows, seq.base[10 - C : 30 + C])

    def test_replicate_padding_only_at_true_ends(self) -> None:
        seq = _seq(24)
        rows = op.window_rows(seq, 0, 20, position=False)
        assert np.array_equal(rows[:C], np.repeat(seq.base[:1], C, 0))
        assert np.array_equal(rows[C + 20 :], seq.base[20 : 20 + C])

    def test_position_is_relative_to_the_window(self) -> None:
        rows = op.window_rows(_seq(40), 8, 28, position=True)
        assert np.allclose(rows[C : C + 20, 3:], position_features(20))

    def test_position_follows_the_original_index_under_a_shuffle(self) -> None:
        base = _seq(10)
        perm = np.array([1, 3, 0, 2, 4, 6, 8, 5, 9, 7])
        shuffled = op.Sequence(base.base[perm], perm, base.labels[perm])
        rows = op.sequence_rows(shuffled, position=True)
        assert np.allclose(rows[C : C + 10, 3:], position_features(10)[perm])


class TestProbe:
    def test_receptive_field_is_nine_steps(self) -> None:
        torch.manual_seed(0)
        probe = op.TemporalProbe(np.zeros(3), np.ones(3))
        rows = torch.randn(1, 30 + 2 * C, 3)
        base = probe(rows)
        bumped = rows.clone()
        bumped[0, 15 + C] += 5.0
        changed = set(torch.nonzero((probe(bumped) - base).abs()[0] > 0).flatten().tolist())
        assert base.shape == (1, 30)
        assert 15 in changed
        assert changed <= set(range(15 - C, 15 + C + 1))

    def test_a_planted_local_pattern_is_learned_and_seeds_reproduce(self) -> None:
        rng = np.random.default_rng(0)
        windows = rng.normal(size=(200, 20 + 2 * C, 3)).astype(np.float32)
        labels = (windows[:, C:-C, 0] > 0.5).astype(np.float32)
        a = op.fit_probe(windows, labels, seed=1, device=CPU)
        b = op.fit_probe(windows, labels, seed=1, device=CPU)
        rows = op.sequence_rows(op.Sequence(windows[0, C:-C].astype(np.float64), np.arange(20),
                                            labels[0]), position=False)
        sa, sb = op.score(a, rows, CPU), op.score(b, rows, CPU)
        assert np.array_equal(sa, sb)
        assert np.corrcoef(sa, windows[0, C:-C, 0])[0, 1] > 0.8

    def test_one_class_training_steps_raise(self) -> None:
        windows = np.zeros((4, 20 + 2 * C, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="both classes"):
            op.fit_probe(windows, np.zeros((4, 20), dtype=np.float32), seed=0, device=CPU)
