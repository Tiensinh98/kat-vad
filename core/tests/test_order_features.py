"""v2.2 order features (architecture §2-§4).

The silent failures: a "shuffle" that leaves originally adjacent steps side by side (it would keep
the velocity it claims to destroy), a twin whose neighbours are close in the original order, a twin
whose values differ from the true feature's (then it tests distribution, not order), κ that leaks a
clip-start spike, and κ that depends on a per-clip offset or scale (the fixed effects it must drop).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from core import constants
from core.tools import order_features as of

DIM = 16


def _walk(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, DIM)).cumsum(axis=0)


def _stats() -> of.OrderStats:
    train = [_walk(30, s) for s in range(6)]
    return of.fit_kappa_stats(of.fit_stats(train, dim=4), train)


class TestBlockShuffle:
    def test_valid_permutation_counts_match_a002464(self) -> None:
        counts = [len(of.valid_block_permutations(m)) for m in range(2, 10)]
        assert counts == [0, 0, 2, 14, 90, 646, 5242, 47622]

    @pytest.mark.parametrize(("n", "sizes"), [
        (3, []), (4, [4]), (5, [5]), (9, [9]), (10, [5, 5]), (14, [5, 9]), (23, [5, 5, 5, 8]),
    ])
    def test_block_sizes_follow_the_ragged_tail_rule(self, n: int, sizes: list[int]) -> None:
        assert of.block_sizes(n) == sizes

    @pytest.mark.parametrize("n", [4, 5, 7, 10, 14, 20, 33, 57])
    def test_no_originally_adjacent_steps_are_neighbours(self, n: int) -> None:
        perm = of.block_permutation(n, np.random.default_rng(n))
        assert sorted(perm) == list(range(n))
        assert all(abs(a - b) != 1 for a, b in itertools.pairwise(perm))

    def test_steps_stay_inside_their_block(self) -> None:
        perm = of.block_permutation(23, np.random.default_rng(1))
        offset = 0
        for size in of.block_sizes(23):
            assert sorted(perm[offset : offset + size]) == list(range(offset, offset + size))
            offset += size

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_short_sequences_are_left_unshuffled(self, n: int) -> None:
        assert list(of.block_permutation(n, np.random.default_rng(0))) == list(range(n))

    def test_the_sequence_seed_is_stable_and_tag_specific(self) -> None:
        a = of.block_permutation(20, of.sequence_rng("t", "clip"))
        b = of.block_permutation(20, of.sequence_rng("t", "clip"))
        assert np.array_equal(a, b)
        draws = {tuple(of.block_permutation(20, of.sequence_rng(tag, "clip"))) for tag in "abcdef"}
        assert len(draws) > 1


class TestTwin:
    @pytest.mark.parametrize("n", [12, 13, 40, 150])
    def test_neighbours_are_six_apart_from_twelve_steps(self, n: int) -> None:
        order, gap = of.far_ordering(n, np.random.default_rng(n))
        assert gap == constants.V22_TWIN_GAP
        assert sorted(order) == list(range(n))
        assert all(abs(a - b) >= gap for a, b in itertools.pairwise(order))

    @pytest.mark.parametrize("n", [10, 11])
    def test_below_twelve_the_largest_feasible_gap_is_used(self, n: int) -> None:
        _order, gap = of.far_ordering(n, np.random.default_rng(0))
        assert gap == 5

    def test_quantile_match_keeps_values_and_takes_the_twins_ranks(self) -> None:
        rng = np.random.default_rng(0)
        true, twin = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))
        matched = of.quantile_match(true, twin)
        for j in range(3):
            assert np.array_equal(np.sort(matched[:, j]), np.sort(true[:, j]))
            assert np.array_equal(np.argsort(matched[:, j]), np.argsort(twin[:, j]))

    def test_twin_holds_the_true_values_per_channel(self) -> None:
        stats, x = _stats(), _walk(40, 9)
        true = of.kappa(of.change_parts(x, stats))
        twin, gap = of.twin_rows(
            x, true, lambda z: of.kappa(of.change_parts(z, stats)), np.random.default_rng(3)
        )
        assert gap == constants.V22_TWIN_GAP
        assert np.allclose(np.sort(twin, axis=0), np.sort(true, axis=0))
        assert not np.allclose(twin, true)


class TestKappa:
    def test_width_is_two_pca_blocks_plus_three_summaries(self) -> None:
        stats = _stats()
        assert of.kappa(of.change_parts(_walk(25, 1), stats)).shape == (25, 2 * 4 + 3)

    def test_start_is_replicate_padded_without_a_spike(self) -> None:
        parts = of.change_parts(_walk(25, 2), _stats())
        assert np.allclose(parts["v_bar"][0], parts["v_bar"][1])
        assert np.allclose(parts["a_bar"][0], parts["a_bar"][2])
        assert np.allclose(parts["a_bar"][1], parts["a_bar"][2])
        assert np.allclose(parts["s"][0], parts["s"][1])
        assert np.allclose(parts["m5"][0], parts["s"][0])

    def test_per_clip_offset_and_scale_cancel(self) -> None:
        stats, x = _stats(), _walk(30, 4)
        base = of.kappa(of.change_parts(x, stats))
        assert np.allclose(of.kappa(of.change_parts(x + 7.0, stats)), base)
        assert np.allclose(of.kappa(of.change_parts(3.0 * x, stats)), base)

    def test_m5_is_the_causal_max_over_five_steps(self) -> None:
        parts = of.change_parts(_walk(20, 5), _stats())
        s = parts["s"][:, 0]
        for t in range(20):
            assert parts["m5"][t, 0] == pytest.approx(s[max(0, t - 4) : t + 1].max())

    def test_fewer_than_three_steps_have_no_kappa(self) -> None:
        with pytest.raises(ValueError, match="needs >= 3"):
            of.change_parts(_walk(2, 0), _stats())

    def test_standardisation_is_fitted_on_train(self) -> None:
        train = [_walk(30, s) for s in range(6)]
        stats = of.fit_kappa_stats(of.fit_stats(train, dim=4), train)
        rows = np.concatenate([of.standardise(of.kappa(of.change_parts(x, stats)), stats)
                               for x in train])
        assert np.allclose(rows.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(rows.std(axis=0), 1.0)

    def test_standardise_without_kappa_stats_raises(self) -> None:
        stats = of.fit_stats([_walk(30, 0), _walk(30, 1)], dim=4)
        with pytest.raises(ValueError, match="no κ statistics"):
            of.standardise(np.zeros((3, 11)), stats)
