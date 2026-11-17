"""``ClassContrastLoss`` / ``class_infonce_loss``: the class form of ``L_neg`` (KIM_RUN.md §28)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from core import constants
from core.losses.contrastive import (
    CONTRAST_TYPE_VANILLA,
    ClassContrastLoss,
    asymmetric_infonce_loss,
    caption_rows,
    class_infonce_loss,
    near_duplicates,
    normalize_caption,
    uniq_rows,
    unique_target,
)

DIM = 8
LENGTH = 6


def _batch(cls_idx: list[int], seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    batch = len(cls_idx)
    logits = torch.randn(batch, LENGTH, generator=gen) * 3.0  # prob range well above 0.2
    return {
        "logits": logits,
        "lengths": torch.full((batch,), LENGTH),
        "v_feats": torch.randn(batch, LENGTH, DIM, generator=gen, requires_grad=True),
        "class_feats": torch.randn(3, DIM, generator=gen),
        "cls_idx": torch.tensor(cls_idx),
        "pseudo_frame_label": torch.zeros(batch, LENGTH),
    }


def _call(loss: ClassContrastLoss, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = loss(b["logits"], b["lengths"], b["v_feats"],
                                        b["class_feats"], b["cls_idx"], b["pseudo_frame_label"])
    return out


class TestClassInfonce:
    def test_one_positive_per_row_equals_one_hot(self) -> None:
        sim = torch.randn(3, 5)
        label = torch.tensor([0, 2, 4])
        loss, _ = class_infonce_loss(sim, torch.arange(3), torch.tensor([0, 9, 1, 9, 2]))
        ref = F.cross_entropy(sim, label, label_smoothing=constants.CONTRASTIVE_LABEL_SMOOTHING)
        torch.testing.assert_close(loss, ref)

    def test_same_class_column_is_pulled_not_pushed(self) -> None:
        """The gradient raises a same-class column; the baseline lowers it as a negative."""
        classes = torch.tensor([1, 1, 0])
        sim = torch.zeros(1, 3, requires_grad=True)
        class_infonce_loss(sim, torch.tensor([1]), classes)[0].backward()
        assert sim.grad is not None and float(sim.grad[0, 1]) < 0.0
        base = torch.zeros(1, 3, requires_grad=True)
        asymmetric_infonce_loss(base, torch.tensor([0])).backward()
        assert base.grad is not None and float(base.grad[0, 1]) > 0.0

    def test_floor_is_the_minimum(self) -> None:
        """Soft-target CE is minimised at logits = log(smoothed target): there it is the floor."""
        classes = torch.tensor([1, 1, 0, 0])
        loss, floor = class_infonce_loss(torch.randn(1, 4), torch.tensor([1]), classes)
        eps = constants.CONTRASTIVE_LABEL_SMOOTHING
        target = (1.0 - eps) * torch.tensor([[0.5, 0.5, 0.0, 0.0]]) + eps / 4
        best, _ = class_infonce_loss(target.log(), torch.tensor([1]), classes)
        assert float(floor) <= float(loss)
        assert float(best) == pytest.approx(float(floor), abs=1e-5)

    def test_row_without_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="no positive"):
            class_infonce_loss(torch.zeros(1, 2), torch.tensor([3]), torch.tensor([0, 1]))


class TestClassContrastLoss:
    def test_one_class_mines_without_pseudo_labels(self) -> None:
        """DVS off (pseudo all zero): every abnormal video still yields a mined column."""
        b = _batch([1, 1, 0, 0])
        out = _call(ClassContrastLoss(), b)
        assert float(out["mined"]) == 2.0
        assert torch.isfinite(out["loss"])
        assert float(out["floor"]) <= float(out["loss"])

    def test_no_zero_column_when_scores_are_flat(self) -> None:
        b = _batch([1, 0])
        b["logits"] = torch.zeros_like(b["logits"])  # prob range 0 < N3_MIN_SCORE_RANGE
        assert float(_call(ClassContrastLoss(), b)["mined"]) == 0.0

    def test_vanilla_mines_nothing(self) -> None:
        out = _call(ClassContrastLoss(mining=CONTRAST_TYPE_VANILLA), _batch([1, 1, 0]))
        assert float(out["mined"]) == 0.0

    def test_synthesized_row_mines_inside_its_anchor(self) -> None:
        """With an anchor span, only its rows feed the mined column (baseline n3)."""
        b = _batch([1, 0])
        b["pseudo_frame_label"][0, :2] = 1.0
        mined = ClassContrastLoss()._mine(b["logits"], b["lengths"], b["v_feats"],
                                         b["cls_idx"], b["pseudo_frame_label"])
        weights = torch.softmax(-b["logits"][0, :2] / constants.CONTRASTIVE_TEMP, dim=0)
        torch.testing.assert_close(mined[0], weights @ b["v_feats"][0, :2])

    def test_scale_free(self) -> None:
        """Cosine similarity: rescaling features leaves the loss unchanged (C37)."""
        b = _batch([1, 2, 0, 1])
        ref = _call(ClassContrastLoss(), b)["loss"]
        b["class_feats"] = b["class_feats"] * 40.0
        b["v_feats"] = (b["v_feats"] * 40.0).detach()
        torch.testing.assert_close(_call(ClassContrastLoss(), b)["loss"], ref)

    def test_one_class_is_text_to_video_only(self) -> None:
        """Nexar has one abnormal class: the loss is exactly the text -> video term."""
        b = _batch([1, 1, 0])
        out = _call(ClassContrastLoss(mining=CONTRAST_TYPE_VANILLA), b)
        cols = (torch.softmax(b["logits"] / constants.CONTRASTIVE_TEMP, dim=1)[:, None]
                @ b["v_feats"]).squeeze(1)
        sim = (F.normalize(b["class_feats"][1:2], dim=-1) @ F.normalize(cols, dim=-1).T
               / constants.LNEG_CLASS_TEMP)
        ref, _ = class_infonce_loss(sim, torch.tensor([1]), b["cls_idx"])
        torch.testing.assert_close(out["loss"], ref)

    def test_two_classes_add_video_to_text(self) -> None:
        two = _call(ClassContrastLoss(mining=CONTRAST_TYPE_VANILLA), _batch([1, 2, 0]))
        assert torch.isfinite(two["loss"]) and float(two["floor"]) <= float(two["loss"])

    def test_gradient_reaches_video_features(self) -> None:
        b = _batch([1, 1, 0, 0])
        _call(ClassContrastLoss(), b)["loss"].backward()
        grad = b["v_feats"].grad
        assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0

    def test_no_abnormal_video_raises(self) -> None:
        with pytest.raises(ValueError, match="no abnormal"):
            _call(ClassContrastLoss(), _batch([0, 0]))

    def test_unknown_mining_raises(self) -> None:
        with pytest.raises(ValueError, match="contrast_type"):
            ClassContrastLoss(mining="n1")


class TestUniqTargets:
    """§28.3: one fixed meaningless unit target per source video."""

    def test_target_is_fixed_unit_and_key_specific(self) -> None:
        a = unique_target("00123", DIM)
        torch.testing.assert_close(a, unique_target("00123", DIM))
        assert float(a.norm()) == pytest.approx(1.0)
        assert not torch.allclose(a, unique_target("00124", DIM))
        assert not torch.allclose(a, unique_target("00123", DIM, seed=1))

    def test_target_leaves_the_global_rng_alone(self) -> None:
        torch.manual_seed(0)
        before = torch.get_rng_state()
        unique_target("00123", DIM)
        assert torch.equal(before, torch.get_rng_state())

    def test_rows_group_windows_of_one_source(self) -> None:
        cls_idx = torch.tensor([1, 1, 0, 1])
        targets, idx = uniq_rows(cls_idx, ["v1", "v1", "v1", "v2"], DIM)
        assert idx.tolist() == [1, 1, 0, 2]  # the normal window of v1 is no row
        assert targets.shape == (3, DIM) and float(targets[0].abs().sum()) == 0.0
        torch.testing.assert_close(targets[2], unique_target("v2", DIM))
        with pytest.raises(ValueError, match="source ids"):
            uniq_rows(cls_idx, ["v1"], DIM)

    def test_other_sources_are_negatives(self) -> None:
        """Two sources: each row's only positive is its own source's column."""
        b = _batch([1, 1, 0])
        targets, idx = uniq_rows(b["cls_idx"], ["v1", "v2", "v3"], DIM)
        loss = ClassContrastLoss(mining=CONTRAST_TYPE_VANILLA)
        out = loss(b["logits"], b["lengths"], b["v_feats"], targets, idx,
                   b["pseudo_frame_label"])
        same = ClassContrastLoss(mining=CONTRAST_TYPE_VANILLA)(
            b["logits"], b["lengths"], b["v_feats"], targets[[0, 1, 1]],
            torch.tensor([1, 1, 0]), b["pseudo_frame_label"])
        assert torch.isfinite(out["loss"]) and not torch.isclose(out["loss"], same["loss"])


CAPS = {"a": "Collision: a van scrapes the door.", "b": "collision: a van scrapes the door",
        "c": "Accident: a truck is rear-ended."}


class TestCaptionRows:
    """§30 / §30.1: per-caption rows, merged identical captions, masked near-duplicates."""

    def test_rows_group_one_source_and_identical_captions(self) -> None:
        texts, idx = caption_rows(torch.tensor([1, 1, 0, 1, 1]), ["a", "a", "x", "b", "c"],
                                  CAPS)
        assert texts == [CAPS["a"], CAPS["c"]]
        assert idx.tolist() == [1, 1, 0, 1, 2]  # a and b normalize to one caption

    def test_normalize(self) -> None:
        assert normalize_caption("  Collision:  A van,  scrapes!") == "collision a van scrapes"

    def test_missing_caption_raises(self) -> None:
        with pytest.raises(KeyError, match="no caption"):
            caption_rows(torch.tensor([1]), ["zzz"], CAPS)

    def test_normal_items_need_no_caption(self) -> None:
        texts, idx = caption_rows(torch.tensor([0, 0]), ["zzz", "yyy"], CAPS)
        assert texts == [] and idx.tolist() == [0, 0]

    def test_near_duplicates_skip_normal_row_and_diagonal(self) -> None:
        feats = torch.tensor([[9.0, 9.0], [1.0, 0.0], [1.0, 0.01], [0.0, 1.0]])
        dup = near_duplicates(feats)
        assert dup.tolist() == [[False, False, False, False], [False, False, True, False],
                                [False, True, False, False], [False, False, False, False]]


class TestMaskedInfonce:
    def test_no_ignored_column_equals_cross_entropy(self) -> None:
        sim = torch.randn(3, 7, generator=torch.Generator().manual_seed(1))
        rows, cols = torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 0, 0, 1, 2])
        ref, ref_floor = class_infonce_loss(sim, rows, cols)
        none = torch.zeros(3, 7, dtype=torch.bool)
        got, got_floor = class_infonce_loss(sim, rows, cols, ignore=none)
        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(got_floor, ref_floor)

    def test_ignored_column_does_not_move_the_loss(self) -> None:
        sim = torch.randn(2, 4, generator=torch.Generator().manual_seed(2))
        rows, cols = torch.tensor([1, 2]), torch.tensor([1, 2, 0, 0])
        ignore = torch.tensor([[False, True, False, False], [True, False, False, False]])
        a, _ = class_infonce_loss(sim, rows, cols, ignore=ignore)
        moved = sim.clone()
        moved[0, 1], moved[1, 0] = 50.0, -50.0
        b, _ = class_infonce_loss(moved, rows, cols, ignore=ignore)
        torch.testing.assert_close(a, b)

    def test_ignoring_a_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="also ignored"):
            class_infonce_loss(torch.zeros(1, 2), torch.tensor([1]), torch.tensor([1, 0]),
                               ignore=torch.tensor([[True, False]]))

    def test_dup_masks_the_other_rows_columns(self) -> None:
        b = _batch([1, 2, 0, 1])
        b["class_feats"] = torch.randn(3, DIM)
        dup = torch.zeros(3, 3, dtype=torch.bool)
        dup[1, 2] = dup[2, 1] = True
        loss = ClassContrastLoss(cosine_temp=constants.LNEG_CAPTION_TEMP)
        out = loss(b["logits"], b["lengths"], b["v_feats"], b["class_feats"], b["cls_idx"],
                   b["pseudo_frame_label"], dup=dup)
        assert float(out["masked_share"]) == 1.0  # every other-row abnormal column is masked
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        plain = _call(loss, b)
        assert float(plain["masked_share"]) == 0.0
