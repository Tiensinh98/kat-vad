"""``L_neg`` — caption contrastive loss with hard-negative mining (n3).

Re-implemented from the LaGoVAD baseline (``CapContrastLoss`` +
``asymmetric_infonce_loss``). The module is stateless w.r.t. parameters and is
instantiated once in the model/training setup (not per step, unlike the
baseline training loop).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants

CONTRAST_TYPE_VANILLA = "vanilla"
CONTRAST_TYPE_N3 = "n3"


def asymmetric_infonce_loss(
    sim: Tensor,
    label: Tensor,
    label_smoothing: float = constants.CONTRASTIVE_LABEL_SMOOTHING,
) -> Tensor:
    """Bidirectional InfoNCE on an asymmetric similarity matrix.

    ``sim (S, B')``: S anomaly captions vs B' aggregated video features;
    ``label (S,)``: index of each caption's own video among the B' columns.
    """
    loss_t2v = F.cross_entropy(sim, label, label_smoothing=label_smoothing)
    ano_sim = sim[:, label]  # (S, S)
    loss_v2t = F.cross_entropy(
        ano_sim,
        torch.arange(ano_sim.shape[0], device=ano_sim.device),
        label_smoothing=label_smoothing,
    )
    return (loss_t2v + loss_v2t) / 2


class CapContrastLoss(nn.Module):
    """Anomaly-caption vs. attention-aggregated video-feature contrastive loss.

    ``contrast_type='n3'`` additionally mines the *normal* portion of abnormal
    videos (per pseudo frame labels) as extra negative video features.
    """

    def __init__(
        self,
        contrast_type: str = CONTRAST_TYPE_N3,
        temperature: float = constants.CONTRASTIVE_TEMP,
    ) -> None:
        super().__init__()
        if contrast_type not in (CONTRAST_TYPE_VANILLA, CONTRAST_TYPE_N3):
            raise ValueError(f"Unknown contrast_type: {contrast_type!r}")
        self.contrast_type = contrast_type
        self.temperature = temperature

    def forward(
        self,
        logits: Tensor,
        lengths: Tensor,
        v_feats: Tensor,
        t_feats: Tensor,
        cls_label_idx: Tensor,
        pseudo_frame_label: Tensor,
    ) -> Tensor:
        """See baseline: ``logits (B, T)`` pre-sigmoid, ``t_feats (S, E)`` excl. Normal.

        ``cls_label_idx (B,)`` with 0 = normal; ``pseudo_frame_label (B, T)``.
        """
        max_len = logits.shape[1]
        batch = logits.shape[0]
        device = logits.device
        ano_indices = torch.where(cls_label_idx != 0)[0]  # (S,)

        # attention-aggregate each video's features under its anomaly curve
        pad = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
        masked_logits = logits + torch.where(pad, constants.NEG_INF_MASK_VALUE, 0.0)
        attn = torch.softmax(masked_logits / self.temperature, dim=1).unsqueeze(1)
        agg_v_feats = (attn @ v_feats).squeeze(1)  # (B, E)

        if self.contrast_type == CONTRAST_TYPE_N3:
            mined = []
            for b in range(batch):
                if cls_label_idx[b] == 0:
                    continue
                ano_logit = logits[b, : int(lengths[b])]
                ano_prob = ano_logit.sigmoid()
                # skip videos scored (nearly) uniformly — no separable normal part
                if ano_prob.max() - ano_prob.min() < constants.N3_MIN_SCORE_RANGE:
                    continue
                # baseline selects `pseudo_frame_label != 0` frames and weights the
                # lowest-scoring ones (softmax over negated logits) — kept verbatim
                mined_part = pseudo_frame_label[b, : int(lengths[b])] != 0
                mining_logit = ano_logit[mined_part]
                mining_attn = torch.softmax(-mining_logit / self.temperature, dim=0)
                mined.append(mining_attn @ v_feats[b][pseudo_frame_label[b] != 0])
            if mined:
                agg_v_feats = torch.cat([agg_v_feats, torch.stack(mined)], dim=0)

        sim_mat = torch.einsum("se,be->sb", t_feats, agg_v_feats)
        return asymmetric_infonce_loss(sim_mat, ano_indices)


def class_infonce_loss(
    sim: Tensor,
    row_class: Tensor,
    col_class: Tensor,
    label_smoothing: float = constants.CONTRASTIVE_LABEL_SMOOTHING,
) -> tuple[Tensor, Tensor]:
    """Soft-target InfoNCE whose positives are every column of the row's class.

    ``sim (K, N)``, ``row_class (K,)``, ``col_class (N,)``. Each row's target is uniform
    over its class's columns (the "L_out" form of supervised contrastive), so same-class
    columns are never negatives. With one positive per row it equals the one-hot loss.
    Returns ``(loss, floor)``: ``floor`` is the smoothed target's entropy, the lowest
    value the loss can reach, logged so the excess is readable (KIM_RUN.md §28 NL3).
    """
    pos = (row_class[:, None] == col_class[None, :]).to(sim.dtype)
    counts = pos.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("class_infonce_loss: a row has no positive column")
    target = pos / counts
    loss = F.cross_entropy(sim, target, label_smoothing=label_smoothing)
    smoothed = (1.0 - label_smoothing) * target + label_smoothing / sim.shape[1]
    floor = -(smoothed * smoothed.clamp_min(torch.finfo(sim.dtype).tiny).log()).sum(1).mean()
    return loss, floor.detach()


class ClassContrastLoss(nn.Module):
    """``L_neg`` with class positives (KIM_RUN.md §28 NL2); the baseline form stays above.

    Rows are the text features of the abnormal classes present in the batch. Columns are
    every video's features aggregated under its anomaly curve, plus one mined "normal
    part" per abnormal video (lowest-scoring rows: over the anchor span when synthesized,
    over all valid rows when not). Similarity is cosine / ``cosine_temp``.
    """

    def __init__(
        self,
        mining: str = CONTRAST_TYPE_N3,
        temperature: float = constants.CONTRASTIVE_TEMP,
        cosine_temp: float = constants.LNEG_CLASS_TEMP,
    ) -> None:
        super().__init__()
        if mining not in (CONTRAST_TYPE_VANILLA, CONTRAST_TYPE_N3):
            raise ValueError(f"Unknown contrast_type: {mining!r}")
        self.mining = mining
        self.temperature = temperature
        self.cosine_temp = cosine_temp

    def _mine(
        self, logits: Tensor, lengths: Tensor, v_feats: Tensor, cls_idx: Tensor, pseudo: Tensor
    ) -> list[Tensor]:
        mined = []
        for b in torch.where(cls_idx != 0)[0].tolist():
            n = int(lengths[b])
            logit = logits[b, :n]
            prob = logit.sigmoid()
            if n == 0 or prob.max() - prob.min() < constants.N3_MIN_SCORE_RANGE:
                continue
            part = pseudo[b, :n] != 0
            if not bool(part.any()):  # un-synthesized: no anchor span, mine the whole window
                part = torch.ones_like(part)
            weights = torch.softmax(-logit[part] / self.temperature, dim=0)
            mined.append(weights @ v_feats[b, :n][part])
        return mined

    def forward(
        self,
        logits: Tensor,
        lengths: Tensor,
        v_feats: Tensor,
        class_feats: Tensor,
        cls_idx: Tensor,
        pseudo_frame_label: Tensor,
    ) -> dict[str, Tensor]:
        """``logits (B, T)`` pre-sigmoid, ``v_feats (B, T, E)``, ``class_feats (C, E)`` with
        row 0 = Normal, ``cls_idx (B,)`` with 0 = normal, ``pseudo_frame_label (B, T)``.

        Returns ``loss``, ``floor`` and ``mined`` (number of mined columns, detached).
        """
        present = torch.unique(cls_idx[cls_idx != 0])
        if present.numel() == 0:
            raise ValueError("ClassContrastLoss: the batch has no abnormal video")
        max_len = logits.shape[1]
        pad = torch.arange(max_len, device=logits.device)[None, :] >= lengths[:, None]
        masked = logits + torch.where(pad, constants.NEG_INF_MASK_VALUE, 0.0)
        attn = torch.softmax(masked / self.temperature, dim=1).unsqueeze(1)
        cols = (attn @ v_feats).squeeze(1)  # (B, E)
        col_class = cls_idx
        mined = (self._mine(logits, lengths, v_feats, cls_idx, pseudo_frame_label)
                 if self.mining == CONTRAST_TYPE_N3 else [])
        if mined:
            cols = torch.cat([cols, torch.stack(mined)], dim=0)
            col_class = torch.cat([cls_idx, cls_idx.new_zeros(len(mined))])
        rows = F.normalize(class_feats[present], dim=-1)
        sim = rows @ F.normalize(cols, dim=-1).T / self.cosine_temp  # (K, N)
        loss, floor = class_infonce_loss(sim, present, col_class)
        if present.numel() > 1:  # video -> text only when the texts can be told apart
            ano = torch.where(cls_idx != 0)[0]
            v2t, v2t_floor = class_infonce_loss(sim[:, ano].T, cls_idx[ano], present)
            loss, floor = (loss + v2t) / 2, (floor + v2t_floor) / 2
        return {"loss": loss, "floor": floor,
                "mined": torch.tensor(float(len(mined)), device=logits.device)}


__all__ = ["CapContrastLoss", "ClassContrastLoss", "asymmetric_infonce_loss",
           "class_infonce_loss"]
