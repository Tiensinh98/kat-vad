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


__all__ = ["CapContrastLoss", "asymmetric_infonce_loss"]
