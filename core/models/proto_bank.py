"""Class prototypes on ``v^t`` in the main score, KIM off (rule ``core/docs/final/KIM_RUN.md`` §61).

Two buffers, ``mu_0`` (normal) and ``mu_1`` (abnormal), with no gradient, kept by a momentum
update in the training loop and frozen at inference (they live in the state dict). With the score
term on (PR2) the bin logits gain::

    term_t = lambda * (cos(v_t, mu_1) - cos(v_t, mu_0))   on valid steps, 0 until both exist

``lambda`` is one scalar, initialised to 0, so the row starts equal to its host. Without the score
term (row K0dfiL) the bank only feeds the normal-anchored loss and has no parameter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants

NORMAL, ABNORMAL = 0, 1


class PrototypeBank(nn.Module):
    """``mu_0`` / ``mu_1`` buffers on ``feat_dim`` features, plus ``lambda`` when ``score``."""

    def __init__(self, feat_dim: int, score: bool) -> None:
        super().__init__()
        self.register_buffer("prototypes", torch.zeros(constants.PROTO_CLASSES, feat_dim))
        self.register_buffer("proto_init", torch.zeros(constants.PROTO_CLASSES, dtype=torch.bool))
        self.lam: nn.Parameter | None = (
            nn.Parameter(torch.full((), constants.PROTO_LAMBDA_INIT)) if score else None
        )

    @property
    def ready(self) -> bool:
        return bool(self.proto_init.all())

    def score_term(self, feats: Tensor, mask: Tensor) -> Tensor:
        """``(B, L)`` term added to the bin logits; zeros before both prototypes exist."""
        if self.lam is None:
            raise RuntimeError("this bank was built without the score term")
        if not self.ready:
            return feats.new_zeros(feats.shape[:2])
        protos = self.prototypes.to(feats.dtype)
        cos_abn = F.cosine_similarity(feats, protos[ABNORMAL].view(1, 1, -1), dim=-1)
        cos_nor = F.cosine_similarity(feats, protos[NORMAL].view(1, 1, -1), dim=-1)
        term: Tensor = self.lam.to(feats.dtype) * (cos_abn - cos_nor) * mask.to(feats.dtype)
        return term

    @torch.no_grad()
    def update(self, feats: Tensor, normals: Tensor, anchors: Tensor, momentum: float) -> None:
        """Momentum update: ``mu_0`` from ``normals``, ``mu_1`` from ``anchors`` (``(B, L)`` bool).

        A class absent from the batch keeps its prototype; its first appearance sets it.
        """
        flat = feats.detach().float()
        for cls, chosen in ((NORMAL, normals), (ABNORMAL, anchors)):
            picked = flat[chosen.bool()]
            if not picked.numel():
                continue
            mean = picked.mean(dim=0)
            if bool(self.proto_init[cls]):
                self.prototypes[cls].mul_(1.0 - momentum).add_(momentum * mean)
            else:
                self.prototypes[cls].copy_(mean)
                self.proto_init[cls] = True

    def cosine(self) -> Tensor:
        """``cos(mu_0, mu_1)``; 0 until both exist."""
        if not self.ready:
            zero: Tensor = self.prototypes.new_zeros(())
            return zero
        cos: Tensor = F.cosine_similarity(self.prototypes[NORMAL], self.prototypes[ABNORMAL], dim=0)
        return cos


def bottomk_steps(score: Tensor, labels: Tensor, lengths: Tensor, topk_pct: int) -> Tensor:
    """``(B, L)`` bool: the bottom-k valid steps of each abnormal bag by ``score`` (PR4).

    k = ``max(1, n_valid // topk_pct)``, the host's top-k count.
    """
    chosen = torch.zeros_like(score, dtype=torch.bool)
    detached = score.detach()
    for b in range(score.shape[0]):
        n = int(lengths[b])
        if n == 0 or labels[b] < 0.5:
            continue
        low = detached[b, :n].topk(max(1, n // topk_pct), largest=False).indices
        chosen[b, low] = True
    return chosen


__all__ = ["PrototypeBank", "bottomk_steps"]
