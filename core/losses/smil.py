"""v5 spatial MIL loss ``L_sMIL`` (PREREG_V5 §2.1 S2-6, proposal §3.2).

Per step the strongest query decides, ``S_t = max_m s_{t,m}``; per sequence the top-k steps
decide, ``s_hat = mean of the top-k valid S_t`` with ``k = max(floor(L / 16), 1)``;
``L_sMIL = BCE(s_hat, y)``. It sees every sequence the trunk's ``L_MIL`` sees (DVS-synthesized
ones included), with the same label and valid lengths.

The scores are probabilities, not logits; ``binary_cross_entropy`` is unsafe under autocast, so
``BCE(s_hat, y)`` is taken as ``binary_cross_entropy_with_logits(logit(s_hat, eps), y)`` in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from core import constants


def spatial_mil_loss(
    probs: Tensor,
    labels: Tensor,
    lengths: Tensor,
    topk_pct: int = constants.V5_SMIL_TOPK_PCT,
) -> Tensor:
    """``probs (B, L, M)`` in ``(0, 1)``, ``labels (B,)`` in {0, 1}, ``lengths (B,)`` -> scalar."""
    with torch.autocast(probs.device.type, enabled=False):
        step = probs.float().amax(dim=-1)
        pooled = []
        for b in range(step.shape[0]):
            n = int(lengths[b])
            k = max(n // topk_pct, 1)
            pooled.append(step[b, :n].topk(k).values.mean())
        logits = torch.logit(torch.stack(pooled), eps=constants.V5_SMIL_EPS)
        return F.binary_cross_entropy_with_logits(logits, labels.float())


__all__ = ["spatial_mil_loss"]
