"""SA²-v2: a prior-biased, text-conditioned scorer over 49 local CLIP features of a frame.

Proposal ``core/docs/final/`` (LaGoVAD-STP-V2) §3.2.2; run rule ``SA2V2_RUN.md``. One input row is
``[c_t ; u_t ; P (49 * 512) ; D (49 * 512)]``: ``P`` are 7 x 7 pooled CLIP patch tokens and ``D`` is
their second-order change, computed per source video (``core.data.stp_patches.cell_columns``). The
priors are derived here, so the cache holds no derived tensors::

    u = W_cell [P ; P - c ; D] + e_pos + W_pi [z(Mo) ; z(rho) ; L(Mo) ; L(rho)],  H = Layer(u)
    l = beta . priors + gamma z(cos(P, tau_delta)) + w^T H
    q = sigma(l + b),  alpha = softmax(l / T),  x^AS = sum alpha P,  c~ = c + x^AS
    (SV3: c~ = c + tanh(g) x^AS, g = 0 at init; x^AS itself, which L_branch reads, is ungated)

Only valid rows pass through the encoder. ``tau_delta`` goes through CLIP's frozen text projection:
the host's text features are not in ``image_embeds``' space, where ``P`` lives (rule D3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants


@dataclass(frozen=True)
class SA2V2Output:
    """``c~``, ``x^AS (B, L, D)``; ``alpha``, ``l + b (B, L, 49)``; ``l``'s spread (scalar)."""

    features: Tensor
    x_as: Tensor
    attn: Tensor
    cell_logits: Tensor
    spread: Tensor


def zscore_cells(values: Tensor) -> Tensor:
    """Per-frame z-score across the cells (last dimension)."""
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, keepdim=True, unbiased=False)
    return (values - mean) / (std + constants.SV2_Z_EPS)


@torch.no_grad()
def cell_priors(
    cells: Tensor, change: Tensor, k: int = constants.SV2_RHO_K
) -> tuple[Tensor, Tensor]:
    """``(N, 49, D)`` cells and their change -> ``Mo`` and ``rho``, both ``(N, 49)``.

    ``rho`` = mean distance of a cell to its ``k`` nearest other cells of the same frame, on
    L2-normalised features.
    """
    unit = F.normalize(cells, dim=-1, eps=constants.STP_NORM_EPS)
    distance = (2.0 - 2.0 * unit @ unit.transpose(-1, -2)).clamp_min(0.0).sqrt()
    distance.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
    rho = distance.topk(k, dim=-1, largest=False).values.mean(dim=-1)
    return change.norm(dim=-1), rho


class SA2V2(nn.Module):
    """Patch encoder + scorer + the training-only head on ``x^AS`` (proposal §3.2.2-3.2.5)."""

    def __init__(self, feature_dim: int = constants.SV2_DIM, gate: bool = False) -> None:
        super().__init__()
        # SV3 (SA2V2_RUN.md §10): absent unless gated, so every SV2 checkpoint loads unchanged
        self.gate = nn.Parameter(torch.zeros(())) if gate else None
        self.feature_dim = feature_dim
        self.cells = constants.SV2_CELLS
        token = constants.SV2_TOKEN_DIM
        self.w_cell = nn.Linear(constants.SV2_CELL_INPUTS * feature_dim, token)
        self.w_prior = nn.Linear(constants.SV2_PRIORS, token)
        self.pos = nn.Parameter(torch.empty(self.cells, token))
        self.layer = nn.TransformerEncoderLayer(
            token, constants.SV2_HEADS, constants.SV2_FFN_DIM, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.beta = nn.Parameter(torch.tensor(constants.SV2_BETA_INIT))
        self.gamma = nn.Parameter(torch.zeros(()))
        self.w = nn.Linear(token, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(constants.SV2_BIAS_INIT))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.branch_head = nn.Linear(feature_dim, 1)
        # frozen, stored in the checkpoint. Level std 0 = not set yet: forward refuses it (C5)
        self.register_buffer("level", torch.tensor([0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("text_projection", torch.eye(feature_dim))
        for module in (self.w_cell, self.w_prior, self.layer):
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        nn.init.normal_(self.pos, std=constants.SV2_POS_INIT_STD)
        nn.init.zeros_(self.w.weight)

    @property
    def width(self) -> int:
        """Columns ``P`` and ``D`` occupy at the end of an input row."""
        return constants.SV2_ROW_WIDTH

    def set_level(self, mo_mean: float, mo_std: float, rho_mean: float, rho_std: float) -> None:
        """Freeze ``m_x, s_x`` of the training split; the checkpoint carries them to every test."""
        if mo_std <= 0 or rho_std <= 0:
            raise ValueError(f"level std must be positive, got Mo {mo_std}, rho {rho_std}")
        self.level.copy_(self.level.new_tensor([mo_mean, mo_std, rho_mean, rho_std]))

    def set_text_projection(self, weight: Tensor) -> None:
        """CLIP's ``text_projection`` matrix, ``(D, D)``."""
        self.text_projection.copy_(weight.to(self.text_projection))

    def text_delta(self, class_feats: Tensor) -> Tensor:
        """``tau_delta``: normalised mean of the abnormal categories minus "normal" (index 0)."""
        if class_feats.shape[0] < constants.SV2_MIN_CLASSES:
            raise ValueError(f"tau_delta needs Normal + a category, got {class_feats.shape[0]}")
        with torch.autocast(device_type=class_feats.device.type, enabled=False):
            embedded = F.normalize(class_feats.float() @ self.text_projection.T, dim=-1)
            abnormal = F.normalize(embedded[1:].mean(dim=0), dim=-1)
            return abnormal - embedded[0]

    def branch_scores(self, pooled: Tensor) -> Tensor:
        """The training-only head: ``(*, D)`` -> ``(*)`` logits."""
        scores: Tensor = self.branch_head(pooled).squeeze(-1)
        return scores

    def _logits(
        self, cls: Tensor, cells: Tensor, change: Tensor, delta: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``(N, D)``, ``(N, 49, D)`` x 2, ``(D,)`` -> logits ``l (N, 49)``, float32, and priors."""
        if float(self.level[1]) <= 0 or float(self.level[3]) <= 0:
            raise ValueError("SA²-v2 level constants are not set: run sa2v2_cells stats (rule D4)")
        with torch.autocast(device_type=cells.device.type, enabled=False):
            mo, rho = cell_priors(cells, change)
            level_mo = (mo - self.level[0]) / self.level[1]
            level_rho = (rho - self.level[2]) / self.level[3]
            priors = torch.stack([zscore_cells(mo), zscore_cells(rho), level_mo, level_rho], dim=-1)
            unit = F.normalize(cells, dim=-1, eps=constants.STP_NORM_EPS)
            cosine = unit @ F.normalize(delta, dim=-1)
            text = self.gamma * zscore_cells(cosine)
            prior_term = priors @ self.beta
        tokens = (
            self.w_cell(torch.cat([cells, cells - cls.unsqueeze(1), change], dim=-1))
            + self.pos
            + self.w_prior(priors)
        )
        residual = self.w(self.layer(tokens)).squeeze(-1).float()
        return prior_term + text + residual, priors

    def forward(
        self, cls: Tensor, columns: Tensor, class_feats: Tensor, mask: Tensor | None = None
    ) -> SA2V2Output:
        """``cls (B, L, D)``, ``columns (B, L, 2 * 49 * D)``, ``class_feats (C, E)``.

        ``mask`` (``(B, L)``, 1 = valid): padded steps get ``x^AS = 0``, so they stay what the host
        sees there, and zero logits.
        """
        if columns.shape[-1] != self.width:
            raise ValueError(f"SA²-v2 columns {columns.shape[-1]} != {self.width}")
        valid = (
            mask.bool() if mask is not None
            else torch.ones(cls.shape[:2], dtype=torch.bool, device=cls.device)
        )
        flat = columns[valid].float().reshape(-1, 2, self.cells, self.feature_dim)
        cells, change = flat[:, 0], flat[:, 1]
        logits, _ = self._logits(cls[valid].float(), cells, change, self.text_delta(class_feats))
        alpha = (logits / self.log_temperature.exp()).softmax(dim=-1)
        pooled = (alpha.unsqueeze(-1) * cells).sum(dim=1)

        x_as = torch.zeros_like(cls, dtype=pooled.dtype)
        x_as[valid] = pooled
        attn = torch.zeros((*cls.shape[:2], self.cells), dtype=alpha.dtype, device=cls.device)
        attn[valid] = alpha
        cell_logits = torch.zeros_like(attn)
        cell_logits[valid] = logits + self.bias
        spread = logits.detach().std(dim=-1, unbiased=False).mean()
        added = x_as if self.gate is None else torch.tanh(self.gate) * x_as
        return SA2V2Output(cls + added.to(cls.dtype), x_as, attn, cell_logits, spread)

    def gate_value(self) -> Tensor | None:
        """``tanh(g)``, detached; ``None`` for an ungated SV2."""
        return None if self.gate is None else torch.tanh(self.gate.detach())


__all__ = ["SA2V2", "SA2V2Output", "cell_priors", "zscore_cells"]
