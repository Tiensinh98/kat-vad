"""Pseudo-flow generation head (PMG) — spec §2.

Autoencoder-style translator from RGB-temporal features to flow embeddings:
Conv1d 512→128 k3 → GELU → per-step Linear 128→128 → GELU → Conv1d 128→256 k3.
Derived from Pi-VAD PMG, single modality (optical flow).
"""

from __future__ import annotations

from torch import Tensor, nn

from core import constants

PMG_CONV_KERNEL = 3


class PMGFlowHead(nn.Module):
    """Translate temporal RGB features ``v^t`` into pseudo-flow embeddings ``ê_O``.

    Input ``(B, L, d_in)`` → output ``(B, L, d_flow)``.
    """

    def __init__(
        self,
        d_in: int = constants.HIDDEN_DIM,
        d_lat: int = constants.PMG_LATENT_DIM,
        d_flow: int = constants.FLOW_DIM,
    ) -> None:
        super().__init__()
        padding = PMG_CONV_KERNEL // 2
        self.enc = nn.Conv1d(d_in, d_lat, PMG_CONV_KERNEL, padding=padding)
        self.translator = nn.Linear(d_lat, d_lat)
        self.dec = nn.Conv1d(d_lat, d_flow, PMG_CONV_KERNEL, padding=padding)
        self.act = nn.GELU()

    def forward(self, vt: Tensor, mask: Tensor | None = None) -> Tensor:
        """Map ``vt (B, L, d_in)`` to ``ê_O (B, L, d_flow)``.

        Padded positions (``mask == 0``) are zeroed on input so the k=3 convs see
        the same zero boundary a shorter sequence would.
        """
        if mask is not None:
            vt = vt * mask.unsqueeze(-1)
        hidden = self.act(self.enc(vt.transpose(1, 2))).transpose(1, 2)  # (B, L, d_lat)
        hidden = self.act(self.translator(hidden))  # (B, L, d_lat)
        eo: Tensor = self.dec(hidden.transpose(1, 2))  # (B, d_flow, L)
        return eo.transpose(1, 2)  # (B, L, d_flow)


__all__ = ["PMGFlowHead"]
