"""KAT-VAD assembled model (plain ``nn.Module``, no Lightning).

Data flow (spec §6):

    F (CLIP feats) → TemporalEncoder → v^t → KIP → v^k
    v^u, z^u = fusion(v^k, z^t);  y^bin = H_bin(pre=v^k, post=v^u);
    y^mul = H_mul(v^u, z^u)

With KIP disabled (ablation 1) ``v^k = v^t`` and the model is the LaGoVAD
baseline. ``vis_feats`` in the output dict is whatever feeds fusion/H_bin/L_neg
(``v^k`` when KIP is on — the spec's L_neg compatibility rule).

Text encoding is decoupled from the visual forward: ``encode_text`` needs the
(frozen) CLIP text tower + tokenizer, while ``forward`` accepts precomputed
text features so all visual-path code is testable without CLIP weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from core import constants
from core.config import validate_v2, validate_v5
from core.data.collate import padding_mask
from core.kip.kip_module import KIP
from core.models.clip_text import SoftPromptCLIPTextModel
from core.models.fusion import CoAttentionFusion
from core.models.heads import BinaryHead, MultiClassHead
from core.models.motion_residual import MotionResidual
from core.models.mqr import GridReadout, SpatialMILHead, build_readout
from core.models.temporal_encoder import TemporalEncoder

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from core.config import Config


class KATVAD(nn.Module):
    """LaGoVAD baseline + config-gated KIP splice."""

    def __init__(
        self,
        hidden_dim: int = constants.HIDDEN_DIM,
        temporal_layers: int = constants.TEMPORAL_LAYERS,
        temporal_heads: int = constants.TEMPORAL_HEADS,
        temporal_window: int = constants.TEMPORAL_WINDOW,
        temporal_max_positions: int = constants.TEMPORAL_MAX_POSITIONS,
        temp_gate: bool = False,
        fusion_num_layers: int = constants.FUSION_NUM_LAYERS,
        fusion_heads: int = constants.FUSION_HEADS,
        score_head_layers: int = constants.SCORE_HEAD_LAYERS,
        score_head_kernel: int = constants.SCORE_HEAD_KERNEL,
        bin_head_type: str = "adaptive",
        multiclass_temp: float = constants.MULTICLASS_TEMP,
        kip: KIP | None = None,
        kip_on_raw_features: bool = False,
        clip_text_model: SoftPromptCLIPTextModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        motion_dim: int = 0,
        grid_readout: GridReadout | None = None,
        smil_head: SpatialMILHead | None = None,
    ) -> None:
        super().__init__()
        # v2 motion stream (architecture §5): absent unless motion_dim > 0, so every
        # v1 / A0 / A1 checkpoint builds -- and loads -- exactly as before.
        self.motion_residual = (
            MotionResidual(hidden_dim, motion_dim) if motion_dim > 0 else None
        )
        # v5 (PREREG_V5 §2): the motion term comes from a readout of the V7 grid appended to
        # each row; absent unless v5.readout != none, so v1 / v2 checkpoints load unchanged.
        if grid_readout is not None and self.motion_residual is None:
            raise ValueError("a v5 grid readout feeds the v2 motion path (motion_dim > 0)")
        self.grid_readout = grid_readout
        self.smil_head = smil_head
        self.temporal_encoder = TemporalEncoder(
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            window_size=temporal_window,
            max_position_embeddings=temporal_max_positions,
            use_gate=temp_gate,
        )
        self.kip = kip
        self.kip_on_raw_features = kip_on_raw_features
        self.fusion = CoAttentionFusion(
            d_model=hidden_dim, nhead=fusion_heads, num_layers=fusion_num_layers
        )
        self.bin_head = BinaryHead(
            bin_head_type=bin_head_type,
            d_model=hidden_dim,
            num_layers=score_head_layers,
            kernel_size=score_head_kernel,
        )
        self.sim_head = MultiClassHead(temperature_init=multiclass_temp)
        self.clip_text_model = clip_text_model
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, cfg: Config, load_clip: bool = False) -> KATVAD:
        """Build from :class:`core.config.Config`; ``load_clip`` pulls HF weights."""
        validate_v2(cfg)
        validate_v5(cfg)
        v5 = cfg.v5
        grid_readout = smil_head = None
        if v5.readout != constants.V5_READOUT_OFF:
            grid_readout = build_readout(v5)
            if v5.smil_weight > 0 and v5.readout == constants.V5_READOUT_MQR:
                smil_head = SpatialMILHead(constants.V5_GRID_DIM // v5.queries)
        clip_text_model: SoftPromptCLIPTextModel | None = None
        tokenizer: PreTrainedTokenizerBase | None = None
        if load_clip:
            from transformers import AutoTokenizer

            clip_text_model = SoftPromptCLIPTextModel(
                constants.CLIP_MODEL_NAME, cfg.model.num_soft_prompts
            )
            tokenizer = AutoTokenizer.from_pretrained(
                constants.CLIP_MODEL_NAME, revision=constants.CLIP_MODEL_REVISION
            )
        return cls(
            hidden_dim=cfg.model.hidden_dim,
            temporal_layers=cfg.model.temporal_layers,
            temporal_heads=cfg.model.temporal_heads,
            temporal_window=cfg.model.temporal_window,
            temporal_max_positions=cfg.model.temporal_max_positions,
            temp_gate=cfg.model.temp_gate,
            fusion_num_layers=cfg.model.fusion_num_layers,
            fusion_heads=cfg.model.fusion_heads,
            score_head_layers=cfg.model.score_head_layers,
            score_head_kernel=cfg.model.score_head_kernel,
            bin_head_type=cfg.model.bin_head_type,
            multiclass_temp=cfg.model.multiclass_temp,
            kip=(
                KIP.from_config(cfg.kip, d=cfg.model.hidden_dim)
                if cfg.kip.enabled
                else None
            ),
            kip_on_raw_features=cfg.kip.enabled and cfg.kip.on_raw_features,
            clip_text_model=clip_text_model,
            tokenizer=tokenizer,
            motion_dim=cfg.model.motion_dim,
            grid_readout=grid_readout,
            smil_head=smil_head,
        )

    def encode_text(self, texts: list[str], use_soft_prompt: bool = True) -> Tensor:
        """Tokenize + encode class names/captions → ``(len(texts), D)``."""
        if self.clip_text_model is None or self.tokenizer is None:
            raise RuntimeError(
                "Text encoding requires clip_text_model and tokenizer "
                "(build with load_clip=True)"
            )
        max_length = (
            self.clip_text_model.max_text_tokens
            if use_soft_prompt
            else constants.CLIP_MAX_TOKENS
        )
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        pooled: Tensor = self.clip_text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_soft_prompt=use_soft_prompt,
        ).pooler_output
        return pooled

    def _fuse_and_head(
        self, vis_feats: Tensor, txt_feats: Tensor, lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fusion + both heads for one text-feature set (``(C, E)``)."""
        expanded_txt = txt_feats[None, :, :].expand(vis_feats.shape[0], -1, -1)
        v_fused, t_fused = self.fusion(vis_feats, expanded_txt, lengths)
        bin_logits = self.bin_head(before_fused=vis_feats, after_fused=v_fused)
        sim_mat = self.sim_head(v_fused, t_fused)
        return bin_logits, sim_mat

    def _grid_rows(self, rows: Tensor, outputs: dict[str, Tensor]) -> Tensor:
        """v5: ``[x ; u_baked ; G (; ΔG)]`` -> ``[x ; motion(readout(G))]`` (PREREG_V5 §2.1).

        The baked ``u`` columns are skipped: they exist for the input-cache check and S1-G2(b).
        """
        assert self.grid_readout is not None and self.motion_residual is not None  # nosec B101
        hidden, base = self.motion_residual.hidden_dim, (
            self.motion_residual.hidden_dim + self.motion_residual.motion_dim
        )
        tiles, dim = self.grid_readout.tiles, self.grid_readout.dim
        width = tiles * dim
        blocks = 2 if self.grid_readout.uses_change else 1
        if rows.shape[-1] != base + blocks * width:
            raise ValueError(
                f"v5 rows have {rows.shape[-1]} columns, expected {base} (A2) + {blocks} x "
                f"{width} (grid{' + change' if blocks == 2 else ''}) -- is the grid appended?"
            )
        lead = rows.shape[:-1]
        grid = rows[..., base : base + width].reshape(*lead, tiles, dim)
        change = (
            rows[..., base + width :].reshape(*lead, tiles, dim) if blocks == 2 else None
        )
        out = self.grid_readout(grid, change)
        if out.attn is not None:
            outputs["v5_attn"] = out.attn
        if self.smil_head is not None and out.z is not None:
            outputs["v5_smil"] = self.smil_head(out.z)
        if out.tile_logits is not None:  # v5.1: scorer logits e (L_tile, evidence maps)
            outputs["v5_tile_logits"] = out.tile_logits
        return torch.cat([rows[..., :hidden], out.motion], dim=-1)

    def forward(
        self,
        v_feat: Tensor,
        v_feat_l: Tensor,
        class_feats: Tensor | None = None,
        caption_feats: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Visual forward + optional per-text-set fusion/heads.

        ``v_feat (B, L, D)`` CLIP features, ``v_feat_l (B,)`` valid lengths,
        ``class_feats (C, D)`` / ``caption_feats (S, D)`` precomputed text
        features (see :meth:`encode_text`).

        Returns ``vis_feats`` (= ``v^k``, or ``v^t`` with KIP off), KIP outputs
        (``eo_hat``, ``motion_scores``) when KIP is on, and per-text-set
        ``cls_bin_logits``/``cls_sim_mat`` (+ ``cap_*`` for captions).
        """
        mask = padding_mask(v_feat_l, v_feat.shape[1]).to(v_feat.device)
        outputs: dict[str, Tensor] = {}
        if self.grid_readout is not None:
            v_feat = self._grid_rows(v_feat, outputs)
        if self.motion_residual is not None:
            # v2 A2/A3: rows are [x ; c*u~/sigma_u] -> h = x + W_u(...)  (architecture §5)
            v_feat, outputs["motion_share"] = self.motion_residual(v_feat, mask)
        vt = self.temporal_encoder(v_feat, v_feat_l)

        if self.kip is not None:
            # ablation 6: splice KIP on raw CLIP features instead of v^t
            kip_input = v_feat if self.kip_on_raw_features else vt
            vk, eo_hat, motion_scores = self.kip(kip_input, mask)
            outputs["vt"] = vt
            outputs["eo_hat"] = eo_hat
            outputs["motion_scores"] = motion_scores
            vis_feats = vk
        else:
            vis_feats = vt
        outputs["vis_feats"] = vis_feats

        if class_feats is not None:
            outputs["cls_bin_logits"], outputs["cls_sim_mat"] = self._fuse_and_head(
                vis_feats, class_feats, v_feat_l
            )
        if caption_feats is not None:
            outputs["cap_bin_logits"], outputs["cap_sim_mat"] = self._fuse_and_head(
                vis_feats, caption_feats, v_feat_l
            )
        return outputs


__all__ = ["KATVAD"]
