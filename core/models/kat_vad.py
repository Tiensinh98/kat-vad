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
from core.config import validate_stp, validate_v2, validate_v5
from core.data.collate import padding_mask
from core.kip.kip_module import KIP
from core.models.clip_text import SoftPromptCLIPTextModel, clip_text_projection
from core.models.fusion import CoAttentionFusion
from core.models.heads import BinaryHead, MultiClassHead
from core.models.motion_residual import MotionResidual
from core.models.mqr import GridReadout, SpatialMILHead
from core.models.sa2 import SpatialAttentionAggregation
from core.models.sa2v2 import SA2V2
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
        sa2: SpatialAttentionAggregation | None = None,
        motion_init_std: float = 0.0,
        motion_init_seed: int = 0,
        sa2v2: SA2V2 | None = None,
        motion_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # v2 motion stream (architecture §5): absent unless motion_dim > 0, so every
        # v1 / A0 / A1 checkpoint builds -- and loads -- exactly as before. W_u starts at
        # zero unless motion_init_std > 0 (core/docs/final/WU_ARMS.md, arm WI).
        self.motion_residual = (
            MotionResidual(hidden_dim, motion_dim, motion_init_std, motion_init_seed)
            if motion_dim > 0
            else None
        )
        # v5 (PREREG_V5 §2): the motion term comes from a readout of the V7 grid appended to
        # each row; absent unless v5.readout != none, so v1 / v2 checkpoints load unchanged.
        if grid_readout is not None and self.motion_residual is None:
            raise ValueError("a v5 grid readout feeds the v2 motion path (motion_dim > 0)")
        self.grid_readout = grid_readout
        self.smil_head = smil_head
        # LaGoVAD-STP (proposal §3.2.2): the K selected CLIP patch features are the last columns
        # of each row; absent unless stp.sa2, so every earlier checkpoint loads unchanged.
        if sa2 is not None and grid_readout is not None:
            raise ValueError("SA² and the v5 grid readout cannot be combined")
        self.sa2 = sa2
        # SA²-v2 (rule core/docs/final/SA2V2_RUN.md): absent unless stp.sa2v2
        self.sa2v2 = sa2v2
        self.motion_dropout = motion_dropout
        self.hidden_dim = hidden_dim
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
        validate_stp(cfg)
        v5 = cfg.v5
        grid_readout = smil_head = None
        if v5.readout != constants.V5_READOUT_OFF:
            grid_readout = GridReadout(v5.readout, v5.queries, v5.keys)
            if v5.smil_weight > 0:
                smil_head = SpatialMILHead(constants.V5_GRID_DIM // v5.queries)
        clip_text_model: SoftPromptCLIPTextModel | None = None
        tokenizer: PreTrainedTokenizerBase | None = None
        if load_clip:
            from transformers import AutoTokenizer

            prefix = cfg.stp.text_prompt == constants.STP_PROMPT_PREFIX
            clip_text_model = SoftPromptCLIPTextModel(
                constants.CLIP_MODEL_NAME,
                cfg.stp.prefix_len if prefix else cfg.model.num_soft_prompts,
                layout=cfg.stp.text_prompt,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                constants.CLIP_MODEL_NAME, revision=constants.CLIP_MODEL_REVISION
            )
        model = cls(
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
            motion_init_std=cfg.model.motion_init_std,
            motion_init_seed=cfg.train.seed,
            grid_readout=grid_readout,
            smil_head=smil_head,
            sa2=(
                SpatialAttentionAggregation(cfg.model.hidden_dim, cfg.stp.top_k)
                if cfg.stp.sa2
                else None
            ),
            sa2v2=SA2V2(cfg.model.hidden_dim) if cfg.stp.sa2v2 else None,
            motion_dropout=cfg.stp.motion_dropout if cfg.stp.sa2v2 else 0.0,
        )
        if load_clip and model.sa2v2 is not None:
            model.sa2v2.set_text_projection(clip_text_projection())  # rule D3
        return model

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
        width, tiles, dim = constants.V5_GRID_WIDTH, constants.V5_GRID_TILES, constants.V5_GRID_DIM
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
        return torch.cat([rows[..., :hidden], out.motion], dim=-1)

    def _stp_rows(self, rows: Tensor, mask: Tensor, outputs: dict[str, Tensor]) -> Tensor:
        """STP: ``[c ; u ; P_sel]`` -> ``[c + x^AS ; u]`` (proposal §3.2.2, Eq. 3).

        ``u`` (the motion columns, possibly none) passes through untouched; the motion fusion
        ``x_t = c~_t + W_u u_t`` then runs on ``c~`` (proposal §3.1, with ``W_u`` in place of
        ``tanh(alpha) W LN``: ``STP_BUILD.md`` D1).
        """
        assert self.sa2 is not None  # nosec B101 - caller checks
        width = self.sa2.width
        base = rows.shape[-1] - width
        motion_dim = 0 if self.motion_residual is None else self.motion_residual.motion_dim
        if base != self.hidden_dim + motion_dim:
            raise ValueError(
                f"STP rows have {rows.shape[-1]} columns, expected {self.hidden_dim} (CLIP) + "
                f"{motion_dim} (motion) + {width} (top-{self.sa2.top_k} patches) -- "
                "is the patch cache appended (--patch-dir)?"
            )
        out = self.sa2(rows[..., : self.hidden_dim], rows[..., base:], mask)
        outputs["stp_attn"] = out.attn
        return torch.cat([out.features, rows[..., self.hidden_dim : base]], dim=-1)

    def _sv2_rows(
        self, rows: Tensor, mask: Tensor, class_feats: Tensor | None, outputs: dict[str, Tensor]
    ) -> Tensor:
        """SA²-v2: ``[c ; u ; P ; D]`` -> ``[c + x^AS ; u]`` (rule ``SA2V2_RUN.md``)."""
        assert self.sa2v2 is not None  # nosec B101 - caller checks
        if class_feats is None:
            raise ValueError("SA²-v2 needs class_feats: its text term reads the definition")
        base = rows.shape[-1] - self.sa2v2.width
        motion_dim = 0 if self.motion_residual is None else self.motion_residual.motion_dim
        if base != self.hidden_dim + motion_dim:
            raise ValueError(
                f"SA²-v2 rows have {rows.shape[-1]} columns, expected {self.hidden_dim} (CLIP) + "
                f"{motion_dim} (motion) + {self.sa2v2.width} -- is the cell cache appended?"
            )
        out = self.sa2v2(rows[..., : self.hidden_dim], rows[..., base:], class_feats, mask)
        valid = mask.bool()
        ratio = out.x_as.detach()[valid].norm() / rows[..., : self.hidden_dim][valid].norm()
        outputs |= {"sv2_attn": out.attn, "sv2_x_as": out.x_as, "sv2_xas_ratio": ratio,
                    "sv2_cell_logits": out.cell_logits, "sv2_spread": out.spread}
        return torch.cat([out.features, rows[..., self.hidden_dim : base]], dim=-1)

    def _motion_mask(self, mask: Tensor) -> Tensor:
        """Motion dropout: dropped bags read as padding to ``W_u``, so its whole term is zero."""
        if not self.training or self.motion_dropout <= 0.0:
            return mask
        keep = torch.rand(mask.shape[0], device=mask.device) >= self.motion_dropout
        return mask * keep.unsqueeze(1).to(mask.dtype)

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
        if self.sa2 is not None:
            v_feat = self._stp_rows(v_feat, mask, outputs)
        if self.sa2v2 is not None:
            v_feat = self._sv2_rows(v_feat, mask, class_feats, outputs)
        if self.motion_residual is not None:
            # v2 A2/A3: rows are [x ; c*u~/sigma_u] -> h = x + W_u(...)  (architecture §5)
            v_feat, outputs["motion_share"] = self.motion_residual(
                v_feat, self._motion_mask(mask)
            )
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
