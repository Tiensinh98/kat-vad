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

from torch import Tensor, nn

from core import constants
from core.data.collate import padding_mask
from core.kip.kip_module import KIP
from core.models.clip_text import SoftPromptCLIPTextModel
from core.models.fusion import CoAttentionFusion
from core.models.heads import BinaryHead, MultiClassHead
from core.models.temporal_encoder import TemporalEncoder

KIP_DIAG_PREFIX = "kip_diag/"

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
    ) -> None:
        super().__init__()
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
    def from_config(
        cls, cfg: Config, load_clip: bool = False, training: bool = True
    ) -> KATVAD:
        """Build from :class:`core.config.Config`; ``load_clip`` pulls HF weights.

        ``training=False`` builds KIP's inference graph — no motion head, no
        alignment projections (v3 Phases 3e/3f are train-time only).
        """
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
                KIP.from_config(cfg.kip, d=cfg.model.hidden_dim, training=training)
                if cfg.kip.enabled
                else None
            ),
            kip_on_raw_features=cfg.kip.enabled and cfg.kip.on_raw_features,
            clip_text_model=clip_text_model,
            tokenizer=tokenizer,
        )

    def encode_text(
        self, texts: list[str], use_soft_prompt: bool = True
    ) -> Tensor:
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

    def forward(
        self,
        v_feat: Tensor,
        v_feat_l: Tensor,
        class_feats: Tensor | None = None,
        caption_feats: Tensor | None = None,
        return_kip_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        """Visual forward + optional per-text-set fusion/heads.

        ``v_feat (B, L, D)`` CLIP features, ``v_feat_l (B,)`` valid lengths,
        ``class_feats (C, D)`` / ``caption_feats (S, D)`` precomputed text
        features (see :meth:`encode_text`).

        Returns ``vis_feats`` (= ``v^k``, or ``v^t`` with KIP off), KIP outputs
        (``eo_hat``, ``motion_scores``) when KIP is on, and per-text-set
        ``cls_bin_logits``/``cls_sim_mat`` (+ ``cap_*`` for captions).

        ``return_kip_diagnostics`` adds the gate internals under ``kip_diag/*``
        keys (``s``, ``gate_ratio``, ``m``, ``mu_norm``, ``eo_norm``). Off by
        default: it costs a second gate evaluation and the training path has no
        use for it.
        """
        mask = padding_mask(v_feat_l, v_feat.shape[1]).to(v_feat.device)
        vt = self.temporal_encoder(v_feat, v_feat_l)

        outputs: dict[str, Tensor] = {}
        if self.kip is not None:
            # ablation 6: splice KIP on raw CLIP features instead of v^t
            kip_input = v_feat if self.kip_on_raw_features else vt
            kip_out = self.kip(kip_input, mask, return_diagnostics=return_kip_diagnostics)
            outputs["vt"] = vt
            outputs["eo_hat"] = kip_out.eo_hat
            if kip_out.motion_scores is not None:  # absent on the inference graph
                outputs["motion_scores"] = kip_out.motion_scores
            if kip_out.diagnostics is not None:
                for name, value in kip_out.diagnostics.items():
                    outputs[f"{KIP_DIAG_PREFIX}{name}"] = value
            vis_feats = kip_out.vk
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


__all__ = ["KATVAD", "KIP_DIAG_PREFIX"]
