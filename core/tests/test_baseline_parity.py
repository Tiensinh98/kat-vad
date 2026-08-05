"""Phase 3 parity tests: core re-implementations vs. LaGoVAD baseline modules.

Each test builds the baseline module and ours with identical (randomly
initialized) weights — state dicts are copied 1:1, which also pins the
key layout the Phase-5 checkpoint loader relies on — and asserts equal
outputs on identical random inputs. No pretrained weights, CPU-only.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, RoFormerConfig

from core.losses import (
    CapContrastLoss,
    asymmetric_infonce_loss,
    mil_loss,
    multi_class_mil_loss,
    multi_class_mil_loss_v2,
    pseudo_sup_mil_loss,
    supervised_loss,
)
from core.models import (
    BinaryHead,
    CoAttentionFusion,
    MultiClassHead,
    RoFormerEncoder,
    SoftPromptCLIPTextModel,
    extended_local_attention_mask,
)
from core.tests.baseline_ref import load_baseline_module

HIDDEN, HEADS, LAYERS = 64, 4, 2
BATCH, LENGTH, NUM_CLS = 2, 30, 3
LENGTHS = (30, 17)
ATOL = 1e-5


def _padding_mask() -> torch.Tensor:
    lengths = torch.tensor(LENGTHS)
    return (torch.arange(LENGTH)[None, :] < lengths[:, None]).float()


class TestRoFormerParity:
    def _build_pair(self) -> tuple[torch.nn.Module, RoFormerEncoder]:
        ref_mod = load_baseline_module("modeling_roformer")
        config = RoFormerConfig(
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            num_hidden_layers=LAYERS,
            intermediate_size=HIDDEN * 4,
            hidden_act="gelu",
            use_cache=False,
            max_position_embeddings=128,
        )
        torch.manual_seed(0)
        ref = ref_mod.RoFormerEncoder(config).eval()
        ours = RoFormerEncoder(
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            num_heads=HEADS,
            max_position_embeddings=128,
        ).eval()
        ours.load_state_dict(ref.state_dict())
        return ref, ours

    def test_state_dict_keys_identical(self) -> None:
        ref, ours = self._build_pair()
        assert set(ref.state_dict().keys()) == set(ours.state_dict().keys())

    def test_outputs_match_with_window_mask(self) -> None:
        ref, ours = self._build_pair()
        torch.manual_seed(1)
        x = torch.randn(BATCH, LENGTH, HIDDEN)
        attn = extended_local_attention_mask(_padding_mask(), 25, dtype=x.dtype)
        with torch.no_grad():
            expected = ref(x, attn).last_hidden_state
            actual = ours(x, attn)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=1e-4)

    def test_window_mask_matches_baseline_formula(self) -> None:
        pad = _padding_mask()
        actual = extended_local_attention_mask(pad, 5, dtype=torch.float32)
        ws = 2
        band = torch.triu(torch.ones(LENGTH, LENGTH), -ws) * torch.tril(
            torch.ones(LENGTH, LENGTH), ws
        )
        expected = pad[:, None, None, :] * band[None, None, :, :]
        expected = (1.0 - expected) * torch.finfo(torch.float32).min
        torch.testing.assert_close(actual, expected)


class TestFusionParity:
    def test_co_attn_outputs_match(self) -> None:
        ref_mod = load_baseline_module("fusion_encoders")
        torch.manual_seed(0)
        ref = ref_mod.FusionV1("co_attn", d_model=HIDDEN, nhead=HEADS, num_layers=2).eval()
        ours = CoAttentionFusion(d_model=HIDDEN, nhead=HEADS, num_layers=2).eval()
        ours.load_state_dict(ref.state_dict())

        torch.manual_seed(1)
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        t = torch.randn(NUM_CLS, HIDDEN)[None].expand(BATCH, -1, -1)
        lengths = torch.tensor(LENGTHS)
        with torch.no_grad():
            v_ref, t_ref = ref(v, t, lengths)
            v_ours, t_ours = ours(v, t, lengths)
        torch.testing.assert_close(v_ours, v_ref, atol=ATOL, rtol=1e-4)
        torch.testing.assert_close(t_ours, t_ref, atol=ATOL, rtol=1e-4)


class TestHeadsParity:
    @pytest.mark.parametrize("head_type", ["vanilla", "fused_vanilla", "adaptive"])
    def test_binary_head_outputs_match(self, head_type: str) -> None:
        ref_mod = load_baseline_module("heads")
        torch.manual_seed(0)
        ref = ref_mod.BinaryHead(
            head_type, d_model=HIDDEN, num_layers=1, kernel_size=9,
            activation="gelu", norm="layernorm", adp_alpha_init=0.25,
        ).eval()
        ours = BinaryHead(
            head_type, d_model=HIDDEN, num_layers=1, kernel_size=9,
        ).eval()
        ours.load_state_dict(ref.state_dict())

        torch.manual_seed(1)
        before = torch.randn(BATCH, LENGTH, HIDDEN)
        after = torch.randn(BATCH, LENGTH, HIDDEN)
        kwargs = {
            "vanilla": {"before_fused": before},
            "fused_vanilla": {"after_fused": after},
            "adaptive": {"before_fused": before, "after_fused": after},
        }[head_type]
        with torch.no_grad():
            torch.testing.assert_close(
                ours(**kwargs), ref(**kwargs), atol=ATOL, rtol=1e-4
            )

    def test_multilayer_conv_head_matches(self) -> None:
        ref_mod = load_baseline_module("heads")
        torch.manual_seed(0)
        ref = ref_mod.ConvScoreHead(
            HIDDEN, num_layers=3, kernel_size=3, activation="gelu", norm="layernorm"
        ).eval()
        from core.models import ConvScoreHead

        ours = ConvScoreHead(HIDDEN, num_layers=3, kernel_size=3).eval()
        ours.load_state_dict(ref.state_dict())
        x = torch.randn(BATCH, HIDDEN, LENGTH)
        with torch.no_grad():
            torch.testing.assert_close(ours(x), ref(x), atol=ATOL, rtol=1e-4)

    def test_multiclass_sim_head_matches(self) -> None:
        ref_mod = load_baseline_module("heads")
        torch.manual_seed(0)
        ref = ref_mod.MultiClassHead("sim", temperature_init=0.2).eval()
        ours = MultiClassHead(temperature_init=0.2).eval()
        ours.load_state_dict(ref.state_dict())
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        t = torch.randn(BATCH, NUM_CLS, HIDDEN)
        with torch.no_grad():
            torch.testing.assert_close(ours(v, t), ref(v, t), atol=ATOL, rtol=1e-4)


class TestLossParity:
    def _batch(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(2)
        logits = torch.randn(6, 20) * 3
        lengths = torch.tensor([20, 15, 20, 9, 12, 20])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        cls_label_idx = torch.tensor([0, 1, 0, 2, 3, 0])
        frame_labels = torch.zeros(6, 20)
        for b, label in enumerate(cls_label_idx):
            if label != 0:
                frame_labels[b, 3:9] = 1.0
        return {
            "logits": logits,
            "lengths": lengths,
            "labels": labels,
            "cls_label_idx": cls_label_idx,
            "frame_labels": frame_labels,
        }

    def test_mil_losses_match(self) -> None:
        ref = load_baseline_module("losses")
        b = self._batch()
        torch.testing.assert_close(
            mil_loss(b["logits"], b["labels"], b["lengths"], topk_pct=4),
            ref.mil_loss(b["logits"], b["labels"], b["lengths"], topk_pct=4),
        )
        mul_logits = torch.randn(6, 20, 4)
        torch.testing.assert_close(
            multi_class_mil_loss(mul_logits, b["cls_label_idx"], b["lengths"], topk_pct=4),
            ref.multi_class_mil_loss(mul_logits, b["cls_label_idx"], b["lengths"], topk_pct=4),
        )
        torch.testing.assert_close(
            multi_class_mil_loss_v2(mul_logits, b["cls_label_idx"], b["lengths"], topk_pct=4),
            ref.multi_class_mil_loss_v2(
                mul_logits, b["cls_label_idx"], b["lengths"], topk_pct=4
            ),
        )

    def test_dvs_pair_matches(self) -> None:
        ref = load_baseline_module("losses")
        b = self._batch()
        torch.testing.assert_close(
            supervised_loss(b["logits"], b["frame_labels"], b["lengths"]),
            ref.supervised_loss(b["logits"], b["frame_labels"], b["lengths"]),
        )
        torch.testing.assert_close(
            pseudo_sup_mil_loss(b["logits"], b["frame_labels"], b["lengths"], topk_pct=4),
            ref.pseudo_sup_mil_loss(b["logits"], b["frame_labels"], b["lengths"], topk_pct=4),
        )

    def test_asymmetric_infonce_matches(self) -> None:
        ref = load_baseline_module("losses")
        torch.manual_seed(3)
        sim = torch.randn(3, 6)
        label = torch.tensor([1, 3, 4])
        torch.testing.assert_close(
            asymmetric_infonce_loss(sim, label), ref.asymmetric_infonce_loss(sim, label)
        )

    @pytest.mark.parametrize("contrast_type", ["vanilla", "n3"])
    def test_cap_contrast_matches(self, contrast_type: str) -> None:
        ref_mod = load_baseline_module("losses")
        b = self._batch()
        torch.manual_seed(4)
        v_feats = torch.randn(6, 20, HIDDEN)
        t_feats = torch.randn(3, HIDDEN)  # captions of the 3 abnormal videos
        ref_loss = ref_mod.CapContrastLoss(contrast_type, temperature=0.02)(
            b["logits"], b["lengths"], v_feats, t_feats,
            b["cls_label_idx"], b["frame_labels"],
        )
        our_loss = CapContrastLoss(contrast_type, temperature=0.02)(
            b["logits"], b["lengths"], v_feats, t_feats,
            b["cls_label_idx"], b["frame_labels"],
        )
        torch.testing.assert_close(our_loss, ref_loss)


class TestClipTextParity:
    NUM_PROMPTS = 8
    VOCAB = 100
    EOS_ID = 99
    BOS_ID = 98

    def _tiny_clip(self) -> CLIPTextModel:
        config = CLIPTextConfig(
            vocab_size=self.VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=77,
            eos_token_id=2,  # legacy id → argmax pooling branch (as openai/clip)
            bos_token_id=self.BOS_ID,
        )
        torch.manual_seed(0)
        return CLIPTextModel(config)

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        # rows: [BOS, tokens..., EOS, pad(0)...] — EOS is the highest id (CLIP style)
        input_ids = torch.zeros(2, 12, dtype=torch.long)
        input_ids[0, :7] = torch.tensor([self.BOS_ID, 5, 9, 23, 42, 7, self.EOS_ID])
        input_ids[1, :5] = torch.tensor([self.BOS_ID, 11, 3, 60, self.EOS_ID])
        attention_mask = torch.zeros(2, 12, dtype=torch.long)
        attention_mask[0, :7] = 1
        attention_mask[1, :5] = 1
        return input_ids, attention_mask

    @pytest.mark.parametrize("use_soft_prompt", [True, False])
    def test_pooled_output_matches(self, use_soft_prompt: bool) -> None:
        ref_mod = load_baseline_module("modeling_clip")
        base = self._tiny_clip()

        ref_clip_cls = ref_mod.CLIPTextModel
        try:
            ref_mod.CLIPTextModel = type(  # noqa: RUF100
                "FakeCLIPTextModel",
                (),
                {"from_pretrained": staticmethod(lambda *_a, **_k: copy.deepcopy(base))},
            )
            torch.manual_seed(1)
            ref = ref_mod.SoftPromptCLIPTextModel("unused", self.NUM_PROMPTS).eval()
        finally:
            ref_mod.CLIPTextModel = ref_clip_cls
        # transformers versions differ on exposing _use_flash_attention_2
        if not hasattr(ref.model.text_model, "_use_flash_attention_2"):
            ref.model.text_model._use_flash_attention_2 = False
        # transformers >= 4.5x dropped `return_dict` from CLIPEncoder.forward;
        # shim the ref's encoder call (kwargs-stripping adapter, math unchanged)
        real_encoder = ref.model.text_model.encoder

        class _EncoderShim(torch.nn.Module):
            def forward(self, **kwargs: object) -> object:
                kwargs.pop("return_dict", None)
                return real_encoder(**kwargs)

        ref.model.text_model.encoder = _EncoderShim()

        ours = SoftPromptCLIPTextModel(
            "unused", self.NUM_PROMPTS, clip_model=copy.deepcopy(base)
        ).eval()
        assert ours.prompt_embedding is not None and ref.prompt_embedding is not None
        ours.prompt_embedding.load_state_dict(ref.prompt_embedding.state_dict())

        input_ids, attention_mask = self._inputs()
        with torch.no_grad():
            expected = ref(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_soft_prompt=use_soft_prompt,
            ).pooler_output
            actual = ours(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_soft_prompt=use_soft_prompt,
            ).pooler_output
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=1e-4)

    def test_soft_prompts_receive_gradient(self) -> None:
        ours = SoftPromptCLIPTextModel("unused", self.NUM_PROMPTS, clip_model=self._tiny_clip())
        input_ids, attention_mask = self._inputs()
        pooled = ours(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        pooled.sum().backward()
        assert ours.prompt_embedding is not None
        grad = ours.prompt_embedding.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
