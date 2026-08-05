"""Phase 3 unit tests: ported baseline components + assembled KAT-VAD model."""

from __future__ import annotations

import pytest
import torch

from core.config import Config
from core.models import (
    KATVAD,
    BinaryHead,
    CoAttentionFusion,
    ConvScoreHead,
    SimScoreHead,
    TemporalEncoder,
)

HIDDEN, HEADS = 64, 4
BATCH, LENGTH, NUM_CLS = 2, 30, 3
LENGTHS = torch.tensor([30, 17])


def _small_temporal(**kwargs: object) -> TemporalEncoder:
    return TemporalEncoder(
        hidden_size=HIDDEN, num_layers=2, num_heads=HEADS,
        window_size=25, max_position_embeddings=128, **kwargs,  # type: ignore[arg-type]
    )


class TestTemporalEncoder:
    def test_output_shape_and_finite(self) -> None:
        torch.manual_seed(0)
        enc = _small_temporal().eval()
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        out = enc(v, LENGTHS)
        assert out.shape == (BATCH, LENGTH, HIDDEN)
        assert torch.isfinite(out).all()

    def test_residual_gate_starts_at_identity(self) -> None:
        torch.manual_seed(0)
        enc = _small_temporal(use_gate=True).eval()
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        # tanh(0) = 0 → gated branch contributes nothing at init
        torch.testing.assert_close(enc(v, LENGTHS), v)
        assert enc.gate_alpha is not None

    def test_no_gate_encoder_contributes(self) -> None:
        torch.manual_seed(0)
        enc = _small_temporal().eval()
        assert enc.gate_alpha is None
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        with torch.no_grad():
            out = enc(v, LENGTHS)
        assert not torch.allclose(out, v)  # v^t = F + encoder(F), encoder non-zero

    def test_full_attention_mask_path(self) -> None:
        torch.manual_seed(0)
        enc = _small_temporal()
        enc.window_size = 0  # fall back to plain padding mask
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        out = enc.eval()(v, LENGTHS)
        assert out.shape == v.shape and torch.isfinite(out).all()

    def test_gradients_flow(self) -> None:
        enc = _small_temporal()
        v = torch.randn(BATCH, LENGTH, HIDDEN, requires_grad=True)
        enc(v, LENGTHS).sum().backward()
        assert v.grad is not None and torch.isfinite(v.grad).all()

    def test_valid_positions_invariant_to_padded_content(self) -> None:
        torch.manual_seed(0)
        enc = _small_temporal().eval()
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        v2 = v.clone()
        v2[1, 17:] = 123.0  # scribble on padding of the short video
        with torch.no_grad():
            out1 = enc(v, LENGTHS)
            out2 = enc(v2, LENGTHS)
        torch.testing.assert_close(out1[1, :17], out2[1, :17])
        torch.testing.assert_close(out1[0], out2[0])


class TestFusionAndHeads:
    def test_fusion_shapes(self) -> None:
        torch.manual_seed(0)
        fusion = CoAttentionFusion(d_model=HIDDEN, nhead=HEADS, num_layers=2).eval()
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        t = torch.randn(BATCH, NUM_CLS, HIDDEN)
        v_fused, t_fused = fusion(v, t, LENGTHS)
        assert v_fused.shape == v.shape and t_fused.shape == t.shape

    def test_fusion_valid_rows_invariant_to_padded_content(self) -> None:
        torch.manual_seed(0)
        fusion = CoAttentionFusion(d_model=HIDDEN, nhead=HEADS, num_layers=2).eval()
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        t = torch.randn(BATCH, NUM_CLS, HIDDEN)
        v2 = v.clone()
        v2[1, 17:] = 55.0
        with torch.no_grad():
            v_a, t_a = fusion(v, t, LENGTHS)
            v_b, t_b = fusion(v2, t, LENGTHS)
        torch.testing.assert_close(t_a, t_b)  # padded keys masked out of text attn
        torch.testing.assert_close(v_a[1, :17], v_b[1, :17])

    def test_adaptive_binary_head_blend(self) -> None:
        torch.manual_seed(0)
        head = BinaryHead("adaptive", d_model=HIDDEN, num_layers=1, kernel_size=9).eval()
        before = torch.randn(BATCH, LENGTH, HIDDEN)
        after = torch.randn(BATCH, LENGTH, HIDDEN)
        with torch.no_grad():
            blended = head(before_fused=before, after_fused=after)
            pre = head.bin_head(before.permute(0, 2, 1)).squeeze(1)
            post = head.bin_fused_head(after.permute(0, 2, 1)).squeeze(1)
            w = torch.sigmoid(head.adp_alpha * head.adp_weight)
        torch.testing.assert_close(blended, pre * w + post * (1 - w))
        assert blended.shape == (BATCH, LENGTH)

    def test_conv_score_head_single_layer_shape(self) -> None:
        head = ConvScoreHead(d_model=HIDDEN, num_layers=1, kernel_size=9)
        assert len(head.convs) == 1 and len(head.norm) == 0
        out = head(torch.randn(BATCH, HIDDEN, LENGTH))
        assert out.shape == (BATCH, 1, LENGTH)

    def test_sim_head_is_scaled_cosine(self) -> None:
        head = SimScoreHead(temperature_init=0.2)
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        t = torch.randn(BATCH, NUM_CLS, HIDDEN)
        sim = head(v, t)
        cos = torch.einsum(
            "bte,bce->btc",
            v / v.norm(dim=-1, keepdim=True),
            t / t.norm(dim=-1, keepdim=True),
        )
        torch.testing.assert_close(sim, cos / head.temperature)


def _tiny_config(**kip_overrides: object) -> Config:
    cfg = Config()
    cfg.model.hidden_dim = HIDDEN
    cfg.model.temporal_layers = 1
    cfg.model.temporal_heads = HEADS
    cfg.model.temporal_max_positions = 128
    cfg.model.fusion_num_layers = 1
    cfg.kip.d_flow = 32
    cfg.kip.pmg_latent_dim = 16
    cfg.kip.align_proj_dim = 16
    for key, value in kip_overrides.items():
        setattr(cfg.kip, key, value)
    return cfg


class TestKATVAD:
    def _forward(self, cfg: Config) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model = KATVAD.from_config(cfg)
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        class_feats = torch.randn(NUM_CLS, HIDDEN)
        caption_feats = torch.randn(4, HIDDEN)
        out: dict[str, torch.Tensor] = model(
            v, LENGTHS, class_feats=class_feats, caption_feats=caption_feats
        )
        return out

    def test_kip_on_outputs(self) -> None:
        out = self._forward(_tiny_config())
        assert out["vis_feats"].shape == (BATCH, LENGTH, HIDDEN)
        assert out["eo_hat"].shape == (BATCH, LENGTH, 32)
        assert out["motion_scores"].shape == (BATCH, LENGTH)
        assert out["cls_bin_logits"].shape == (BATCH, LENGTH)
        assert out["cls_sim_mat"].shape == (BATCH, LENGTH, NUM_CLS)
        assert out["cap_bin_logits"].shape == (BATCH, LENGTH)
        assert out["cap_sim_mat"].shape == (BATCH, LENGTH, 4)
        for value in out.values():
            assert torch.isfinite(value).all()
        # v^k (not v^t) feeds fusion/H_bin: shift must have changed the features
        assert not torch.allclose(out["vis_feats"], out["vt"])

    def test_kip_off_is_baseline(self) -> None:
        out = self._forward(_tiny_config(enabled=False))
        assert "eo_hat" not in out and "motion_scores" not in out
        assert out["cls_bin_logits"].shape == (BATCH, LENGTH)

    def test_pmg_only_ablation_keeps_vt(self) -> None:
        out = self._forward(_tiny_config(pmg_only=True))
        assert "eo_hat" in out
        # gate-shift off → v^k equals v^t (up to padding zeroing)
        mask = (torch.arange(LENGTH)[None, :] < LENGTHS[:, None]).float()[..., None]
        torch.testing.assert_close(out["vis_feats"], out["vt"] * mask)

    def test_on_raw_features_ablation(self) -> None:
        cfg = _tiny_config(on_raw_features=True)
        torch.manual_seed(0)
        model = KATVAD.from_config(cfg)
        assert model.kip_on_raw_features is True
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        out = model(v, LENGTHS)
        assert out["eo_hat"].shape == (BATCH, LENGTH, 32)

    def test_gradients_reach_all_trainable_parts(self) -> None:
        cfg = _tiny_config()
        torch.manual_seed(0)
        model = KATVAD.from_config(cfg)
        v = torch.randn(BATCH, LENGTH, HIDDEN)
        out = model(v, LENGTHS, class_feats=torch.randn(NUM_CLS, HIDDEN))
        (out["cls_bin_logits"].sum() + out["cls_sim_mat"].sum()).backward()
        for name in ("temporal_encoder", "fusion", "bin_head", "sim_head"):
            grads = [
                p.grad for p in getattr(model, name).parameters() if p.grad is not None
            ]
            assert grads, f"no gradients reached {name}"

    def test_encode_text_requires_clip(self) -> None:
        model = KATVAD.from_config(_tiny_config())
        with pytest.raises(RuntimeError):
            model.encode_text(["Normal"])
