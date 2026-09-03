"""Phase 3 tests: the v3 inference graph and its checkpoint compat (plan T3.6).

v3 puts KIP's motion head (3e) and alignment projections (3f) off the inference
graph. A training checkpoint therefore carries keys the scoring model does not
build, and the loader must drop exactly those and nothing else — lesson C5: a
silent partial load yields a plausible model with randomly initialized layers.
"""

from __future__ import annotations

import pytest
import torch

from core import constants
from core.config import Config, KIPConfig
from core.kip import GATE_TYPE_CONSTANT, GATE_TYPE_MLP_FROZEN, GATE_TYPE_RANK, KIP
from core.models.ckpt_compat import KIP_TRAIN_ONLY_PREFIXES, load_kip_state_dict
from core.models.kat_vad import KATVAD

BATCH, LENGTH = 2, 16


def make_cfg(gate_type: str = GATE_TYPE_RANK) -> Config:
    cfg = Config()
    cfg.kip = KIPConfig(gate_type=gate_type)
    return cfg


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    feats = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
    lengths = torch.tensor([LENGTH, LENGTH - 5])
    return feats, lengths


class TestInferenceGraphShape:
    def test_train_graph_builds_3e_and_3f(self):
        kip = KIP(train_only_modules=True)
        assert kip.mhead is not None
        assert kip.proj_flow is not None
        assert kip.proj_rgb is not None

    def test_inference_graph_omits_3e_and_3f(self):
        kip = KIP(train_only_modules=False)
        assert kip.mhead is None
        assert kip.proj_flow is None
        assert kip.proj_rgb is None
        assert not any(k.startswith(KIP_TRAIN_ONLY_PREFIXES) for k in kip.state_dict())

    def test_inference_graph_returns_no_motion_curve(self):
        kip = KIP(train_only_modules=False)
        vk, eo, yo, _ = kip(torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM))
        assert vk.shape == (BATCH, LENGTH, constants.HIDDEN_DIM)
        assert eo.shape == (BATCH, LENGTH, constants.FLOW_DIM)
        assert yo is None

    def test_inference_graph_parameter_count_matches_v3_table(self):
        """v3 §9: 311,808 params on the inference path, zero on the score path."""
        kip = KIP(gate_type=GATE_TYPE_RANK, train_only_modules=False)
        assert sum(p.numel() for p in kip.parameters()) == 311_808

    def test_scores_are_identical_across_graphs(self):
        """Dropping 3e/3f must not move the curve — they were never on it."""
        cfg = make_cfg()
        torch.manual_seed(7)
        train_model = KATVAD.from_config(cfg, training=True).eval()
        infer_model = KATVAD.from_config(cfg, training=False).eval()
        load_kip_state_dict(infer_model, train_model.state_dict())

        feats, lengths = make_inputs()
        text = torch.randn(3, constants.HIDDEN_DIM)
        with torch.no_grad():
            a = train_model(feats, lengths, class_feats=text)
            b = infer_model(feats, lengths, class_feats=text)
        assert torch.equal(a["cls_bin_logits"], b["cls_bin_logits"])
        assert torch.equal(a["vis_feats"], b["vis_feats"])
        assert "motion_scores" in a
        assert "motion_scores" not in b

    def test_project_for_align_raises_on_inference_graph(self):
        kip = KIP(train_only_modules=False)
        eo = torch.randn(BATCH, LENGTH, constants.FLOW_DIM)
        vt = torch.randn(BATCH, LENGTH, constants.HIDDEN_DIM)
        with pytest.raises(RuntimeError, match="requires the training graph"):
            kip.project_for_align(eo, vt)


class TestCheckpointCompat:
    def test_train_checkpoint_loads_into_inference_graph(self):
        cfg = make_cfg()
        train_model = KATVAD.from_config(cfg, training=True)
        infer_model = KATVAD.from_config(cfg, training=False)
        report = load_kip_state_dict(infer_model, train_model.state_dict())
        assert len(report.skipped_train_only) == 8  # mhead 4 + proj_flow 2 + proj_rgb 2
        assert all(k.startswith(KIP_TRAIN_ONLY_PREFIXES) for k in report.skipped_train_only)

    def test_weights_actually_transfer(self):
        """A drop-list must not become a skip-everything."""
        cfg = make_cfg()
        torch.manual_seed(11)
        train_model = KATVAD.from_config(cfg, training=True)
        infer_model = KATVAD.from_config(cfg, training=False)
        with torch.no_grad():
            for param in train_model.kip.pmg.parameters():  # type: ignore[union-attr]
                param.add_(1.234)
        load_kip_state_dict(infer_model, train_model.state_dict())
        for ours, theirs in zip(
            infer_model.kip.pmg.parameters(),  # type: ignore[union-attr]
            train_model.kip.pmg.parameters(),  # type: ignore[union-attr]
            strict=True,
        ):
            assert torch.equal(ours, theirs)

    def test_unrelated_extra_key_still_raises(self):
        """The allowlist is a list, not a policy of tolerance."""
        cfg = make_cfg()
        train_model = KATVAD.from_config(cfg, training=True)
        infer_model = KATVAD.from_config(cfg, training=False)
        state = dict(train_model.state_dict())
        state["kip.pmg.bogus_tensor"] = torch.zeros(3)
        with pytest.raises(RuntimeError):
            load_kip_state_dict(infer_model, state)

    def test_missing_required_key_raises(self):
        cfg = make_cfg()
        train_model = KATVAD.from_config(cfg, training=True)
        infer_model = KATVAD.from_config(cfg, training=False)
        state = dict(train_model.state_dict())
        del state["kip.pmg.enc.weight"]
        with pytest.raises(KeyError, match="missing weights"):
            load_kip_state_dict(infer_model, state)

    def test_train_graph_to_train_graph_drops_nothing(self):
        cfg = make_cfg()
        a = KATVAD.from_config(cfg, training=True)
        b = KATVAD.from_config(cfg, training=True)
        report = load_kip_state_dict(b, a.state_dict())
        assert report.skipped_train_only == []

    def test_kip_disabled_model_round_trips(self):
        cfg = Config()
        cfg.kip = KIPConfig(enabled=False)
        a = KATVAD.from_config(cfg, training=True)
        b = KATVAD.from_config(cfg, training=False)
        assert load_kip_state_dict(b, a.state_dict()).skipped_train_only == []


class TestGateTypeMismatchGuard:
    """T3.4 — scoring an mlp_frozen checkpoint under the rank gate must not work."""

    def test_mlp_checkpoint_into_rank_model_raises(self):
        frozen = KATVAD.from_config(make_cfg(GATE_TYPE_MLP_FROZEN), training=True)
        rank = KATVAD.from_config(make_cfg(GATE_TYPE_RANK), training=False)
        with pytest.raises(KeyError, match="trained with the frozen-MLP gate"):
            load_kip_state_dict(rank, frozen.state_dict())

    def test_rank_checkpoint_into_mlp_model_raises(self):
        rank = KATVAD.from_config(make_cfg(GATE_TYPE_RANK), training=True)
        frozen = KATVAD.from_config(make_cfg(GATE_TYPE_MLP_FROZEN), training=False)
        with pytest.raises(KeyError, match=r"carries no kip\.shift\.mlp"):
            load_kip_state_dict(frozen, rank.state_dict())

    def test_error_names_the_remedy(self):
        frozen = KATVAD.from_config(make_cfg(GATE_TYPE_MLP_FROZEN), training=True)
        rank = KATVAD.from_config(make_cfg(GATE_TYPE_RANK), training=False)
        with pytest.raises(KeyError) as excinfo:
            load_kip_state_dict(rank, frozen.state_dict())
        assert "--gate-type mlp_frozen" in str(excinfo.value)

    def test_matching_gate_types_load_cleanly(self):
        for gate_type in (GATE_TYPE_RANK, GATE_TYPE_MLP_FROZEN, GATE_TYPE_CONSTANT):
            cfg = make_cfg(gate_type)
            train_model = KATVAD.from_config(cfg, training=True)
            infer_model = KATVAD.from_config(cfg, training=False)
            load_kip_state_dict(infer_model, train_model.state_dict())

    def test_parameter_free_gates_are_interchangeable_by_keys_alone(self):
        """rank and constant share a key layout, so keys cannot separate them.

        Documents a real limitation of the guard: it catches the mlp/non-mlp
        boundary, not rank-vs-constant. The run manifest (lesson C17, deferred to
        the training-pipeline plan) is what closes that gap.
        """
        rank = KATVAD.from_config(make_cfg(GATE_TYPE_RANK), training=True)
        const = KATVAD.from_config(make_cfg(GATE_TYPE_CONSTANT), training=False)
        load_kip_state_dict(const, rank.state_dict())  # does not raise


class TestInferenceCLI:
    def test_gate_type_flag_is_exposed(self):
        from core.inference import build_arg_parser

        args = build_arg_parser().parse_args(
            ["--features", "x.npy", "--output-dir", "out", "--gate-type", "mlp_frozen"]
        )
        assert args.gate_type == GATE_TYPE_MLP_FROZEN

    def test_gate_type_defaults_to_config(self):
        from core.inference import build_arg_parser

        args = build_arg_parser().parse_args(["--features", "x.npy", "--output-dir", "out"])
        assert args.gate_type is None

    def test_invalid_gate_type_rejected_by_argparse(self):
        from core.inference import build_arg_parser

        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(
                ["--features", "x.npy", "--output-dir", "out", "--gate-type", "bogus"]
            )
