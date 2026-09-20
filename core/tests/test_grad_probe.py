"""Tests for the gradient attribution probe (``core.tools.grad_probe``).

The probe exists to answer one question — how much of the shared trunk's
gradient comes from ``L_KIP_rec`` rather than from the anomaly objective — so the
properties pinned here are the ones that would make its answer wrong rather than
merely ugly:

* the weighted terms it splits must **re-sum to the total the trainer optimizes**,
  because a table that has drifted from ``Trainer.compute_losses`` would attribute
  a sum nobody is descending;
* a term that reaches a module must report a **non-zero** norm there and one that
  does not must report **zero**, not raise;
* the arms it cannot answer for — KIP-off, stage 1 — must fail **loudly**.

No data, no downloads, no GPU: the Phase-4 synthetic fixture on CPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from core.config import load_config
from core.data.collate import collate_variable_length
from core.tests.fixtures import FixtureLayout, build_fixture
from core.tools import grad_probe
from core.train import build_trainer, epoch_permutation

torch.set_num_threads(1)

BATCH = 4


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> FixtureLayout:
    return build_fixture(tmp_path_factory.mktemp("grad_probe"))


def _trainer(fixture: FixtureLayout, out_dir: Path, overrides: list[str] | None = None):
    cfg = load_config(
        None,
        [
            f"train.batch_size={BATCH}",
            "train.device=cpu",
            "train.num_epochs=1",
            *(overrides or []),
        ],
    )
    return build_trainer(
        cfg=cfg,
        output_dir=out_dir,
        data_dir=fixture.data_dir,
        clip_dir=fixture.clip_dir,
        flow_dir=fixture.flow_dir,
        knn_cache_path=fixture.knn_cache_path,
        text_encoder="stub",
    )


def _one_batch(trainer) -> dict:
    order = epoch_permutation(trainer.cfg.train.seed, 0, len(trainer.dataset))
    return collate_variable_length([trainer.dataset[i] for i in order[:BATCH]])


class TestTermSplit:
    def test_weighted_terms_resum_to_the_trainer_total(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        # The guard that keeps term_weights() honest: probe_batch raises when the
        # reassembled sum disagrees with compute_losses, so reaching a record at
        # all is the assertion.
        trainer = _trainer(fixture, tmp_path / "resum")
        trainer.model.train()
        record = grad_probe.probe_batch(trainer, _one_batch(trainer))
        rebuilt = sum(record["weighted"].values())
        assert rebuilt == pytest.approx(record["losses"]["total"], abs=1e-4)

    def test_every_loss_key_the_trainer_emits_has_a_weight(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        trainer = _trainer(
            fixture, tmp_path / "keys", ["loss.captions_from_definitions=true"]
        )
        trainer.model.train()
        losses = trainer.compute_losses(_one_batch(trainer))
        emitted = set(losses) - {"total"}
        assert emitted <= set(grad_probe.term_weights(trainer.cfg))

    def test_a_stale_weight_table_raises_instead_of_misattributing(
        self, fixture: FixtureLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _trainer(fixture, tmp_path / "stale")
        trainer.model.train()
        truth = grad_probe.term_weights
        monkeypatch.setattr(
            grad_probe, "term_weights", lambda cfg: {k: v * 2 for k, v in truth(cfg).items()}
        )
        with pytest.raises(ValueError, match="fallen out of step"):
            grad_probe.probe_batch(trainer, _one_batch(trainer))

    def test_an_unweighted_loss_key_raises(
        self, fixture: FixtureLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = _trainer(fixture, tmp_path / "unknown")
        trainer.model.train()
        truth = grad_probe.term_weights
        monkeypatch.setattr(
            grad_probe,
            "term_weights",
            lambda cfg: {k: v for k, v in truth(cfg).items() if k != "kip_rec"},
        )
        with pytest.raises(ValueError, match="no entry in term_weights"):
            grad_probe.probe_batch(trainer, _one_batch(trainer))


class TestGradientAttribution:
    def test_kip_rec_reaches_the_trunk_and_not_the_heads(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        # The whole premise of the diagnostic: PMG reads v^t, so L_KIP_rec writes
        # into the temporal encoder. It never reaches fusion or the heads, and
        # that must read as 0 rather than as an exception (allow_unused).
        trainer = _trainer(fixture, tmp_path / "reach")
        trainer.model.train()
        norms = grad_probe.probe_batch(trainer, _one_batch(trainer))["grad_norm"]
        assert norms["temporal_encoder"]["kip_rec"] > 0.0
        assert norms["kip"]["kip_rec"] > 0.0
        assert norms["fusion"]["kip_rec"] == pytest.approx(0.0)
        assert norms["heads"]["kip_rec"] == pytest.approx(0.0)
        assert norms["temporal_encoder"]["task"] > 0.0

    def test_trunk_block_reports_ratio_and_a_valid_cosine(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        trainer = _trainer(fixture, tmp_path / "trunk")
        trainer.model.train()
        trunk = grad_probe.probe_batch(trainer, _one_batch(trainer))["trunk"]
        assert trunk["kip_over_task"]["kip_rec"] > 0.0
        assert -1.0 <= trunk["cosine_with_task"]["kip_rec"] <= 1.0

    def test_lambda_rec_scales_the_trunk_ratio_linearly(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        # The probe measures the WEIGHTED gradient, so halving lambda_rec must
        # halve the ratio. If it does not, the weight is not where the report says
        # it is and no lambda recommendation read off this probe would hold.
        base = _trainer(fixture, tmp_path / "lam1", ["loss.lambda_rec=1.0"])
        half = _trainer(fixture, tmp_path / "lam05", ["loss.lambda_rec=0.5"])
        half.model.load_state_dict(base.model.state_dict())
        base.model.eval()
        half.model.eval()
        ratio_base = grad_probe.probe_batch(base, _one_batch(base))["trunk"][
            "kip_over_task"
        ]["kip_rec"]
        ratio_half = grad_probe.probe_batch(half, _one_batch(half))["trunk"][
            "kip_over_task"
        ]["kip_rec"]
        assert ratio_half == pytest.approx(ratio_base / 2.0, rel=1e-3)


class TestRefusals:
    def test_kip_off_is_refused(self, fixture: FixtureLayout, tmp_path: Path) -> None:
        trainer = _trainer(fixture, tmp_path / "off", ["kip.enabled=false"])
        with pytest.raises(ValueError, match="kip_on arm"):
            grad_probe.run_probe(trainer, num_batches=1, epoch=0)

    def test_stage_one_is_refused(self, fixture: FixtureLayout, tmp_path: Path) -> None:
        trainer = _trainer(fixture, tmp_path / "stage1", ["train.stage=1"])
        with pytest.raises(ValueError, match="stage"):
            grad_probe.run_probe(trainer, num_batches=1, epoch=0)


def _cli_args(fixture: FixtureLayout, out: Path) -> list[str]:
    return [
        "--data-dir", str(fixture.data_dir),
        "--clip-dir", str(fixture.clip_dir),
        "--flow-dir", str(fixture.flow_dir),
        "--knn-cache", str(fixture.knn_cache_path),
        "--output-dir", str(out),
        "--text-encoder", "stub",
        "--num-batches", "2",
        "--batch-size", str(BATCH),
        "--set", "train.device=cpu",
        "--set", "train.num_epochs=1",
    ]


class TestCli:
    def test_cli_writes_both_artifacts(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        out = tmp_path / "cli"
        grad_probe.main(_cli_args(fixture, out))
        payload = json.loads((out / grad_probe.REPORT_JSON).read_text(encoding="utf-8"))
        assert payload["batches"] == 2
        assert payload["checkpoint"] is None
        summary = payload["summary"]
        assert math.isfinite(summary["trunk_kip_total_over_task"])
        markdown = (out / grad_probe.REPORT_MARKDOWN).read_text(encoding="utf-8")
        assert "Gradient arriving at the shared temporal encoder" in markdown
        assert "`kip_rec`" in markdown

    def test_cli_loads_a_checkpoint_without_the_text_tower(
        self, fixture: FixtureLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stage-1 checkpoint carries no ``clip_text_model.*`` — probe it anyway.

        Stage 1 builds the model with ``load_clip=False`` (``build_trainer``), so
        its checkpoint has no CLIP text tower while the stage-2 model the probe
        builds does. Probing that checkpoint *under the stage-2 objective* is the
        pre-registered D2-c arm, and the strict same-run resume path refuses it
        over that one absence. Reproduced here without downloading CLIP by
        attaching a stand-in module under the same name; the stub text encoder
        never calls it.
        """
        donor = _trainer(fixture, tmp_path / "donor")
        ckpt = tmp_path / "stage1.pt"
        donor.save_checkpoint(ckpt)
        assert not any(k.startswith("clip_text_model.") for k in donor.model.state_dict())

        real_build = grad_probe.build_trainer

        def _with_text_tower(**kwargs: object):
            trainer = real_build(**kwargs)  # type: ignore[arg-type]
            trainer.model.clip_text_model = torch.nn.Linear(2, 2)
            return trainer

        monkeypatch.setattr(grad_probe, "build_trainer", _with_text_tower)
        out = tmp_path / "from_stage1"
        grad_probe.main([*_cli_args(fixture, out), "--checkpoint", str(ckpt)])
        payload = json.loads((out / grad_probe.REPORT_JSON).read_text(encoding="utf-8"))
        assert payload["checkpoint"] == str(ckpt)
        assert math.isfinite(payload["summary"]["trunk_kip_total_over_task"])

    def test_cli_rejects_a_checkpoint_that_does_not_match(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """Only the text tower may be absent; anything else is a loud ValueError.

        This is what separates the weights-only loader from ``strict=False``: a
        KIP-off checkpoint probed under a KIP-on config must not load silently.
        """
        donor = _trainer(fixture, tmp_path / "donor_bad")
        ckpt = tmp_path / "bad.pt"
        donor.save_checkpoint(ckpt)
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        dropped = next(k for k in payload["model"] if k.startswith("kip."))
        del payload["model"][dropped]
        torch.save(payload, ckpt)
        with pytest.raises(ValueError, match="--checkpoint"):
            grad_probe.main(
                [*_cli_args(fixture, tmp_path / "rejected"), "--checkpoint", str(ckpt)]
            )
