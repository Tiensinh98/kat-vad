"""Synthetic end-to-end test (Phase-5 code-complete milestone).

On the Phase-4 fixture (real preprocessor + KNN cache, random features):
train via the CLI (KIP-on stage 2, KIP-off, stage 1) → kill → resume (same
RNG/data stream, weights equal within FP tolerance) → inference CLI →
evaluate CLI → visualize CLI. CPU-only, no downloads, stub text encoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from core import constants, evaluate, inference, train
from core.config import load_config
from core.tests.fixtures import FixtureLayout, build_fixture
from core.tools import visualize

# Reduce (not eliminate) FP scheduling noise: on Apple Silicon the Accelerate
# BLAS threads internally regardless of torch's pool, so identical op streams
# can differ at ULP level run-to-run. Resume tests therefore assert weight
# equality within a tight FP tolerance — an RNG/data-stream divergence would
# show up as O(1) drift after a single optimizer step, orders above it.
torch.set_num_threads(1)
RESUME_ATOL = 1e-5
RESUME_RTOL = 1e-5

EPOCHS = 2
BATCH = 4


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> FixtureLayout:
    return build_fixture(tmp_path_factory.mktemp("e2e"))


def _train_args(
    fixture: FixtureLayout,
    out_dir: Path,
    epochs: int = EPOCHS,
    extra: list[str] | None = None,
) -> list[str]:
    return [
        "--data-dir", str(fixture.data_dir),
        "--clip-dir", str(fixture.clip_dir),
        "--flow-dir", str(fixture.flow_dir),
        "--knn-cache", str(fixture.knn_cache_path),
        "--output-dir", str(out_dir),
        "--text-encoder", "stub",
        "--set", f"train.num_epochs={epochs}",
        "--set", f"train.batch_size={BATCH}",
        "--set", "loss.captions_from_definitions=true",
        # machine-independent: torch 2.4 MPS training diverges on this graph
        # (pending lesson P6), so the E2E always pins CPU
        "--set", "train.device=cpu",
        *(extra or []),
    ]


def _read_metrics(out_dir: Path) -> list[dict[str, float]]:
    lines = (out_dir / train.METRICS_FILENAME).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


class TestTraining:
    def test_kip_on_two_epochs(self, fixture: FixtureLayout, tmp_path: Path) -> None:
        out = tmp_path / "kip_on"
        train.main(_train_args(fixture, out))

        assert (out / train.CHECKPOINT_LAST).exists()
        assert (out / "config.yaml").exists()
        records = _read_metrics(out)
        assert records, "no metrics logged"
        for record in records:
            for key, value in record.items():
                if key not in ("epoch", "batch", "global_step"):
                    assert np.isfinite(value), f"{key} not finite: {record}"
        for key in ("mil", "mul_mil", "kip_rec", "kip_align", "kin", "cap_contrastive"):
            assert any(key in r for r in records), f"loss {key} never computed"

        payload = torch.load(out / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        assert payload["epoch"] == EPOCHS
        assert payload["batches_done"] == 0
        assert payload["class_names"][0] == "Normal"

    def test_kip_off_epoch_trains(self, fixture: FixtureLayout, tmp_path: Path) -> None:
        out = tmp_path / "kip_off"
        train.main(
            _train_args(fixture, out, epochs=1, extra=["--set", "kip.enabled=false"])
        )
        records = _read_metrics(out)
        assert records
        assert all("kip_rec" not in r for r in records)
        state = torch.load(out / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        assert not any(k.startswith("kip.") for k in state["model"])

    def test_stage2_warm_starts_from_stage1_ckpt(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """Spec §8 handoff: stage 2 seeds from stage-1 weights via --init-weights,
        with fresh counters/schedule (unlike --resume, which would inherit the
        exhausted stage-1 LR horizon and epoch counter)."""
        s1 = tmp_path / "s1"
        train.main(_train_args(fixture, s1, epochs=1, extra=["--set", "train.stage=1"]))
        s1_ckpt = s1 / train.CHECKPOINT_LAST

        s2 = tmp_path / "s2"
        train.main(
            _train_args(fixture, s2, epochs=1, extra=["--init-weights", str(s1_ckpt)])
        )
        records = _read_metrics(s2)
        assert records[0]["epoch"] == 0 and records[0]["global_step"] == 1, (
            "warm start must not inherit stage-1 counters"
        )
        assert any("mil" in r for r in records), "stage 2 losses never computed"
        payload = torch.load(s2 / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        assert payload["epoch"] == 1

        # KIP-off arch cannot warm-start from a KIP-on checkpoint: fail loud
        with pytest.raises(ValueError, match="init-weights"):
            train.main(_train_args(
                fixture, tmp_path / "bad", epochs=1,
                extra=["--set", "kip.enabled=false", "--init-weights", str(s1_ckpt)],
            ))

        # --resume and --init-weights together is an argparse error
        with pytest.raises(SystemExit):
            train.main(_train_args(
                fixture, tmp_path / "both", epochs=1,
                extra=["--resume", str(s1_ckpt), "--init-weights", str(s1_ckpt)],
            ))

    def test_stage1_warmup_reduces_rec_loss(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        out = tmp_path / "stage1"
        train.main(
            _train_args(
                fixture,
                out,
                epochs=12,
                extra=[
                    "--set", "train.stage=1",
                    # tiny fixture: a visible MSE drop needs a real LR, no warmup
                    # and θ=1 (no synthesis) so epochs see comparable data
                    "--set", "train.learning_rate=0.001",
                    "--set", "train.warmup_steps=0",
                    "--set", "dvs.theta=1.0",
                ],
            )
        )
        records = _read_metrics(out)
        assert all(np.isfinite(r["kip_rec"]) for r in records)
        last = int(records[-1]["epoch"])

        def epoch_mean(key: str, epoch: int) -> float:
            return float(np.mean([r[key] for r in records if r["epoch"] == epoch]))

        assert epoch_mean("kip_rec", last) < epoch_mean("kip_rec", 0), "L_KIP_rec did not drop"
        assert epoch_mean("kip_align", last) < epoch_mean("kip_align", 0), (
            "L_KIP_align did not drop"
        )


class TestKillAndResume:
    def test_resume_matches_straight_run(self, fixture: FixtureLayout, tmp_path: Path) -> None:
        # run A: straight through
        out_a = tmp_path / "straight"
        train.main(_train_args(fixture, out_a, epochs=EPOCHS))
        # run B: same 2-epoch horizon, killed after epoch 1, then resumed
        out_b = tmp_path / "resumed"
        train.main([*_train_args(fixture, out_b, epochs=EPOCHS), "--stop-after-epochs", "1"])
        train.main([
            *_train_args(fixture, out_b, epochs=EPOCHS),
            "--resume", str(out_b / train.CHECKPOINT_LAST),
        ])

        state_a = torch.load(out_a / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        state_b = torch.load(out_b / train.CHECKPOINT_LAST, map_location="cpu", weights_only=False)
        assert state_a["epoch"] == state_b["epoch"] == EPOCHS
        assert state_a["global_step"] == state_b["global_step"]
        for key, tensor in state_a["model"].items():
            torch.testing.assert_close(
                state_b["model"][key], tensor, atol=RESUME_ATOL, rtol=RESUME_RTOL,
                msg=f"weight drift in {key}",
            )

    def test_checkpoint_last_is_the_only_checkpoint(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """No step checkpoints, and no ``.part`` staging file left behind."""
        out = tmp_path / "only_last"
        train.main(_train_args(fixture, out, epochs=EPOCHS))

        assert (out / train.CHECKPOINT_LAST).exists()
        assert (out / train.CHECKPOINT_LAST).stat().st_size > 0
        written = sorted(q.name for q in out.iterdir() if q.suffix == ".pt")
        assert written == [train.CHECKPOINT_LAST], f"unexpected checkpoints: {written}"
        leftovers = [q.name for q in out.iterdir() if constants.CACHE_PART_SUFFIX in q.name]
        assert leftovers == [], f"staging files left behind: {leftovers}"

    def test_checkpoint_every_steps_is_rejected(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """The knob was removed, not defaulted off: an old runbook must fail loud.

        A config key that parses and silently does nothing is the failure mode
        lessons C19 and C24 are both about.
        """
        with pytest.raises(KeyError, match=r"train\.checkpoint_every_steps"):
            train.main([
                *_train_args(fixture, tmp_path / "rejected", epochs=1),
                "--set", "train.checkpoint_every_steps=1",
            ])

    def test_checkpoint_write_is_atomic(self, fixture: FixtureLayout, tmp_path: Path) -> None:
        """A save killed mid-write leaves the previous checkpoint intact.

        Pins the C11b fix directly: the torn payload lands in the ``.part``
        sibling, so ``checkpoint_last.pt`` is never the 0-byte file that used to
        survive a Colab timeout and only surface hours later at load time.
        """
        out = tmp_path / "atomic"
        train.main(_train_args(fixture, out, epochs=1))
        target = out / train.CHECKPOINT_LAST
        good_bytes = target.read_bytes()

        staged = target.with_name(target.name + constants.CACHE_PART_SUFFIX)
        real_save = torch.save

        def die_mid_save(obj: object, handle: object, *args: object, **kw: object) -> None:
            real_save(obj, handle, *args, **kw)  # type: ignore[arg-type]
            raise RuntimeError("killed mid-save (simulated Colab timeout)")

        with mock.patch.object(torch, "save", die_mid_save):
            with pytest.raises(RuntimeError, match="killed mid-save"):
                train.main([
                    *_train_args(fixture, out, epochs=EPOCHS),
                    "--resume", str(target),
                ])

        assert target.read_bytes() == good_bytes, "target was clobbered by a failed save"
        assert staged.exists(), "the torn write should have landed in .part"


class TestInferenceEvalViz:
    @pytest.fixture(scope="class")
    def trained(self, fixture: FixtureLayout, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("trained")
        train.main(_train_args(fixture, out, epochs=1))
        return out / train.CHECKPOINT_LAST

    def test_inference_cli(self, fixture: FixtureLayout, trained: Path, tmp_path: Path) -> None:
        out = tmp_path / "infer"
        inference.main([
            "--ckpt", str(trained),
            "--features", str(fixture.clip_dir),
            "--defs", str(fixture.data_dir / "defs.json"),
            "--output-dir", str(out),
            "--text-encoder", "stub",
            "--set", "train.device=cpu",
        ])
        npz_files = sorted(out.glob("*.npz"))
        assert len(npz_files) == len(fixture.train_ids) + len(fixture.test_ids)
        sample = np.load(npz_files[0])
        length = len(np.load(fixture.clip_dir / f"{npz_files[0].stem}.npy"))
        assert sample["score"].shape == (length,)
        assert sample["sim"].shape[0] == length
        assert np.isfinite(sample["score"]).all()
        assert (sample["score"] >= 0).all() and (sample["score"] <= 1).all()

    def test_equalize_window_anchors(self) -> None:
        """Lesson C28 control: which N-frame window survives (pure, no data)."""
        assert evaluate.equalize_window(10, 4, constants.EQUALIZE_ANCHOR_START) == (0, 4)
        assert evaluate.equalize_window(10, 4, constants.EQUALIZE_ANCHOR_END) == (6, 10)
        assert evaluate.equalize_window(10, 4, constants.EQUALIZE_ANCHOR_CENTER) == (3, 7)
        # exact fit keeps everything, whatever the anchor
        for anchor in constants.EQUALIZE_ANCHOR_CHOICES:
            assert evaluate.equalize_window(5, 5, anchor) == (0, 5)

    def test_equalize_window_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="anchor must be one of"):
            evaluate.equalize_window(10, 4, "middle")
        with pytest.raises(ValueError, match="must be positive"):
            evaluate.equalize_window(10, 0, constants.EQUALIZE_ANCHOR_CENTER)
        with pytest.raises(ValueError, match="shorter than target"):
            evaluate.equalize_window(3, 4, constants.EQUALIZE_ANCHOR_CENTER)

    def test_evaluate_equalize_length_crops_and_records(
        self, fixture: FixtureLayout, trained: Path, tmp_path: Path
    ) -> None:
        """Every scored clip ends up the same length, and the run says so."""
        out = tmp_path / "eval_eq"
        target = 3
        evaluate.main([
            "--ckpt", str(trained),
            "--data-dir", str(fixture.data_dir),
            "--clip-dir", str(fixture.clip_dir),
            "--output-dir", str(out),
            "--text-encoder", "stub",
            "--set", "train.device=cpu",
            "--save-scores",
            "--equalize-length", str(target),
            "--equalize-anchor", constants.EQUALIZE_ANCHOR_END,
        ])
        with (out / evaluate.RESULTS_FILENAME).open("r", encoding="utf-8") as fh:
            results = json.load(fh)
        eq = results["equalize"]
        assert eq["length"] == target
        assert eq["anchor"] == constants.EQUALIZE_ANCHOR_END
        assert eq["clips_kept"] + eq["clips_dropped"] == len(fixture.test_ids)
        assert results["num_videos"] == eq["clips_kept"]
        assert results["num_videos_total"] == len(fixture.test_ids)
        # the point of the control: no length variation is left to read
        for npz in (out / evaluate.SCORES_DIRNAME).glob("*.npz"):
            with np.load(npz) as payload:
                assert payload["score"].shape == (target,)
                assert payload["gt"].shape == (target,)

    def test_scoring_a_subset_reproduces_the_full_run_exactly(
        self, fixture: FixtureLayout, trained: Path, tmp_path: Path
    ) -> None:
        """Lesson C30: an item's score must not depend on which other items ran.

        Definitions are sampled per window, so a run-wide verbalizer RNG made
        ``--equalize-length``'s dropped clips shift every later clip's curve
        (measured on DADA: 32 of 34 uncropped clips moved, up to 0.0033). With a
        per-item stream, a subset reproduces the full run.

        Tolerance ``atol=1e-6``, not exact equality: the CPU forward is
        reproducible to float32 ULP across two invocations in one process, not
        bit-for-bit (~1.2e-7 observed once the rest of the suite has run first --
        same family as ``pending.md`` P3). The defect this pins moves scores by
        1e-3 to 1e-1, three orders above that floor.
        """
        def score(data_dir: Path, out: Path) -> dict[str, np.ndarray]:
            evaluate.main([
                "--ckpt", str(trained),
                "--data-dir", str(data_dir),
                "--clip-dir", str(fixture.clip_dir),
                "--output-dir", str(out),
                "--text-encoder", "stub",
                "--set", "train.device=cpu",
                "--save-scores",
            ])
            return {
                npz.stem: np.load(npz)["score"]
                for npz in (out / evaluate.SCORES_DIRNAME).glob("*.npz")
            }

        full = score(fixture.data_dir, tmp_path / "eval_full")

        subset_dir = tmp_path / "data_subset"
        subset_dir.mkdir()
        for name in (constants.DEFS_FILENAME, constants.META_FILENAME):
            source = fixture.data_dir / name
            if source.exists():
                (subset_dir / name).write_text(source.read_text(encoding="utf-8"), "utf-8")
        labels_path = fixture.data_dir / constants.FRAME_LABELS_TEST_FILENAME
        with labels_path.open("r", encoding="utf-8") as fh:
            labels = json.load(fh)
        kept = dict(sorted(labels.items())[1:])  # drop the first clip, as a filter would
        assert kept and len(kept) < len(labels)
        with (subset_dir / constants.FRAME_LABELS_TEST_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(kept, fh)

        subset = score(subset_dir, tmp_path / "eval_subset")
        assert set(subset) == set(kept)
        for video_id, curve in subset.items():
            np.testing.assert_allclose(
                curve, full[video_id], rtol=0, atol=1e-6,
                err_msg=f"{video_id} changed because another clip was skipped (C30)",
            )

    def test_evaluate_cli_and_visualize(
        self, fixture: FixtureLayout, trained: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "eval"
        evaluate.main([
            "--ckpt", str(trained),
            "--data-dir", str(fixture.data_dir),
            "--clip-dir", str(fixture.clip_dir),
            "--output-dir", str(out),
            "--text-encoder", "stub",
            "--set", "train.device=cpu",
            "--save-scores",
        ])
        with (out / evaluate.RESULTS_FILENAME).open("r", encoding="utf-8") as fh:
            results = json.load(fh)
        assert 0.0 <= results["auc"] <= 1.0
        assert 0.0 <= results["ap"] <= 1.0
        assert results["num_videos"] == len(fixture.test_ids)
        scores_dir = out / evaluate.SCORES_DIRNAME
        assert len(list(scores_dir.glob("*.npz"))) == len(fixture.test_ids)

        plots = tmp_path / "plots"
        visualize.main([
            "--scores", str(scores_dir),
            "--output-dir", str(plots),
            "--threshold", "0.5",
        ])
        pngs = list(plots.glob("*.png"))
        assert len(pngs) == len(fixture.test_ids)
        assert all(p.stat().st_size > 0 for p in pngs)

    def test_evaluate_is_run_to_run_deterministic(
        self, fixture: FixtureLayout, trained: Path, tmp_path: Path
    ) -> None:
        """Two identical evaluate runs must agree to FP tolerance.

        Guards the seeded verbalizer: definition sampling is per-window, so an
        unseeded verbalizer feeds different texts each run (measured on real
        data: ±0.3 per-video max_score swings across identical invocations).
        Tolerance is 1e-5, not bitwise — Apple Accelerate BLAS is
        ULP-nondeterministic (lesson P7, same bound as the resume tests).
        """
        results = []
        for name in ("det_a", "det_b"):
            out = tmp_path / name
            evaluate.main([
                "--ckpt", str(trained),
                "--data-dir", str(fixture.data_dir),
                "--clip-dir", str(fixture.clip_dir),
                "--output-dir", str(out),
                "--text-encoder", "stub",
                "--set", "train.device=cpu",
                "--save-scores",
            ])
            with (out / evaluate.RESULTS_FILENAME).open("r", encoding="utf-8") as fh:
                results.append(json.load(fh))
        assert results[0]["auc"] == pytest.approx(results[1]["auc"], abs=1e-5)
        assert results[0]["ap"] == pytest.approx(results[1]["ap"], abs=1e-5)
        for vid, stats in results[0]["per_video"].items():
            assert stats["max_score"] == pytest.approx(
                results[1]["per_video"][vid]["max_score"], abs=1e-5
            ), vid
        for npz_a in sorted((tmp_path / "det_a" / evaluate.SCORES_DIRNAME).glob("*.npz")):
            npz_b = tmp_path / "det_b" / evaluate.SCORES_DIRNAME / npz_a.name
            a, b = np.load(npz_a), np.load(npz_b)
            np.testing.assert_allclose(a["score"], b["score"], atol=1e-5, rtol=0,
                                       err_msg=npz_a.name)

    def test_baseline_ckpt_scores_through_eval(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """Gate-(a) path: a baseline-keyed ckpt runs through evaluate.main."""
        from core.config import Config
        from core.models.ckpt_compat import baseline_key_for
        from core.models.kat_vad import KATVAD

        cfg = Config()
        cfg.kip.enabled = False
        donor = KATVAD.from_config(cfg)
        state = {
            baseline_key_for(k): v
            for k, v in donor.state_dict().items()
            if baseline_key_for(k) is not None
        }
        ckpt = tmp_path / "fake_best.ckpt"
        torch.save({"state_dict": state}, ckpt)

        out = tmp_path / "gate_a"
        evaluate.main([
            "--baseline-ckpt", str(ckpt),
            "--data-dir", str(fixture.data_dir),
            "--clip-dir", str(fixture.clip_dir),
            "--output-dir", str(out),
            "--text-encoder", "stub",
            "--set", "kip.enabled=false",
            "--set", "train.device=cpu",
        ])
        assert (out / evaluate.RESULTS_FILENAME).exists()


class TestCheckpointArchitecture:
    """Lesson C34: eval must rebuild the architecture the checkpoint was trained with.

    Two failure modes with opposite loudness, which is the whole point:
    ``score_head_kernel`` changes a conv weight's shape and raises on load, while
    ``temporal_window`` only sizes an attention mask and loads clean — silently
    scoring the arm with the receptive field it was trained to *not* have.
    """

    def _trained_with(
        self, fixture: FixtureLayout, out: Path, extra: list[str]
    ) -> Path:
        train.main(_train_args(fixture, out, epochs=1, extra=extra))
        return out / train.CHECKPOINT_LAST

    def _eval_args(self, fixture: FixtureLayout, ckpt: Path, out: Path) -> list[str]:
        return [
            "--ckpt", str(ckpt),
            "--data-dir", str(fixture.data_dir),
            "--clip-dir", str(fixture.clip_dir),
            "--output-dir", str(out),
            "--text-encoder", "stub",
            "--set", "train.device=cpu",
        ]

    def test_shape_changing_field_no_longer_needs_a_flag(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """The reported crash: kernel 3 checkpoint, eval with no --set, used to raise."""
        ckpt = self._trained_with(
            fixture, tmp_path / "k3", ["--set", "model.score_head_kernel=3"]
        )
        evaluate.main(self._eval_args(fixture, ckpt, tmp_path / "eval_k3"))
        assert (tmp_path / "eval_k3" / "results.json").exists()

    def test_silent_field_is_restored_from_the_checkpoint(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """The dangerous one: temporal_window mismatch never raised, it just scored wrong."""
        ckpt = self._trained_with(
            fixture, tmp_path / "tw3", ["--set", "model.temporal_window=3"]
        )
        cfg = load_config(None, ["train.device=cpu"])
        assert cfg.model.temporal_window != 3, "fixture must differ from the CLI default"
        model = inference.load_model_for_scoring(
            cfg, torch.device("cpu"), ckpt, None, "stub", ["train.device=cpu"]
        )
        assert model.temporal_encoder.window_size == 3
        assert cfg.model.temporal_window == 3, "cfg must be corrected for downstream users"

    def test_contradicting_override_raises(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        ckpt = self._trained_with(
            fixture, tmp_path / "k3b", ["--set", "model.score_head_kernel=3"]
        )
        cfg = load_config(None, ["model.score_head_kernel=9"])
        with pytest.raises(ValueError, match="contradicts the checkpoint"):
            inference.load_model_for_scoring(
                cfg, torch.device("cpu"), ckpt, None, "stub",
                ["model.score_head_kernel=9"],
            )

    def test_unknown_architecture_key_raises(
        self, fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        """A checkpoint from a branch with extra fields (e.g. v3's kip.gate_type)."""
        ckpt = self._trained_with(fixture, tmp_path / "alien", [])
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        payload["config"]["kip"]["gate_type"] = "rank"
        torch.save(payload, ckpt)
        cfg = load_config(None, ["train.device=cpu"])
        with pytest.raises(KeyError, match="gate_type"):
            inference.load_model_for_scoring(
                cfg, torch.device("cpu"), ckpt, None, "stub", []
            )
