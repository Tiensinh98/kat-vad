"""Tests for the z-scored flow cache (Option A: ``core.flow.zscore`` + ``zscore_cache``).

The properties pinned are the ones that would make a KIP-on number on the new
cache mean something other than what the plan
(``.project/plans/katvad-flow-zscore-option-a.md``) says it means:

* ``mu, sigma`` come from the **train windows only**, overlap counted, exactly the
  rows ``L_KIP_rec`` averages over;
* every rebuilt ``e_O`` is ``((s - mu) / sigma) @ M`` with v1's **own** ``M``, and
  the raw stats the KNN motion key reads are carried over byte-identical;
* the hard stops fire: G0 (v1 ``e_O`` != stats @ M), a dead stat, ``dst == src``,
  a directory already fitted on another split, a missing projection;
* KIP-off is inert to ``loss.lambda_rec`` and the flow dir, which is what lets
  the plan reuse the KIP-off arms instead of re-running them.

No data, no downloads, no GPU: synthetic caches under ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.config import load_config
from core.data.collate import collate_variable_length
from core.data.windows import FeatureSlicer, Window
from core.eda import features
from core.flow import raft_extract, zscore, zscore_cache
from core.tests.fixtures import FixtureLayout, build_fixture
from core.tools.feature_cache import part_path
from core.train import build_trainer, epoch_permutation

DATASET = "SYN"
ROWS = 64
SEED = 11
STATS = constants.FLOW_STATS_SUFFIX


def _write_corpus(
    data_dir: Path, train: dict[str, int], windows: dict[str, tuple[str, int, int]] | None
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, object] = {
        constants.LABELS_TRAIN_FILENAME: train,
        constants.FRAME_LABELS_TEST_FILENAME: {"t0": [0, 1, 0]},
        constants.DEFS_FILENAME: ["Normal", "CarAccident"],
        constants.META_FILENAME: {},
    }
    if windows is not None:
        payloads[constants.WINDOWS_FILENAME] = {
            wid: {"source": src, "start": a, "end": b} for wid, (src, a, b) in windows.items()
        }
    for name, payload in payloads.items():
        (data_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def _write_v1(src_root: Path, sources: dict[str, np.ndarray]) -> np.ndarray:
    """A v1 cache exactly as raft_extract lays it out; returns the projection."""
    projection = raft_extract.get_or_create_projection(src_root)
    flow_dir = src_root / DATASET
    flow_dir.mkdir(parents=True, exist_ok=True)
    for sid, raw in sources.items():
        raw32 = raw.astype(np.float32)
        np.save(flow_dir / f"{sid}{STATS}", raw32)
        np.save(flow_dir / f"{sid}.npy", (raw32 @ projection).astype(np.float32))
    return projection


def _raw(rng: np.random.Generator, rows: int = ROWS, scale: float = 1.0) -> np.ndarray:
    # Heterogeneous scales, like the real stats (mag_max ~ 23 px vs histogram ~ 0.05).
    widths = np.linspace(0.05, 20.0, constants.FLOW_STATS_DIM)
    return rng.normal(3.0, 1.0, size=(rows, constants.FLOW_STATS_DIM)) * widths * scale


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(SEED)
    sources = {"a": _raw(rng), "b": _raw(rng), "t0": _raw(rng, scale=50.0)}
    _write_v1(tmp_path / "v1", sources)
    _write_corpus(tmp_path / "data", {"a": 1, "b": 0}, windows=None)
    return {"data": tmp_path / "data", "src": tmp_path / "v1", "dst": tmp_path / "v2"}


def _build(paths: dict[str, Path], force: bool = False) -> dict[str, Any]:
    return zscore_cache.build_zscore_cache(
        paths["data"], DATASET, paths["src"], paths["dst"], force=force
    )


class TestFit:
    def test_moments_come_from_train_items_only(self, corpus: dict[str, Path]) -> None:
        # t0 is a test source with 50x the scale: if it leaked into the fit, mu moves.
        _build(corpus)
        fitted = zscore.load_zscore_stats(
            corpus["dst"] / DATASET / constants.FLOW_ZSCORE_STATS_FILENAME
        )
        src = corpus["src"] / DATASET
        pooled = np.concatenate(
            [np.load(src / f"a{STATS}"), np.load(src / f"b{STATS}")]
        ).astype(np.float64)
        np.testing.assert_allclose(fitted.mean, pooled.mean(axis=0), rtol=1e-9)
        np.testing.assert_allclose(fitted.std, pooled.std(axis=0), rtol=1e-5)
        assert fitted.n_items == 2 and fitted.n_rows == 2 * ROWS

    def test_windowed_moments_count_overlapping_rows_twice(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        raw = _raw(rng, rows=6)
        _write_v1(tmp_path / "v1", {"s": raw})
        windows = {"s__w000": ("s", 0, 4), "s__w001": ("s", 2, 6)}
        _write_corpus(tmp_path / "data", {"s__w000": 1, "s__w001": 0}, windows)
        slicer = FeatureSlicer({wid: Window(*spec) for wid, spec in windows.items()})
        fitted = zscore_cache.fit_train_zscore(
            tmp_path / "v1" / DATASET, sorted(windows), slicer, "v1"
        )
        rows = np.load(tmp_path / "v1" / DATASET / f"s{STATS}").astype(np.float64)
        pooled = np.concatenate([rows[0:4], rows[2:6]])
        np.testing.assert_allclose(fitted.mean, pooled.mean(axis=0), rtol=1e-9)
        assert fitted.n_rows == 8

    def test_a_dead_stat_raises(self) -> None:
        acc = zscore.MomentAccumulator(constants.FLOW_STATS_DIM)
        rows = np.random.default_rng(SEED).normal(size=(10, constants.FLOW_STATS_DIM))
        rows[:, 2] = 4.0
        acc.add(rows)
        with pytest.raises(ValueError, match="mag_max"):
            zscore.fit_zscore(acc, "sha", "v1")

    def test_digest_ignores_order(self) -> None:
        assert zscore.ids_digest(["b", "a"]) == zscore.ids_digest(["a", "b"])


class TestRebuild:
    def test_every_file_is_standardized_then_projected_with_v1s_matrix(
        self, corpus: dict[str, Path]
    ) -> None:
        manifest = _build(corpus)
        src, dst = corpus["src"] / DATASET, corpus["dst"] / DATASET
        fitted = zscore.load_zscore_stats(dst / constants.FLOW_ZSCORE_STATS_FILENAME)
        projection = raft_extract.load_projection(
            corpus["src"] / constants.FLOW_PROJECTION_FILENAME
        )
        for sid in ("a", "b", "t0"):  # the test source is rebuilt too
            raw = np.load(src / f"{sid}{STATS}")
            expected = (zscore.standardize(raw, fitted) @ projection).astype(np.float32)
            np.testing.assert_allclose(np.load(dst / f"{sid}.npy"), expected, rtol=1e-5)
        assert manifest["coverage"] == {"sources": 3, "written_this_run": 3}

    def test_raw_stats_and_projection_are_copied_byte_identical(
        self, corpus: dict[str, Path]
    ) -> None:
        _build(corpus)
        for sid in ("a", "b", "t0"):
            src_bytes = (corpus["src"] / DATASET / f"{sid}{STATS}").read_bytes()
            assert (corpus["dst"] / DATASET / f"{sid}{STATS}").read_bytes() == src_bytes
        name = constants.FLOW_PROJECTION_FILENAME
        assert (corpus["dst"] / name).read_bytes() == (corpus["src"] / name).read_bytes()

    def test_resume_redoes_a_partial_write_and_skips_finished_files(
        self, corpus: dict[str, Path]
    ) -> None:
        _build(corpus)
        target = corpus["dst"] / DATASET / "b.npy"
        target.unlink()
        part_path(target).write_bytes(b"truncated")
        manifest = _build(corpus)
        assert manifest["coverage"]["written_this_run"] == 1
        assert target.is_file() and not part_path(target).exists()


class TestHardStops:
    def test_g0_names_the_file_whose_e_o_is_not_its_stats_at_m(
        self, corpus: dict[str, Path]
    ) -> None:
        path = corpus["src"] / DATASET / "b.npy"
        np.save(path, np.load(path) + 1.0)
        with pytest.raises(ValueError, match="G0 failed for b"):
            _build(corpus)

    def test_orphan_stats_without_e_o_raise(self, corpus: dict[str, Path]) -> None:
        (corpus["src"] / DATASET / "t0.npy").unlink()
        with pytest.raises(FileNotFoundError, match="coverage 2/3"):
            _build(corpus)

    def test_dst_equal_to_src_raises(self, corpus: dict[str, Path]) -> None:
        with pytest.raises(ValueError, match="must differ"):
            zscore_cache.build_zscore_cache(
                corpus["data"], DATASET, corpus["src"], corpus["src"]
            )

    def test_a_directory_fitted_on_another_split_is_refused(
        self, corpus: dict[str, Path]
    ) -> None:
        _build(corpus)
        _write_corpus(corpus["data"], {"a": 1}, windows=None)
        with pytest.raises(ValueError, match="different split"):
            _build(corpus)
        assert _build(corpus, force=True)["zscore"]["n_items"] == 1

    def test_missing_projection_raises(self, corpus: dict[str, Path]) -> None:
        (corpus["src"] / constants.FLOW_PROJECTION_FILENAME).unlink()
        with pytest.raises(FileNotFoundError, match="projection"):
            _build(corpus)

    def test_a_failed_hard_gate_raises_after_writing_the_manifest(
        self, corpus: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(constants, "FLOW_ZSCORE_G2C_ROUNDTRIP_BAND", (5.0, 6.0))
        with pytest.raises(RuntimeError, match="G2c_roundtrip"):
            _build(corpus)
        path = corpus["dst"] / DATASET / constants.FLOW_ZSCORE_MANIFEST_FILENAME
        assert json.loads(path.read_text(encoding="utf-8"))["passed"] is False


class TestGatesAndLambda:
    def test_gates_pass_and_lambda_is_one_over_v_on_the_new_cache(
        self, tmp_path: Path
    ) -> None:
        # Enough independent rows that V's prediction (~1) is tight.
        rng = np.random.default_rng(SEED)
        sources = {f"s{i}": _raw(rng, rows=400) for i in range(6)}
        _write_v1(tmp_path / "v1", sources)
        _write_corpus(tmp_path / "data", {sid: i % 2 for i, sid in enumerate(sources)}, None)
        manifest = zscore_cache.build_zscore_cache(
            tmp_path / "data", DATASET, tmp_path / "v1", tmp_path / "v2"
        )
        assert manifest["passed"] is True
        gates = manifest["gates"]
        assert all(g["passed"] for g in gates.values())
        variance = manifest["target"]["mse_global_mean_predictor"]
        assert variance == pytest.approx(1.0, abs=0.1)
        assert manifest["lambda_rec"] == round(1.0 / variance, 4)
        assert manifest["zscore"]["histogram_block_share"] == pytest.approx(16 / 23)

    def test_eda_predicts_the_round_trip_from_standardized_stats_on_v2(
        self, corpus: dict[str, Path]
    ) -> None:
        # Raw stats stay raw in v2; predicting from them would report ~100x off.
        _build(corpus)
        report = features.flow_stats(corpus["dst"] / DATASET, ["a", "b"])
        assert report["target_normalized"] is True
        target = report["target"]
        ratio = target["predicted_second_moment_from_raw"] / target["mse_zero_predictor"]
        assert ratio == pytest.approx(1.0, abs=0.15)
        assert max(abs(m) for m in report["standardized"]["per_dim_mean"]) < 1e-9

    def test_v1_reports_are_unchanged_without_zscore_stats(
        self, corpus: dict[str, Path]
    ) -> None:
        report = features.flow_stats(corpus["src"] / DATASET, ["a", "b"])
        assert report["target_normalized"] is False and "standardized" not in report


@pytest.fixture(scope="module")
def trainer_fixture(tmp_path_factory: pytest.TempPathFactory) -> FixtureLayout:
    return build_fixture(tmp_path_factory.mktemp("zscore_kip_off"))


class TestKipOffIsInert:
    """The plan reuses KIP-off arms: that is only sound if these knobs do nothing."""

    def test_lambda_rec_and_flow_dir_do_not_reach_a_kip_off_run(
        self, trainer_fixture: FixtureLayout, tmp_path: Path
    ) -> None:
        torch.set_num_threads(1)
        cfg = load_config(
            None,
            ["train.batch_size=4", "train.device=cpu", "train.num_epochs=1",
             "kip.enabled=false"],
        )
        trainer = build_trainer(
            cfg=cfg,
            output_dir=tmp_path / "off",
            data_dir=trainer_fixture.data_dir,
            clip_dir=trainer_fixture.clip_dir,
            flow_dir=tmp_path / "does-not-exist",
            knn_cache_path=trainer_fixture.knn_cache_path,
            text_encoder="stub",
        )
        assert trainer.dataset.flow_dir is None
        order = epoch_permutation(cfg.train.seed, 0, len(trainer.dataset))
        batch = collate_variable_length([trainer.dataset[i] for i in order[:4]])
        # NaN-poison the weight: if it reached the objective, total would be NaN.
        # (compute_losses is not bitwise repeatable, so equality would be flaky.)
        trainer.cfg = dataclasses.replace(
            cfg, loss=dataclasses.replace(cfg.loss, lambda_rec=float("nan"))
        )
        trainer.model.eval()
        with torch.no_grad():
            losses = trainer.compute_losses(batch)
        assert "kip_rec" not in losses
        assert bool(torch.isfinite(losses["total"]))
