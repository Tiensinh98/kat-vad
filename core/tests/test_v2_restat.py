"""``restat_rows`` / ``build_v2_inputs restat`` (``KIM_RUN.md`` §42): an A2 cache baked with one
set of statistics re-expressed with another equals a fresh bake with the second set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import constants
from core.data import v2_inputs
from core.tools import build_v2_inputs

ENC = constants.V5_ENCODER
OFF = constants.V2_OFF
D_X, D_V = 16, 6
RESTAT_ATOL = constants.V2_RESTAT_GATE_ATOL  # the notebook gate's tolerance


def _corpus(seed: int, n: int, shift: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    clips, motions = {}, {}
    for i in range(n):
        t = int(rng.integers(5, 12))
        clips[f"v{seed}_{i}"] = rng.normal(0.0, 1.0, (t, D_X)).astype(np.float32)
        motions[f"v{seed}_{i}"] = (rng.normal(shift, 1.0 + shift, (t, D_V))).astype(np.float32)
    return clips, motions


def _stats(seed: int, shift: float) -> v2_inputs.V2Stats:
    clips, motions = _corpus(seed, 8, shift)
    return v2_inputs.fit_stats(clips, motions, OFF, ENC)


class TestRestatRows:
    def test_equals_a_fresh_bake_with_the_new_statistics(self) -> None:
        old, new = _stats(1, 0.0), _stats(2, 2.5)
        clips, motions = _corpus(3, 4, 1.0)
        for v in clips:
            baked_old = v2_inputs.bake_rows(clips[v], motions[v], old)
            want = v2_inputs.bake_rows(clips[v], motions[v], new)
            got = v2_inputs.restat_rows(baked_old, old, new)
            assert got.dtype == np.float32 and got.shape == want.shape
            assert np.array_equal(got[:, :D_X], baked_old[:, :D_X])  # CLIP columns copied
            assert np.max(np.abs(got - want)) < RESTAT_ATOL

    def test_same_statistics_is_identity(self) -> None:
        st = _stats(1, 0.0)
        clips, motions = _corpus(4, 2, 0.5)
        for v in clips:
            baked = v2_inputs.bake_rows(clips[v], motions[v], st)
            assert np.max(np.abs(v2_inputs.restat_rows(baked, st, st) - baked)) < RESTAT_ATOL

    def test_crn_is_refused(self) -> None:
        clips, motions = _corpus(5, 6, 0.0)
        crn = next(c for c in constants.V2_CRN_CHOICES if c != OFF)
        with_crn = v2_inputs.fit_stats(clips, motions, crn, ENC)
        rows = np.zeros((4, D_X + D_V), dtype=np.float32)
        with pytest.raises(ValueError, match="CRN off"):
            v2_inputs.restat_rows(rows, with_crn, _stats(1, 0.0))

    def test_no_motion_is_refused(self) -> None:
        clips, _ = _corpus(6, 4, 0.0)
        plain = v2_inputs.fit_stats(clips, None, OFF, OFF)
        with pytest.raises(ValueError, match="motion stream"):
            v2_inputs.restat_rows(np.zeros((3, D_X), dtype=np.float32), _stats(1, 0.0), plain)

    def test_rows_without_a_motion_block_are_refused(self) -> None:
        st = _stats(1, 0.0)
        with pytest.raises(ValueError, match="motion block"):
            v2_inputs.restat_rows(np.zeros((3, D_V), dtype=np.float32), st, st)


def _write(folder: Path, rows: dict[str, np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for v, r in rows.items():
        np.save(folder / f"{v}.npy", r)


def _ids(path: Path, ids: list[str]) -> Path:
    path.write_text("".join(f"{v}\n" for v in ids))
    return path


class TestRestatCli:
    def test_restat_of_an_old_bake_equals_a_direct_fit_ids_bake(self, tmp_path: Path) -> None:
        clips_a, motions_a = _corpus(10, 6, 0.0)   # "Nexar": fits the old statistics
        clips_b, motions_b = _corpus(11, 6, 3.0)   # "ACCIDENT": fits the new ones, is re-read
        clips, motions = {**clips_a, **clips_b}, {**motions_a, **motions_b}
        _write(tmp_path / "clip", clips)
        _write(tmp_path / "video", motions)
        every = _ids(tmp_path / "all.txt", sorted(clips))

        def fit(name: str, train: list[str]) -> Path:
            out = tmp_path / name
            build_v2_inputs.main([
                "fit-ids", "--train-ids-file", str(_ids(tmp_path / f"{name}.txt", train)),
                "--ids-file", str(every), "--clip-dir", str(tmp_path / "clip"),
                "--video-dir", str(tmp_path / "video"), "--crn", OFF, "--motion", ENC,
                "--out-dir", str(out)])
            return out

        old_dir, new_dir = fit("old", sorted(clips_a)), fit("new", sorted(clips_b))
        moved = tmp_path / "moved"
        build_v2_inputs.main(["restat", "--from-dir", str(old_dir), "--stats-dir", str(new_dir),
                              "--ids-file", str(_ids(tmp_path / "b.txt", sorted(clips_b))),
                              "--out-dir", str(moved)])
        for v in clips_b:
            got, want = np.load(moved / f"{v}.npy"), np.load(new_dir / f"{v}.npy")
            assert np.max(np.abs(got - want)) < RESTAT_ATOL
        m = json.loads((moved / constants.V2_INPUT_MANIFEST_FILENAME).read_text())
        new_m = json.loads((new_dir / constants.V2_INPUT_MANIFEST_FILENAME).read_text())
        old_m = json.loads((old_dir / constants.V2_INPUT_MANIFEST_FILENAME).read_text())
        assert m["mode"] == "restat" and (m["crn"], m["motion"]) == (OFF, ENC)
        assert m["train_ids_sha1"] == new_m["train_ids_sha1"] != old_m["train_ids_sha1"]
        assert m["restat_from_train_ids_sha1"] == old_m["train_ids_sha1"]
        assert m["baked_ids"] == len(clips_b)
        st = v2_inputs.load_stats(moved)
        assert st.c == pytest.approx(new_m["c"])
        v2_inputs.check_input_manifest(moved, OFF, ENC)  # train / evaluate accept it

    def test_empty_source_is_refused(self, tmp_path: Path) -> None:
        clips, motions = _corpus(12, 3, 0.0)
        _write(tmp_path / "clip", clips)
        _write(tmp_path / "video", motions)
        every = _ids(tmp_path / "all.txt", sorted(clips))
        build_v2_inputs.main(["fit-ids", "--train-ids-file", str(every), "--ids-file", str(every),
                              "--clip-dir", str(tmp_path / "clip"), "--video-dir",
                              str(tmp_path / "video"), "--crn", OFF, "--motion", ENC,
                              "--out-dir", str(tmp_path / "st")])
        (tmp_path / "empty").mkdir()
        for f in (tmp_path / "st").glob("*.json"):
            (tmp_path / "empty" / f.name).write_text(f.read_text())
        for f in (tmp_path / "st").glob("*.npz"):
            (tmp_path / "empty" / f.name).write_bytes(f.read_bytes())
        with pytest.raises(SystemExit, match="nothing to restat"):
            build_v2_inputs.main(["restat", "--from-dir", str(tmp_path / "empty"),
                                  "--stats-dir", str(tmp_path / "st"),
                                  "--out-dir", str(tmp_path / "out")])
