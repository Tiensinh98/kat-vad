"""Zero-shot TAD / DADA-2000 for SA² (``core/docs/final/ZS_TAD_DADA.md``): the row-frame
archives, ``clip_windows packed``, ``--patch-dir`` on ``core.evaluate`` / ``dada_whole_eval``,
and the read-out."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import constants, evaluate, train
from core.data.dataset import FeatureEvalDataset
from core.data.stp_patches import EvalPatches
from core.tests.test_stp import BASE, D, FakeEncoder, K, _jpegs, _stp_fixture
from core.tools import clip_windows, dada_whole_eval, zs_frames, zs_readout
from core.tools.dota_cap import OK


def _cls_rows(paths: list[Path]) -> np.ndarray:
    """The 1 x 1 cell FakeEncoder gives on ``paths`` -- what a CLIP _ncc cache would hold."""
    from core.data.video_io import read_images

    frames = torch.from_numpy(read_images(paths)).permute(0, 3, 1, 2).float() / 255.0
    out: np.ndarray = clip_windows.encode_frame_windows(frames, FakeEncoder(), (1,),
                                                        constants.STP_CELLS_RESIZED)[:, 0]
    return out


# --------------------------------------------------------------------------- DADA plan


class TestDadaPlan:
    LISTING = "\n".join([
        "Path = DADA2000/1/001/images",
        "Folder = +",
        *[f"Path = DADA2000/1/001/images/{i:04d}.png" for i in range(1, 18)],
        "Path = DADA2000/1/001/maps/0001.png",
        *[f"Path = DADA2000\\10\\002\\images\\{i:04d}.png" for i in range(1, 9)],
    ])

    def test_folder_of_an_id(self) -> None:
        assert zs_frames.dada_folder("t10_v002") == "DADA2000/10/002/images"
        with pytest.raises(ValueError, match="not a DADA"):
            zs_frames.dada_folder("Normal_012")

    def test_listing_keeps_only_images_and_sorts(self) -> None:
        images = zs_frames.listing_images(self.LISTING)
        assert sorted(images) == ["DADA2000/1/001/images", "DADA2000/1/001/maps",
                                  "DADA2000/10/002/images"]
        assert images["DADA2000/1/001/images"][0] == "0001.png"
        assert len(images["DADA2000/1/001/images"]) == 17

    def test_plan_takes_every_stride_th_frame_and_checks_the_census(self) -> None:
        images = zs_frames.listing_images(self.LISTING)
        plan = zs_frames.dada_plan(images, {"t1_v001": 17, "t10_v002": 9},
                                   ["t1_v001", "t10_v002", "t2_v003"], 8)
        ok = plan["t1_v001"]
        assert ok["reason"] == OK and ok["native_frames"] == 17
        assert ok["members"] == [f"DADA2000/1/001/images/{n:04d}.png" for n in (1, 9, 17)]
        assert plan["t10_v002"]["reason"] == zs_frames.CENSUS_MISMATCH  # 8 listed, census 9
        assert plan["t2_v003"]["reason"] == zs_frames.NO_FRAMES

    def test_cli_refuses_an_unplannable_source(self, tmp_path: Path) -> None:
        (tmp_path / "l.txt").write_text(self.LISTING)
        (tmp_path / "c.json").write_text(json.dumps({"t1_v001": 17}))
        (tmp_path / "ids.txt").write_text("t1_v001\nt2_v003\n")
        args = ["dada-plan", "--listing", str(tmp_path / "l.txt"), "--census",
                str(tmp_path / "c.json"), "--ids-file", str(tmp_path / "ids.txt"),
                "--out-list", str(tmp_path / "m.txt"), "--out-plan", str(tmp_path / "p.json")]
        with pytest.raises(ValueError, match="cannot be planned"):
            zs_frames.main(args)
        (tmp_path / "ids.txt").write_text("t1_v001\n")
        zs_frames.main(args)
        assert (tmp_path / "m.txt").read_text().splitlines()[1] == "DADA2000/1/001/images/0009.png"


# ------------------------------------------------------------------ pack / verify / windows


def _farm(tmp_path: Path, n: int, stride: int, prestrided: bool) -> tuple[Path, Path, Path]:
    """A one-video farm and the CLIP rows of its ``frames[::stride]``."""
    farm, clip = tmp_path / "farm", tmp_path / "clip"
    paths = _jpegs(farm / "vid_a", n)
    rows = paths[::stride]
    if prestrided:
        for p in set(paths) - set(rows):
            p.unlink()
    clip.mkdir()
    np.save(clip / "vid_a.npy", _cls_rows(rows))
    ids = tmp_path / "ids.txt"
    ids.write_text("vid_a\n")
    return farm, clip, ids


class TestPackVerifyWindows:
    @pytest.mark.parametrize("prestrided", [False, True])
    def test_pack_verify_packed_end_to_end(self, tmp_path: Path, prestrided: bool) -> None:
        farm, clip, ids = _farm(tmp_path, 10, 4, prestrided)
        arc = tmp_path / "arc"
        common = ["--ids-file", str(ids), "--clip-dir", str(clip), "--archive-dir", str(arc),
                  "--stride", "4"]
        zs_frames.main(["pack", "--frames-dir", str(farm), "--report", str(tmp_path / "p.json"),
                        *(["--prestrided"] if prestrided else []), *common])
        rep = json.loads((tmp_path / "p.json").read_text())["vid_a"]
        assert rep["reason"] == OK and rep["frames"] == 3  # frames 0, 4, 8
        zs_frames.main(["verify", "--dataset", "TADX", "--report", str(tmp_path / "v.json"),
                        *common])
        manifest = json.loads((arc / constants.ZS_FRAMES_MANIFEST).read_text())
        assert (manifest["dataset"], manifest["stride"], manifest["frames"]) == ("TADX", 4, 3)

        args = clip_windows.build_arg_parser().parse_args([
            "packed", "--archive-dir", str(arc), "--work-dir", str(tmp_path / "w"),
            "--clip-dir", str(clip), "--ids-file", str(ids), "--report", str(tmp_path / "r.json"),
            "--grids", "1", "7", "--cells", "resized224", "--output-dir", str(tmp_path / "win"),
            "--device", "cpu",
        ])
        clip_windows.run_packed(args, FakeEncoder())
        gate = json.loads((tmp_path / "r.json").read_text())["vid_a"]
        assert gate["reason"] == OK and gate["global_min_cos"] > 0.9999, gate
        assert np.load(tmp_path / "win" / "vid_a.npy").shape == (3, 50 * D)
        win_manifest = json.loads(
            (tmp_path / "win" / constants.VIDEO_MANIFEST_FILENAME).read_text()
        )
        assert (win_manifest["dataset"], win_manifest["stride"]) == ("TADX", 4)
        assert win_manifest["cells"] == constants.STP_CELLS_RESIZED
        assert not (tmp_path / "w" / "vid_a").exists()

    def test_pack_refuses_a_row_mismatch(self, tmp_path: Path) -> None:
        farm, clip, ids = _farm(tmp_path, 10, 4, False)
        with pytest.raises(SystemExit, match="not packed"):  # stride 3 -> 4 frames vs 3 rows
            zs_frames.main(["pack", "--frames-dir", str(farm), "--report", str(tmp_path / "p.json"),
                            "--ids-file", str(ids), "--clip-dir", str(clip),
                            "--archive-dir", str(tmp_path / "arc"), "--stride", "3"])
        rep = json.loads((tmp_path / "p.json").read_text())["vid_a"]
        assert rep == {"reason": zs_frames.ROWS_MISMATCH, "frames": 4, "clip_rows": 3}
        assert not (tmp_path / "arc" / "vid_a.zip").exists()

    def test_pack_resumes_a_complete_archive(self, tmp_path: Path) -> None:
        farm, _, _ = _farm(tmp_path, 10, 4, False)
        frames = zs_frames.row_frames(farm / "vid_a", None, 4, False)
        first = zs_frames.pack_video("vid_a", frames, 3, tmp_path / "arc")
        again = zs_frames.pack_video("vid_a", frames, 3, tmp_path / "arc")
        assert first["reason"] == again["reason"] == OK and again.get("resumed") is True

    def test_verify_without_archive_writes_no_manifest(self, tmp_path: Path) -> None:
        _, clip, ids = _farm(tmp_path, 10, 4, False)
        with pytest.raises(SystemExit):
            zs_frames.main(["verify", "--dataset", "X", "--report", str(tmp_path / "v.json"),
                            "--ids-file", str(ids), "--clip-dir", str(clip),
                            "--archive-dir", str(tmp_path / "arc")])
        assert not (tmp_path / "arc" / constants.ZS_FRAMES_MANIFEST).exists()

    def test_packed_needs_the_manifest(self, tmp_path: Path) -> None:
        args = clip_windows.build_arg_parser().parse_args([
            "packed", "--archive-dir", str(tmp_path), "--work-dir", str(tmp_path / "w"),
            "--clip-dir", str(tmp_path), "--ids-file", str(tmp_path / "i"),
            "--report", str(tmp_path / "r.json"),
        ])
        with pytest.raises(FileNotFoundError, match="zs_frames verify"):
            clip_windows.run_packed(args, FakeEncoder())


# --------------------------------------------------------------------------- eval wiring


class TestEvalPatchDir:
    def test_eval_dataset_appends_patches_and_default_is_unchanged(self, tmp_path: Path) -> None:
        data_dir, a2_dir, patch_dir = _stp_fixture(tmp_path)
        plain = FeatureEvalDataset(data_dir, a2_dir)
        with_patches = FeatureEvalDataset(data_dir, a2_dir, patch_dir)
        a, b = plain[0], with_patches[0]
        assert a["v_feat"].shape[-1] == BASE and b["v_feat"].shape[-1] == BASE + K * D
        assert torch.equal(b["v_feat"][:, :BASE], a["v_feat"])

    def test_dada_source_rows(self, tmp_path: Path) -> None:
        _, a2_dir, patch_dir = _stp_fixture(tmp_path)
        vid = sorted(p.stem for p in a2_dir.glob("*.npy"))[0]
        rows = dada_whole_eval.source_rows(a2_dir, vid, None)
        assert rows.dtype == np.float32 and rows.shape[1] == BASE
        assert dada_whole_eval.source_rows(a2_dir, vid, EvalPatches(patch_dir)).shape[1] == \
            BASE + K * D

    def test_evaluate_scores_an_sa2_checkpoint_only_with_its_patches(self, tmp_path: Path) -> None:
        data_dir, a2_dir, patch_dir = _stp_fixture(tmp_path)
        run = tmp_path / "r4"
        train.main([
            "--data-dir", str(data_dir), "--clip-dir", str(a2_dir), "--patch-dir", str(patch_dir),
            "--output-dir", str(run), "--text-encoder", "stub",
            "--set", "train.num_epochs=1", "--set", "train.batch_size=4",
            "--set", "train.device=cpu", "--set", "kip.enabled=false",
            "--set", f"v2.motion={constants.V5_ENCODER}", "--set", "model.motion_dim=384",
            *[x for k, v in constants.STP_ARMS["R4"].items()
              for x in ("--set", f"stp.{k}={str(v).lower()}")],
        ])
        base = ["--ckpt", str(run / train.CHECKPOINT_LAST), "--data-dir", str(data_dir),
                "--clip-dir", str(a2_dir), "--text-encoder", "stub", "--set", "train.device=cpu"]
        evaluate.main([*base, "--patch-dir", str(patch_dir), "--save-scores",
                       "--output-dir", str(tmp_path / "ev")])
        assert json.loads((tmp_path / "ev" / evaluate.RESULTS_FILENAME).read_text())["num_videos"]
        with pytest.raises(ValueError, match="--patch-dir"):
            evaluate.main([*base, "--output-dir", str(tmp_path / "ev_bad")])
        with pytest.raises(ValueError, match="this run needs"):
            evaluate.main([*base, "--patch-dir", str(patch_dir), "--patch-stride", "3",
                           "--output-dir", str(tmp_path / "ev_bad2")])


# --------------------------------------------------------------------------- read-out


def _tad_run(run_dir: Path, shift: float, rng: np.random.Generator) -> None:
    scores = run_dir / evaluate.SCORES_DIRNAME
    scores.mkdir(parents=True)
    for i in range(constants.V2_TADZ_VIDEOS):
        gt = np.zeros(12, dtype=np.float32)
        if i < constants.V2_TADZ_ABNORMAL:
            gt[4:8] = 1.0
        score = rng.random(12) + shift * gt
        np.savez(scores / f"v{i:03d}.npz", score=score.astype(np.float32), gt=gt)


def _dada_run(arm_dir: Path, aucs: dict[str, float], per_run: dict[str, float]) -> None:
    arm_dir.mkdir(parents=True)
    ci = {"mean": 0.7, "low": 0.6, "high": 0.8, "clips": len(aucs), "clusters": len(aucs)}
    (arm_dir / dada_whole_eval.CLIP_AUCS_JSON).write_text(json.dumps(aucs))
    (arm_dir / dada_whole_eval.READOUT_JSON).write_text(json.dumps({
        "per_run_macro": per_run, "macro": ci, "micro": 0.7, "video_auc": None,
        "monotone_ruler_macro": ci, "middle_ruler_macro": ci,
    }))


class TestReadout:
    def test_end_to_end(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        root, v2 = tmp_path / "zs", tmp_path / "v2"
        _tad_run(root / "eval" / "tad" / "W0" / "s2024", 0.3, rng)
        _tad_run(root / "eval" / "tad" / "R4" / "s2024", 3.0, rng)  # R4 clearly better
        for s in constants.NEXAR_RUN_SEEDS:
            _tad_run(v2 / "eval" / "tad" / "A2" / f"s{s}", 0.3, rng)
        ids = [f"d{i}" for i in range(40)]
        _dada_run(root / "eval" / "dada" / "W0", dict.fromkeys(ids, 0.7), {"s2024": 0.7})
        _dada_run(root / "eval" / "dada" / "R4", dict.fromkeys(ids, 0.7), {"s2024": 0.7})
        _dada_run(v2 / "eval" / "dada" / "A2", dict.fromkeys(ids, 0.7),
                  {f"s{s}": 0.7 for s in constants.NEXAR_RUN_SEEDS})
        r = zs_readout.run(
            zs_readout.build_arg_parser().parse_args(["--root", str(root), "--v2-root", str(v2)])
        )
        tad, dada = r["sets"]["tad"], r["sets"]["dada"]
        assert tad["primary"]["sentence"] == "gain shown"
        assert dada["primary"]["sentence"] == "no difference shown"
        assert dada["primary"]["delta"]["mean"] == 0.0
        assert dada["reproduction"]["flag"] is False
        assert "w0_minus_a2_same_seed" in tad and "w0_minus_a2_same_seed" not in dada
        md = (root / "REPORTS" / "zs_readout.md").read_text()
        assert "PRELIMINARY (n = 1 seed)" in md and "R4 - W0" in md

    def test_missing_eval_raises(self, tmp_path: Path) -> None:
        args = zs_readout.build_arg_parser().parse_args(["--root", str(tmp_path),
                                                          "--v2-root", str(tmp_path)])
        with pytest.raises(FileNotFoundError):
            zs_readout.run(args)
