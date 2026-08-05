"""Tests for the download CLI (all transfers mocked — no network)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from core import constants
from core.tools import download


@pytest.fixture()
def ckpt_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(constants, "CKPT_ROOT", tmp_path / "ckpts")
    return tmp_path / "ckpts"


class TestCkpt:
    def test_downloads_and_verifies(
        self, ckpt_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_fetch(file_id: str, output: Path) -> None:
            assert file_id == constants.LAGOVAD_BEST_CKPT_GDRIVE_ID
            output.write_bytes(b"weights")

        monkeypatch.setattr(download, "_gdown_fetch", fake_fetch)
        target = download.download_ckpt(sha256=None)
        assert target.exists()
        assert target.read_bytes() == b"weights"

    def test_checksum_mismatch_raises(
        self, ckpt_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            download, "_gdown_fetch", lambda fid, out: out.write_bytes(b"weights")
        )
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            download.download_ckpt(sha256="0" * 64)

    def test_skip_if_exists(
        self, ckpt_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt_root.mkdir(parents=True)
        target = ckpt_root / constants.LAGOVAD_BEST_CKPT_FILENAME
        target.write_bytes(b"old")

        def boom(file_id: str, output: Path) -> None:
            raise AssertionError("should not download")

        monkeypatch.setattr(download, "_gdown_fetch", boom)
        assert download.download_ckpt() == target
        assert target.read_bytes() == b"old"

    def test_dry_run_touches_nothing(self, ckpt_root: Path) -> None:
        target = download.download_ckpt(dry_run=True)
        assert not target.exists()

    def test_zip_wrapped_download_is_unwrapped(
        self, ckpt_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_fetch(file_id: str, output: Path) -> None:
            with zipfile.ZipFile(output, "w") as zf:
                zf.writestr("best.ckpt", b"inner-weights")

        monkeypatch.setattr(download, "_gdown_fetch", fake_fetch)
        target = download.download_ckpt()
        assert target.read_bytes() == b"inner-weights"
        assert not target.with_name(target.name + ".unwrapped").exists()

    def test_torch_style_zip_is_left_alone(self, tmp_path: Path) -> None:
        # a real torch checkpoint is a zip with entries under a subdirectory
        target = tmp_path / "best.ckpt"
        with zipfile.ZipFile(target, "w") as zf:
            zf.writestr("best/data.pkl", b"pickle")
            zf.writestr("best/version", b"3")
        before = target.read_bytes()
        download._unwrap_zip_wrapper(target)
        assert target.read_bytes() == before

    def test_multi_checkpoint_zip_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "best.ckpt"
        with zipfile.ZipFile(target, "w") as zf:
            zf.writestr("a.ckpt", b"a")
            zf.writestr("b.ckpt", b"b")
        with pytest.raises(RuntimeError, match="multiple checkpoints"):
            download._unwrap_zip_wrapper(target)

    def test_non_zip_download_untouched(self, tmp_path: Path) -> None:
        target = tmp_path / "best.ckpt"
        target.write_bytes(b"weights")
        download._unwrap_zip_wrapper(target)
        assert target.read_bytes() == b"weights"


class TestOtherTargets:
    def test_clip_passes_pinned_revision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            download, "_hf_snapshot_fetch", lambda repo, rev: calls.append((repo, rev))
        )
        download.download_clip()
        assert calls == [(constants.CLIP_MODEL_NAME, constants.CLIP_MODEL_REVISION)]

    def test_raft_passes_weights_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(download, "_raft_weights_fetch", calls.append)
        download.download_raft()
        assert calls == [constants.RAFT_WEIGHTS_NAME]

    def test_cli_all_dry_run(
        self, ckpt_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("_gdown_fetch", "_hf_snapshot_fetch", "_raft_weights_fetch"):
            monkeypatch.setattr(
                download, name, lambda *a, **k: pytest.fail("dry-run must not fetch")
            )
        download.main(["all", "--dry-run"])


class TestChecksum:
    def test_sha256_of(self, tmp_path: Path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"abc")
        assert download.sha256_of(path) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
