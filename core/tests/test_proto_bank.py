"""§61: class prototypes on ``v^t`` in the main score, KIM off (rule ``KIM_RUN.md`` §61)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from core import constants
from core.config import Config, load_config, validate_proto
from core.data.collate import padding_mask
from core.inference import load_model_for_scoring
from core.models.kat_vad import KATVAD
from core.models.proto_bank import PrototypeBank, bottomk_steps
from core.tests.test_kim import CPU, D, _fixture, _host, _rows
from core.train import build_trainer, set_global_seed


def _cfg(score: bool = True, loss: bool = True, bottomk: bool = False) -> Config:
    cfg = _host()
    cfg.proto.score, cfg.proto.loss, cfg.proto.bottomk_normals = score, loss, bottomk
    return cfg


def _row(row: str) -> Config:
    """The host with the row's proto.* overrides (constants.PROTO_ROWS)."""
    cfg = _host()
    for key, value in constants.PROTO_ROWS[row].items():
        setattr(cfg.proto, key.split(".", 1)[1], value == "true")
    return cfg


def _lam(bank: PrototypeBank | None) -> torch.nn.Parameter:
    assert bank is not None and bank.lam is not None
    return bank.lam


def _build(cfg: Config, seed: int = 0) -> KATVAD:
    torch.manual_seed(seed)
    return KATVAD.from_config(cfg)


class TestConfig:
    def test_default_builds_nothing_and_keeps_the_state_dict(self) -> None:
        cfg = _host()
        assert not cfg.proto.enabled
        model = _build(cfg)
        assert model.proto is None
        assert not any(k.startswith("proto.") for k in model.state_dict())

    def test_rows_add_only_proto_keys(self) -> None:
        host = set(_build(_host()).state_dict())
        for score, loss in ((True, False), (False, True), (True, True)):
            keys = set(_build(_cfg(score, loss)).state_dict())
            extra = keys - host
            assert host <= keys and all(k.startswith("proto.") for k in extra)
            assert ("proto.lam" in extra) == score
            assert {"proto.prototypes", "proto.proto_init"} <= extra

    def test_constants_rows_parse(self) -> None:
        for row, sets in constants.PROTO_ROWS.items():
            parsed = load_config(overrides=[f"{k}={v}" for k, v in sets.items()])
            assert parsed.proto == _row(row).proto, row
            validate_proto(_row(row))

    def test_refuses_what_no_row_defines(self) -> None:
        with_kim = _cfg()
        with_kim.kim.enabled = True
        orphan = _cfg(False, False, bottomk=True)
        bad_temp = _cfg()
        bad_temp.proto.temp = 0.0
        for cfg, match in ((with_kim, "KIM-off"), (orphan, "needs proto.score"),
                           (bad_temp, "proto.temp")):
            with pytest.raises(ValueError, match=match):
                validate_proto(cfg)


class TestBank:
    def test_term_is_zero_until_both_prototypes_exist(self) -> None:
        bank = PrototypeBank(D, score=True)
        _lam(bank).data.fill_(2.0)
        feats, mask = torch.randn(2, 4, D), torch.ones(2, 4)
        assert torch.equal(bank.score_term(feats, mask), torch.zeros(2, 4))
        none = torch.zeros(2, 4, dtype=torch.bool)
        bank.update(feats, ~none, none, 0.1)
        assert not bank.ready
        assert torch.equal(bank.score_term(feats, mask), torch.zeros(2, 4))

    def test_term_formula_and_mask(self) -> None:
        bank = PrototypeBank(D, score=True)
        mu = torch.randn(2, D)
        bank.prototypes.copy_(mu)
        bank.proto_init.fill_(True)
        _lam(bank).data.fill_(1.5)
        feats = torch.randn(1, 3, D)
        mask = padding_mask(torch.tensor([2]), 3)
        want = 1.5 * (F.cosine_similarity(feats, mu[1].view(1, 1, -1), dim=-1)
                      - F.cosine_similarity(feats, mu[0].view(1, 1, -1), dim=-1)) * mask
        torch.testing.assert_close(bank.score_term(feats, mask), want)
        assert bank.score_term(feats, mask)[0, 2] == 0.0

    def test_lambda_starts_at_zero_and_loss_only_bank_has_no_parameter(self) -> None:
        assert _lam(PrototypeBank(D, score=True)).item() == constants.PROTO_LAMBDA_INIT == 0.0
        loss_only = PrototypeBank(D, score=False)
        assert loss_only.lam is None and not list(loss_only.parameters())
        with pytest.raises(RuntimeError, match="without the score term"):
            loss_only.score_term(torch.randn(1, 2, D), torch.ones(1, 2))

    def test_update_sets_then_moves_by_momentum(self) -> None:
        bank = PrototypeBank(2, score=False)
        feats = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        normal = torch.tensor([[True, False]])
        abnormal = torch.tensor([[False, True]])
        bank.update(feats, normal, abnormal, 0.1)
        torch.testing.assert_close(bank.prototypes, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        bank.update(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]), normal, abnormal, 0.1)
        torch.testing.assert_close(bank.prototypes, torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
        assert bank.ready and abs(float(bank.cosine()) - 0.18 / 0.82) < 1e-5

    def test_bottomk_picks_the_lowest_steps_of_abnormal_bags_only(self) -> None:
        score = torch.tensor([[0.9, 0.1, 0.5, 0.2, 0.0], [0.3, 0.2, 0.1, 0.0, 0.0]])
        got = bottomk_steps(score, torch.tensor([1.0, 0.0]), torch.tensor([4, 3]), topk_pct=2)
        assert got[0].tolist() == [False, True, False, True, False]  # padded 0.0 never picked
        assert not got[1].any()


class TestForward:
    @pytest.mark.parametrize("train", [False, True])
    def test_step_one_forward_equals_the_host(self, train: bool) -> None:
        """lambda = 0 and no prototype yet: the row draws and scores exactly what K0dfi does."""
        host, row = _build(_host()), _build(_cfg())
        row.load_state_dict({**host.state_dict(), **{k: v for k, v in row.state_dict().items()
                                                       if k.startswith("proto.")}})
        host.train(train)
        row.train(train)
        rows, lengths = _rows()
        text = torch.randn(3, D)
        torch.manual_seed(7)
        a = host(rows, lengths, text)["cls_bin_logits"]
        torch.manual_seed(7)
        b = row(rows, lengths, text)["cls_bin_logits"]
        torch.testing.assert_close(b, a)

    def test_host_init_equals_the_host(self) -> None:
        host, row = _build(_host()), _build(_cfg())
        row_sd = row.state_dict()
        for key, value in host.state_dict().items():
            torch.testing.assert_close(row_sd[key], value, msg=key)

    def test_term_reaches_the_logits_once_ready(self) -> None:
        model = _build(_cfg()).eval()
        assert model.proto is not None and model.proto.lam is not None
        rows, lengths = _rows()
        text = torch.randn(3, D)
        base = model(rows, lengths, text)
        model.proto.prototypes.copy_(torch.randn(2, D))
        model.proto.proto_init.fill_(True)
        model.proto.lam.data.fill_(1.0)
        out = model(rows, lengths, text)
        mask = padding_mask(lengths, rows.shape[1])
        want = base["cls_bin_logits"] + model.proto.score_term(out["vis_feats"], mask)
        torch.testing.assert_close(out["cls_bin_logits"], want)


class TestTraining:
    @pytest.mark.parametrize("row", list(constants.PROTO_ROWS))
    def test_trains_logs_and_scores(self, tmp_path: Path, row: str) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _row(row)
        cfg.proto.warmup_steps = 0
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 2, "cpu"
        set_global_seed(cfg.train.seed)
        out = tmp_path / "run"
        trainer = build_trainer(cfg, out, data_dir, host, text_encoder="stub", log_every=1)
        cfg.save_yaml(out / "config.yaml")
        trainer.train()
        logged = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x]
        assert logged[-1]["proto_ready"] == 1.0
        assert all(k in logged[-1] for k in ("proto_cos", "proto_anchor_share"))
        assert ("proto_lambda" in logged[-1]) == cfg.proto.score
        assert any("proto_nkc" in r for r in logged) == cfg.proto.loss
        model = load_model_for_scoring(Config(), CPU, out / "checkpoint_last.pt", None, "stub")
        assert model.proto is not None and model.proto.ready
        trained = trainer.model.proto
        assert trained is not None
        torch.testing.assert_close(model.proto.prototypes, trained.prototypes)
        if cfg.proto.score:
            assert float(_lam(model.proto)) != 0.0  # lambda trained
        rows, lengths = _rows()
        assert torch.isfinite(model(rows, lengths, torch.randn(2, D))["cls_bin_logits"]).all()

    def test_host_logs_no_proto_key(self, tmp_path: Path) -> None:
        data_dir, host = _fixture(tmp_path)
        cfg = _host()
        cfg.train.batch_size, cfg.train.num_epochs, cfg.train.device = 4, 1, "cpu"
        set_global_seed(cfg.train.seed)
        out = tmp_path / "run"
        build_trainer(cfg, out, data_dir, host, text_encoder="stub", log_every=1).train()
        logged = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x]
        assert not any(k.startswith("proto_") for r in logged for k in r)
