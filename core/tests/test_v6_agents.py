"""v6 LaGoVAD-AP: agent geometry, tracking bookkeeping, agent rows, the branch + gate, L_agent,
the cache wiring and the Step 0 / read-out arithmetic (``core/docs/v6/PREREG_V6.md``)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from core import constants
from core.agents import layout
from core.agents.features import RowSources, agent_rows, roi_motion, slot_boxes
from core.agents.geometry import (
    LOG_COLS,
    Track,
    TrackLog,
    box_at,
    build_tracks,
    compose_affine,
    crop_square,
    cut_square,
    ego_block,
    iou_matrix,
    kinematics,
    pixel_to_normalized_affine,
    select_agents,
)
from core.agents.tracking import track_video
from core.config import load_config
from core.data.agent_cache import (
    append_agents,
    check_agent_dir,
    read_agent_manifest,
    write_agent_manifest,
)
from core.data.collate import collate_variable_length
from core.data.dataset import DVSFeatureDataset
from core.losses.agent_mil import agent_mil_loss
from core.models.agent_branch import AgentBranch, pair_features
from core.models.kat_vad import KATVAD
from core.models.text_encoding import make_text_encoder
from core.tests.fixtures import build_fixture
from core.tools import v6_agents
from core.tools.v5_stage1 import A2_OVERRIDES
from core.train import (
    Trainer,
    agent_boundary_rows,
    load_class_names,
    optimizer_param_groups,
    set_global_seed,
)

CPU = torch.device("cpu")
K = constants.V6_K_MAX
BASE = constants.HIDDEN_DIM + constants.V5_GRID_DIM
AP = [*A2_OVERRIDES, "agent.enabled=true"]


def _log(tracks: dict[int, list[tuple[int, list[float]]]], frames: int, cls: int = 2) -> TrackLog:
    """A log over detection frames 0..frames-1 (det_step 1): tid -> [(det index, box)]."""
    rows = []
    for tid, items in tracks.items():
        for det, box in items:
            r = np.zeros(LOG_COLS)
            r[0], r[1], r[2:6], r[6], r[7], r[8:12] = det, tid, box, 0.9, cls, box
            rows.append(r)
    return TrackLog(
        det_frames=np.arange(frames),
        affine=np.tile(np.eye(2, 3), (frames, 1, 1)),
        rows=np.stack(rows) if rows else np.zeros((0, LOG_COLS)),
        size=(1280, 720),
        num_frames=frames,
    )


def _agent_block(rows: int, valid: list[list[int]], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    agents = np.zeros((rows, K, constants.V6_AGENT_DIM), dtype=np.float32)
    for r, slots in enumerate(valid):
        for k in slots:
            agents[r, k, layout.VALID] = 1
            x, y = rng.uniform(0.1, 0.7, 2)
            agents[r, k, layout.BOX] = [x, y, x + 0.1, y + 0.1]
            agents[r, k, layout.CONF] = 0.8
            agents[r, k, layout.CLASS] = 2
            agents[r, k, layout.TRACK] = k + 1
            agents[r, k, layout.MOTION.start :] = rng.standard_normal(
                constants.V6_AGENT_MOTION_DIM + constants.V6_AGENT_CROP_DIM
            )
    ego = rng.standard_normal((rows, constants.V6_EGO_DIM)).astype(np.float32) * 0.01
    return layout.pack_rows(agents, ego)


# --------------------------------------------------------------------------- layout / geometry


class TestLayout:
    def test_round_trip(self) -> None:
        packed = _agent_block(3, [[0], [0, 5], []])
        assert packed.shape == (3, constants.V6_ROW_WIDTH)
        agents, ego = layout.unpack_rows(packed)
        assert agents.shape == (3, K, constants.V6_AGENT_DIM) and ego.shape == (3, 48)
        assert agents[1, 5, layout.VALID] == 1 and agents[2, :, layout.VALID].sum() == 0
        np.testing.assert_array_equal(layout.pack_rows(agents, ego), packed)

    def test_split_refuses_a_row_without_agents(self) -> None:
        with pytest.raises(ValueError, match="agent cache"):
            layout.split_row(np.zeros((2, BASE)), layout.RowLayout(base=BASE))


class TestGeometry:
    def test_iou(self) -> None:
        a = np.array([[0, 0, 0.2, 0.2]])
        b = np.array([[0, 0, 0.2, 0.2], [0.1, 0, 0.3, 0.2], [0.5, 0.5, 0.6, 0.6]])
        np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])

    def test_affine_normalization_and_composition(self) -> None:
        shift = np.array([[1.0, 0, 128.0], [0, 1.0, -72.0]])
        norm = pixel_to_normalized_affine(shift, 1280, 720)
        np.testing.assert_allclose(norm, [[1, 0, 0.1], [0, 1, -0.1]])
        affine = np.stack([np.eye(2, 3), norm, norm])
        total = compose_affine(affine, np.array([0, 3, 6]), 0, 7)
        np.testing.assert_allclose(total[:, 2], [0.2, -0.2])
        np.testing.assert_allclose(compose_affine(affine, np.array([0, 3, 6]), 4, 5), np.eye(2, 3))

    def test_short_tracks_drop_and_ids_renumber(self) -> None:
        box = [0.1, 0.1, 0.2, 0.2]
        log = _log({7: [(0, box), (1, box), (2, box)], 9: [(1, box), (2, box)]}, 4)
        tracks = build_tracks(log)
        assert list(tracks) == [1] and tracks[1].cls == 2  # COCO car -> V6 index 2

    def test_rider_merges_into_the_two_wheeler(self) -> None:
        bike, person = [0.4, 0.5, 0.5, 0.7], [0.42, 0.4, 0.48, 0.65]
        log = _log({1: [(f, bike) for f in range(4)]}, 4, cls=3)
        rows = [log.rows]
        for f in range(4):
            r = np.zeros(LOG_COLS)
            r[0], r[1], r[2:6], r[6], r[7], r[8:12] = f, 2, person, 0.9, 0, person
            rows.append(r[None])
        log.rows = np.concatenate(rows)
        tracks = build_tracks(log)
        assert len(tracks) == 1
        (track,) = tracks.values()
        assert track.cls == constants.V6_RIDER_CLASS
        np.testing.assert_allclose(track.boxes[0], [0.4, 0.4, 0.5, 0.7])

    def test_box_interpolates_between_detections(self) -> None:
        t = Track(1, np.array([0, 4]), np.array([[0, 0, 0.1, 0.1], [0.4, 0, 0.5, 0.1]]),
                  np.zeros((2, 4)), np.array([0.5, 0.7]), 2)
        b = box_at(t, 2)
        assert b is not None
        np.testing.assert_allclose(b, [0.2, 0, 0.3, 0.1])
        assert box_at(t, 5) is None and t.mean_conf == pytest.approx(0.6)

    def test_static_object_under_a_camera_pan_has_no_velocity(self) -> None:
        """Ego compensation: the image shifts 0.01 / frame with the camera -> u = 0."""
        frames = 4
        boxes = [[0.3 + 0.01 * f, 0.4, 0.4 + 0.01 * f, 0.5] for f in range(frames)]
        log = _log({1: list(enumerate(boxes))}, frames)
        log.affine[1:] = [[1, 0, 0.01], [0, 1, 0]]
        (track,) = build_tracks(log).values()
        kin = kinematics(track, 3, log, [track], det_step=1)
        np.testing.assert_allclose(kin[:4], 0.0, atol=1e-9)
        assert kin[constants.V6_KIN_NAMES.index("age")] == pytest.approx(0.3)

    def test_ego_block_is_zero_without_camera_motion(self) -> None:
        log = _log({}, 20)
        assert np.abs(ego_block(log, list(range(16)))).max() == 0.0

    def test_crop_is_square_padded(self) -> None:
        x0, y0, side = crop_square(np.array([0.0, 0.0, 0.01, 0.01]), 1280, 720)
        assert side == constants.V6_CROP_MIN_PX
        frame = np.full((720, 1280, 3), 255, np.uint8)
        cut = cut_square(frame, x0, y0, side)
        assert cut.shape == (side, side, 3) and cut[0, 0].sum() == 0 and cut[-1, -1].sum() > 0

    def test_selection_is_by_track_mean_confidence(self) -> None:
        tracks = {
            i: Track(i, np.array([0, 5]), np.zeros((2, 4)), np.zeros((2, 4)),
                     np.array([c, c]), 2)
            for i, c in [(1, 0.2), (2, 0.9), (3, 0.5)]
        }
        assert [t.tid for t in select_agents(tracks, 3, 2)] == [2, 3]


# --------------------------------------------------------------------------- tracking / rows


class _Boxes:
    def __init__(self, xyxy: np.ndarray) -> None:
        self.xyxy = xyxy


class _Tracker:
    """Echoes every detection as track 1 + its index."""

    def update(self, boxes: _Boxes, img: np.ndarray | None = None) -> np.ndarray:
        n = len(boxes.xyxy)
        return np.column_stack(
            [boxes.xyxy, np.arange(1, n + 1), np.full(n, 0.9), np.full(n, 2), np.arange(n)]
        )


class _Gmc:
    def apply(self, raw_frame: np.ndarray) -> np.ndarray:
        return np.array([[1.0, 0, 64.0], [0, 1.0, 0]])


def test_track_video_bookkeeping() -> None:
    frames = [(i * 3, np.zeros((720, 1280, 3), np.uint8)) for i in range(5)]

    def detect(batch: list[np.ndarray]) -> list[_Boxes]:
        return [_Boxes(np.array([[128.0, 72, 256, 144]])) for _ in batch]

    log = track_video(iter(frames), 15, detect, _Tracker(), _Gmc(), batch_size=2)
    np.testing.assert_array_equal(log.det_frames, [0, 3, 6, 9, 12])
    np.testing.assert_allclose(log.affine[0], np.eye(2, 3))
    np.testing.assert_allclose(log.affine[1][:, 2], [0.05, 0.0])
    np.testing.assert_allclose(log.rows[0, 2:6], [0.1, 0.1, 0.2, 0.2])
    assert log.rows.shape == (5, LOG_COLS) and log.size == (1280, 720)


class _Clip:
    def __call__(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(image_embeds=torch.ones(len(pixel_values), 512))


def test_agent_rows_and_roi_motion() -> None:
    frames = 30
    box = [0.2, 0.2, 0.5, 0.6]
    log = _log({1: [(f, box) for f in range(0, frames, 3)]}, frames)
    log.det_frames = np.arange(frames)
    log.rows[:, 0] = np.arange(0, frames, 3)

    def tokens(clips: list[list[int]]) -> torch.Tensor:
        return torch.full((len(clips), 8 * 14 * 14, constants.V5_GRID_DIM), 2.0)

    sources = RowSources(
        num_frames=frames, stride=3, clip_step=1, det_step=1, tokens=tokens,
        frame=lambda t: np.zeros((720, 1280, 3), np.uint8),
    )
    rows, summary = agent_rows(log, sources, _Clip(), CPU)
    assert rows.shape == (10, constants.V6_ROW_WIDTH) and rows.dtype == np.float16
    agents, _ = layout.unpack_rows(rows.astype(np.float32))
    assert agents[:, 0, layout.VALID].min() == 1 and agents[:, 1:, layout.VALID].max() == 0
    np.testing.assert_allclose(agents[:, 0, layout.MOTION], 2.0)
    np.testing.assert_allclose(agents[:, 0, layout.CROP], 1.0)
    assert summary["tracks"] == 1 and summary["agents_max"] == 1


def test_roi_motion_reads_the_box_cells() -> None:
    tok = torch.zeros(1, 8, 14, 14, 3)
    tok[0, :, :7, :7] = 1.0  # top-left quadrant
    tl: list[np.ndarray | None] = [np.array([0.0, 0.0, 0.5, 0.5])] * 8
    br: list[np.ndarray | None] = [np.array([0.5, 0.5, 1.0, 1.0])] * 8
    boxes = [[tl, br]]
    out = roi_motion(tok.reshape(1, -1, 3), boxes)
    np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-5)
    assert np.abs(out[0, 2:]).max() == 0


def test_slot_boxes_skip_absent_frames() -> None:
    t = Track(1, np.array([4, 9]), np.array([[0.1, 0.1, 0.2, 0.2]] * 2), np.zeros((2, 4)),
              np.array([0.5, 0.5]), 2)
    boxes = slot_boxes(t, list(range(16)))
    assert boxes[0] is None and boxes[1] is None and boxes[2] is not None and boxes[5] is None


# --------------------------------------------------------------------------- model / loss


def _rows(b: int, length: int, seed: int = 0) -> torch.Tensor:
    base = torch.randn(b, length, BASE, generator=torch.Generator().manual_seed(seed))
    agents = torch.from_numpy(
        np.stack([_agent_block(length, [[0, 1, 2]] * (length - 1) + [[]], seed + i)
                  for i in range(b)])
    )
    return torch.cat([base, agents], dim=-1)


class TestModel:
    def test_alpha_zero_is_a2(self) -> None:
        a2 = KATVAD.from_config(load_config(None, A2_OVERRIDES)).eval()
        ap = KATVAD.from_config(load_config(None, AP)).eval()
        missing, unexpected = ap.load_state_dict(a2.state_dict(), strict=False)
        assert unexpected == []
        assert all(k.startswith(("agent_branch.", "agent_gate.")) for k in missing)
        rows, lengths = _rows(2, 9), torch.tensor([9, 6])
        text = torch.randn(3, constants.HIDDEN_DIM)
        with torch.no_grad():
            out2 = a2(rows[..., :BASE], lengths, class_feats=text)
            outp = ap(rows, lengths, class_feats=text)
        torch.testing.assert_close(outp["cls_bin_logits"], out2["cls_bin_logits"])
        assert outp["agent_logits"].shape == (2, 9, K + 2)
        assert float(outp["agent_share"]) == 0.0
        # zero-init scorer: every valid candidate starts at p = 0.5
        assert outp["agent_logits"].abs().max() == 0

    def test_candidates_ego_and_no_agent(self) -> None:
        branch = AgentBranch()
        rows = _rows(1, 4)
        _, agents, ego = layout.split_row(rows, layout.RowLayout(base=BASE))
        out = branch(rows[..., :BASE], agents, ego, torch.tensor([[1, 1, 1, 0]]))
        cand = out.candidates[0]
        assert cand[0, :3].all() and not cand[0, 3:K].any() and cand[0, K] and not cand[0, K + 1]
        assert not cand[3].any()  # padded step
        assert out.z[0, 3].abs().max() == 0

    def test_no_agent_token_when_nothing_is_detected(self) -> None:
        branch = AgentBranch()
        rows = _rows(1, 3)
        _, agents, ego = layout.split_row(rows, layout.RowLayout(base=BASE))
        out = branch(rows[..., :BASE], agents, ego, torch.ones(1, 3))
        assert out.candidates[0, 2, K + 1] and out.candidates[0, 2, K]
        assert not out.candidates[0, 2, :K].any()

    def test_gradients_reach_the_scorer_and_alpha(self) -> None:
        ap = KATVAD.from_config(load_config(None, AP))
        rows, lengths = _rows(2, 6), torch.tensor([6, 6])
        out = ap(rows, lengths, class_feats=torch.randn(2, constants.HIDDEN_DIM))
        loss = agent_mil_loss(out["agent_logits"], out["agent_candidates"], torch.tensor([1.0, 0]),
                              lengths, torch.zeros(2, 6), torch.zeros(2, dtype=torch.bool))
        (loss["agent"] + out["cls_bin_logits"].sum()).backward()
        assert ap.agent_branch is not None and ap.agent_gate is not None
        assert float(ap.agent_branch.scorer[-1].weight.grad.abs().sum()) > 0
        assert ap.agent_gate.alpha.grad is not None

    def test_alpha_and_queries_have_no_weight_decay(self) -> None:
        cfg = load_config(None, AP)
        model = KATVAD.from_config(cfg)
        groups = optimizer_param_groups(model, cfg)
        assert len(groups) == 2 and groups[1]["weight_decay"] == 0.0
        assert model.agent_gate is not None and model.agent_branch is not None
        assert {id(p) for p in groups[1]["params"]} == {
            id(model.agent_gate.alpha), id(model.agent_branch.query)
        }
        n = sum(p.numel() for g in groups for p in g["params"])
        assert n == sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_pair_features(self) -> None:
        boxes = torch.tensor([[[0, 0, 0.2, 0.2], [0, 0, 0.2, 0.2]]])
        f = pair_features(boxes)
        assert f.shape == (1, 2, 2, constants.V6_PAIR_DIM)
        assert float(f[0, 0, 1, 4]) == pytest.approx(1.0) and float(f[0, 0, 1, 5]) == 0

    @pytest.mark.parametrize("override,match", [
        (["v2.motion=none", "model.motion_dim=0"], "A2's inputs"),
        (["kip.enabled=true"], "kip.enabled"),
    ])
    def test_bad_configs_raise(self, override: list[str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            KATVAD.from_config(load_config(None, [*AP, *override]))


class TestLoss:
    def _call(self, logits: torch.Tensor, label: float, **kw: Any) -> dict[str, torch.Tensor]:
        b, length, c = logits.shape
        cand = torch.ones(b, length, c, dtype=torch.bool)
        cand[..., -1] = False
        return agent_mil_loss(logits, cand, torch.full((b,), label), torch.full((b,), length),
                              torch.zeros(b, length), torch.zeros(b, dtype=torch.bool), **kw)

    def test_normal_dense_negatives_push_every_candidate(self) -> None:
        logits = torch.zeros(1, 4, 5, requires_grad=True)
        self._call(logits, 0.0)["agent"].backward()
        assert (_grad(logits)[..., :-1] > 0).all() and (_grad(logits)[..., -1] == 0).all()

    def test_abnormal_pulls_only_the_top_candidate(self) -> None:
        logits = torch.zeros(1, 16, 5)
        logits[0, 3, 1] = 2.0
        logits.requires_grad_(True)
        out = self._call(logits, 1.0, beta=0.0)
        out["agent"].backward()
        grad = _grad(logits)[0]
        assert grad[3, 1] < 0 and int((grad != 0).sum()) == 1
        p = torch.sigmoid(torch.tensor(2.0))
        assert float(out["agent"]) == pytest.approx(float(-torch.log(p)), rel=1e-5)

    def test_bag_negatives_touch_only_the_selected(self) -> None:
        logits = torch.zeros(1, 16, 5)
        logits[0, 0, 0] = 1.0
        logits.requires_grad_(True)
        self._call(logits, 0.0, dense_negatives=False)["agent"].backward()
        assert int((_grad(logits) != 0).sum()) == 1

    def test_synthetic_pseudo_normal_steps_skip_the_boundary(self) -> None:
        logits = torch.zeros(1, 8, 3, requires_grad=True)
        cand = torch.ones(1, 8, 3, dtype=torch.bool)
        pseudo = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0]], dtype=torch.float)
        agent_mil_loss(logits, cand, torch.ones(1), torch.tensor([8]), pseudo,
                       torch.ones(1, dtype=torch.bool), beta=0.0,
                       boundary_rows=1)["agent"].backward()
        touched = (_grad(logits)[0] != 0).any(-1)
        assert touched[:3].all() and not touched[3] and not touched[6] and touched[7]

    def test_padding_is_ignored(self) -> None:
        logits = torch.zeros(1, 6, 3, requires_grad=True)
        agent_mil_loss(logits, torch.ones(1, 6, 3, dtype=torch.bool), torch.zeros(1),
                       torch.tensor([4]), torch.zeros(1, 6),
                       torch.zeros(1, dtype=torch.bool))["agent"].backward()
        assert (_grad(logits)[0, 4:] == 0).all()


# --------------------------------------------------------------------------- cache wiring


def _agent_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    fixture = build_fixture(tmp_path / "fx")
    rng = np.random.default_rng(1)
    a2_dir, agent_dir = tmp_path / "a2", tmp_path / "agents"
    a2_dir.mkdir()
    write_agent_manifest(agent_dir, {
        "width": constants.V6_ROW_WIDTH, "k_max": K, "stride": constants.FRAME_STRIDE,
        "clip_step": 3, "layout": "test",
    })
    for path in fixture.clip_dir.glob("*.npy"):
        x = np.load(path)
        np.save(a2_dir / path.name, np.concatenate(
            [x, rng.standard_normal((len(x), constants.V5_GRID_DIM))], 1).astype(np.float32))
        np.save(agent_dir / path.name,
                _agent_block(len(x), [[0, 1]] * len(x), seed=len(x)).astype(np.float16))
    (a2_dir / constants.V2_INPUT_MANIFEST_FILENAME).write_text(
        json.dumps({"crn": constants.V2_OFF, "motion": constants.V5_ENCODER})
    )
    return fixture.data_dir, a2_dir, agent_dir


class TestWiring:
    def test_trainer_losses(self, tmp_path: Path) -> None:
        data_dir, a2_dir, agent_dir = _agent_fixture(tmp_path)
        cfg = load_config(None, AP)
        cfg.train.batch_size = 4
        set_global_seed(cfg.train.seed)
        check_agent_dir(True, agent_dir, cfg.data.frame_stride)
        dataset = DVSFeatureDataset(data_dir=data_dir, clip_dir=a2_dir, dvs=cfg.dvs,
                                    require_flow=False, seed=cfg.train.seed, agent_dir=agent_dir)
        item = dataset[0]
        assert item["v_feat"].shape[-1] == BASE + constants.V6_ROW_WIDTH
        model = KATVAD.from_config(cfg)
        trainer = Trainer(
            cfg=cfg, model=model, dataset=dataset, class_names=load_class_names(data_dir),
            text_encode_fn=make_text_encoder(model, "stub", CPU, dim=constants.HIDDEN_DIM),
            output_dir=tmp_path / "out", device=CPU,
            agent_boundary_rows=agent_boundary_rows(agent_dir),
        )
        losses = trainer.compute_losses(collate_variable_length([dataset[i] for i in range(4)]))
        assert torch.isfinite(losses["total"]) and "agent" in losses
        assert float(losses["agent_alpha"]) == 0.0
        assert agent_boundary_rows(agent_dir) == 3  # 8 frames x step 3 over stride 8

    def test_cache_checks(self, tmp_path: Path) -> None:
        _, _, agent_dir = _agent_fixture(tmp_path)
        assert read_agent_manifest(agent_dir)["stride"] == constants.FRAME_STRIDE
        with pytest.raises(ValueError, match="s1"):
            check_agent_dir(True, agent_dir, 3)
        with pytest.raises(ValueError, match=r"agent\.enabled=false"):
            check_agent_dir(False, agent_dir, 8)
        with pytest.raises(ValueError, match="--agent-dir"):
            check_agent_dir(True, None, 8)
        with pytest.raises(ValueError, match="agent rows"):
            append_agents(np.zeros((3, BASE)), np.zeros((4, constants.V6_ROW_WIDTH)), "x")


# --------------------------------------------------------------------------- Step 0 / read-out


def _grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad


def _gt(frames: dict[int, list[float]], ego: bool = False) -> v6_agents.GroundTruth:
    return v6_agents.GroundTruth(
        by_frame={f: np.array([b]) for f, b in frames.items()},
        tracks={0: sorted(frames)}, track_boxes={(0, f): np.array(b) for f, b in frames.items()},
        ego=ego, night=False, category="turning", start=0, end=99,
    )


class TestStep0:
    def test_row_hits_and_priors(self) -> None:
        rows = np.zeros((2, constants.V6_ROW_WIDTH), np.float32)
        agents, ego = layout.unpack_rows(rows)
        agents[:, 0, layout.VALID] = agents[:, 1, layout.VALID] = 1
        agents[:, 0, layout.BOX] = [0.45, 0.45, 0.55, 0.55]  # central, small: not the GT
        agents[:, 1, layout.BOX] = [0.0, 0.5, 0.4, 1.0]  # large, low: the GT
        rows = layout.pack_rows(agents, ego)
        gt = _gt({0: [0.0, 0.5, 0.4, 1.0], 3: [0.0, 0.5, 0.4, 1.0]})
        scores = v6_agents.agent_scores(rows, np.ones((2, 512)))
        h = v6_agents.row_hits(rows, gt, 3, scores, 0.3)
        assert h["centre"] == [0.0, 0.0] and h["largest"] == [1.0, 1.0] and h["lowest"] == [1, 1]
        assert h["ceiling"] == [1.0, 1.0] and h["chance"] == [0.5, 0.5]

    def test_track_recall(self) -> None:
        box = [0.1, 0.1, 0.3, 0.3]
        gt = _gt(dict.fromkeys(range(4), box))
        t = Track(1, np.array([0, 1]), np.array([box, box]), np.array([box, box]),
                  np.array([0.9, 0.9]), 2)
        (r,) = v6_agents.track_recall(gt, {1: t}, lambda f: [t])
        assert r["found@0.3"] and r["frame_share@0.3"] == 0.5

    def _d(self, overall: float, non_ego: float, lo: float) -> tuple[dict, dict]:
        rec = {"recall": overall, "tracks": 10, "kept_recall": overall}
        d0 = {"overall@0.3": rec, "non_ego": {**rec, "recall": non_ego},
              "night": {**rec, "recall": None},
              "by_size": dict.fromkeys(v6_agents.SIZE_NAMES, rec)}
        d2 = {"iou0.3": {f"{n}-centre": {"mean": 0.1, "low": lo, "high": 0.2}
                         for n in constants.V6_RANK_SCORES}}
        return d0, d2

    @pytest.mark.parametrize("overall,non_ego,lo,label", [
        (0.85, 0.75, 0.01, "GO"),
        (0.85, 0.65, 0.01, "NARROW"),
        (0.85, 0.75, -0.01, "NARROW"),
        (0.70, 0.75, 0.01, "NARROW"),
        (0.55, 0.75, 0.01, "NO-GO"),
    ])
    def test_decision(self, overall: float, non_ego: float, lo: float, label: str) -> None:
        assert v6_agents.decide(*self._d(overall, non_ego, lo))["label"] == label

    def test_detection_block(self) -> None:
        ya, yb = np.array([0, 0, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 1, 1])
        scores = {"a_000001": ya.astype(float), "b_000001": 1.0 - yb}
        labels = {"a_000001": ya, "b_000001": yb}
        d = v6_agents.detection_block(scores, labels, sorted(scores))
        assert d["clips"] == 2 and d["macro"]["mean"] == pytest.approx(0.5)
        # same-position bins pair a's positives (score 1) with b's negatives (score 1 or 0)
        assert d["strat50"] is not None and 0.0 <= d["strat50"] <= 1.0
