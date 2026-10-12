"""v2.2 order diagnostics: the fixed-capacity temporal probe (architecture §5).

Two valid ``Conv1d`` layers (kernel 5, 64 channels, ReLU) give each step a receptive field of 9
steps (±4). Training windows carry 4 **real** context steps on each side, taken from their source
video, so no padding ever falls inside a scored region; replicate padding is used only at a
sequence's true ends. At test a whole clip is replicate-padded by 4 and scored in one pass.

A sequence is given as ``base`` rows in **slot order** plus ``index``, the original step index
of each slot (identity in true order; ``π`` under the block shuffle). The position feature ``p``
is the cubic basis of the step's original index relative to the scored sequence -- the training
window, or the whole clip -- so it moves with its step (plan O5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.adamw import AdamW

from core import constants


def cubic(tau: np.ndarray) -> np.ndarray:
    """``[τ, τ², τ³]`` (``kill_switch_probe.position_features`` for arbitrary ``τ``)."""
    return np.stack([tau ** (k + 1) for k in range(constants.V2_K_POSITION_DEGREE)], axis=1)


@dataclass(frozen=True)
class Sequence:
    """One source video or clip in slot order."""

    base: np.ndarray  # (n, d) features per slot
    index: np.ndarray  # (n,) original step index of each slot
    labels: np.ndarray  # (n,) label of each slot


def _pad(rows: np.ndarray, context: int) -> np.ndarray:
    return np.concatenate([np.repeat(rows[:1], context, 0), rows, np.repeat(rows[-1:], context, 0)])


def _rows(seq: Sequence, start: int, end: int, position: bool, context: int) -> np.ndarray:
    """Input rows for scored slots ``start:end`` plus ``context`` slots each side."""
    base = _pad(seq.base, context)[start : end + 2 * context]
    if not position:
        return base
    index = _pad(seq.index.astype(np.float64)[:, None], context)[start : end + 2 * context, 0]
    tau = (index - start + 0.5) / (end - start)
    return np.concatenate([base, cubic(tau)], axis=1)


def window_rows(seq: Sequence, start: int, end: int, position: bool) -> np.ndarray:
    """A training window: slots ``start:end`` with real context; ``τ`` relative to the window."""
    return _rows(seq, start, end, position, constants.V22_PROBE_CONTEXT)


def sequence_rows(seq: Sequence, position: bool) -> np.ndarray:
    """A whole sequence, replicate-padded at its ends; ``τ`` relative to the sequence."""
    return _rows(seq, 0, len(seq.index), position, constants.V22_PROBE_CONTEXT)


class TemporalProbe(nn.Module):
    """``Conv1d(d → 64, 5, valid) → ReLU → Conv1d(64 → 1, 5, valid)`` with fixed standardisation."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        dim = len(mean)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.conv1 = nn.Conv1d(dim, constants.V22_PROBE_CHANNELS, constants.V22_PROBE_KERNEL)
        self.conv2 = nn.Conv1d(constants.V22_PROBE_CHANNELS, 1, constants.V22_PROBE_KERNEL)

    def forward(self, rows: Tensor) -> Tensor:
        """``(B, L + 8, d)`` → logits ``(B, L)``."""
        x = ((rows - self.mean) / self.std).transpose(1, 2)
        out: Tensor = self.conv2(torch.relu(self.conv1(x))).squeeze(1)
        return out


def fit_probe(
    windows: np.ndarray, labels: np.ndarray, seed: int, device: torch.device
) -> TemporalProbe:
    """Train one probe on ``(N, L + 8, d)`` windows with ``(N, L)`` step labels (arch §5)."""
    context = constants.V22_PROBE_CONTEXT
    core = windows[:, context:-context].reshape(-1, windows.shape[-1])
    std = core.std(axis=0)
    std[std == 0.0] = 1.0
    torch.manual_seed(seed)
    model = TemporalProbe(core.mean(axis=0), std).to(device)
    positives = float(labels.sum())
    if positives in (0.0, float(labels.size)):
        raise ValueError("the probe needs both classes among the training steps")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((labels.size - positives) / positives, device=device)
    )
    optimiser = AdamW(
        model.parameters(), lr=constants.V22_PROBE_LR,
        weight_decay=constants.V22_PROBE_WEIGHT_DECAY,
    )
    x_all = torch.as_tensor(windows, dtype=torch.float32)
    y_all = torch.as_tensor(labels, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(constants.V22_PROBE_EPOCHS):
        for batch in torch.randperm(len(x_all), generator=generator).split(
            constants.V22_PROBE_BATCH
        ):
            optimiser.zero_grad()
            loss = loss_fn(model(x_all[batch].to(device)), y_all[batch].to(device))
            loss.backward()
            optimiser.step()
    return model.eval()


@torch.no_grad()
def score(model: TemporalProbe, rows: np.ndarray, device: torch.device) -> np.ndarray:
    """Logits for one replicate-padded sequence ``(n + 8, d)`` → ``(n,)``."""
    tensor = torch.as_tensor(rows, dtype=torch.float32, device=device)[None]
    return np.asarray(model(tensor)[0].cpu().numpy(), dtype=np.float64)


def seed_averaged_scores(
    train: list[tuple[Sequence, int, int]],
    test: dict[str, Sequence],
    position: bool,
    seeds: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    """Fit one probe per seed; return slot-order logits averaged over seeds, and per seed."""
    windows = np.stack(
        [window_rows(seq, s, e, position).astype(np.float32) for seq, s, e in train]
    )
    labels = np.stack([seq.labels[s:e] for seq, s, e in train])
    rows = {v: sequence_rows(seq, position) for v, seq in test.items()}
    per_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        model = fit_probe(windows, labels, seed, device)
        per_seed[seed] = {v: score(model, r, device) for v, r in rows.items()}
    averaged = {v: np.mean([per_seed[s][v] for s in seeds], axis=0) for v in test}
    return averaged, per_seed


__all__ = [
    "Sequence",
    "TemporalProbe",
    "cubic",
    "fit_probe",
    "score",
    "seed_averaged_scores",
    "sequence_rows",
    "window_rows",
]
