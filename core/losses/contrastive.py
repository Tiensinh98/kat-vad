"""``L_neg`` — caption contrastive loss with hard-negative mining (n3).

Re-implemented from the LaGoVAD baseline (``CapContrastLoss`` +
``asymmetric_infonce_loss``). The module is stateless w.r.t. parameters and is
instantiated once in the model/training setup (not per step, unlike the
baseline training loop).
"""

from __future__ import annotations

import hashlib
import re

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from core import constants

CONTRAST_TYPE_VANILLA = "vanilla"
CONTRAST_TYPE_N3 = "n3"


def asymmetric_infonce_loss(
    sim: Tensor,
    label: Tensor,
    label_smoothing: float = constants.CONTRASTIVE_LABEL_SMOOTHING,
) -> Tensor:
    """Bidirectional InfoNCE on an asymmetric similarity matrix.

    ``sim (S, B')``: S anomaly captions vs B' aggregated video features;
    ``label (S,)``: index of each caption's own video among the B' columns.
    """
    loss_t2v = F.cross_entropy(sim, label, label_smoothing=label_smoothing)
    ano_sim = sim[:, label]  # (S, S)
    loss_v2t = F.cross_entropy(
        ano_sim,
        torch.arange(ano_sim.shape[0], device=ano_sim.device),
        label_smoothing=label_smoothing,
    )
    return (loss_t2v + loss_v2t) / 2


class CapContrastLoss(nn.Module):
    """Anomaly-caption vs. attention-aggregated video-feature contrastive loss.

    ``contrast_type='n3'`` additionally mines the *normal* portion of abnormal
    videos (per pseudo frame labels) as extra negative video features.
    """

    def __init__(
        self,
        contrast_type: str = CONTRAST_TYPE_N3,
        temperature: float = constants.CONTRASTIVE_TEMP,
    ) -> None:
        super().__init__()
        if contrast_type not in (CONTRAST_TYPE_VANILLA, CONTRAST_TYPE_N3):
            raise ValueError(f"Unknown contrast_type: {contrast_type!r}")
        self.contrast_type = contrast_type
        self.temperature = temperature

    def forward(
        self,
        logits: Tensor,
        lengths: Tensor,
        v_feats: Tensor,
        t_feats: Tensor,
        cls_label_idx: Tensor,
        pseudo_frame_label: Tensor,
    ) -> Tensor:
        """See baseline: ``logits (B, T)`` pre-sigmoid, ``t_feats (S, E)`` excl. Normal.

        ``cls_label_idx (B,)`` with 0 = normal; ``pseudo_frame_label (B, T)``.
        """
        max_len = logits.shape[1]
        batch = logits.shape[0]
        device = logits.device
        ano_indices = torch.where(cls_label_idx != 0)[0]  # (S,)

        # attention-aggregate each video's features under its anomaly curve
        pad = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
        masked_logits = logits + torch.where(pad, constants.NEG_INF_MASK_VALUE, 0.0)
        attn = torch.softmax(masked_logits / self.temperature, dim=1).unsqueeze(1)
        agg_v_feats = (attn @ v_feats).squeeze(1)  # (B, E)

        if self.contrast_type == CONTRAST_TYPE_N3:
            mined = []
            for b in range(batch):
                if cls_label_idx[b] == 0:
                    continue
                ano_logit = logits[b, : int(lengths[b])]
                ano_prob = ano_logit.sigmoid()
                # skip videos scored (nearly) uniformly — no separable normal part
                if ano_prob.max() - ano_prob.min() < constants.N3_MIN_SCORE_RANGE:
                    continue
                # baseline selects `pseudo_frame_label != 0` frames and weights the
                # lowest-scoring ones (softmax over negated logits) — kept verbatim
                mined_part = pseudo_frame_label[b, : int(lengths[b])] != 0
                mining_logit = ano_logit[mined_part]
                mining_attn = torch.softmax(-mining_logit / self.temperature, dim=0)
                mined.append(mining_attn @ v_feats[b][pseudo_frame_label[b] != 0])
            if mined:
                agg_v_feats = torch.cat([agg_v_feats, torch.stack(mined)], dim=0)

        sim_mat = torch.einsum("se,be->sb", t_feats, agg_v_feats)
        return asymmetric_infonce_loss(sim_mat, ano_indices)


def class_infonce_loss(
    sim: Tensor,
    row_class: Tensor,
    col_class: Tensor,
    label_smoothing: float = constants.CONTRASTIVE_LABEL_SMOOTHING,
    ignore: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Soft-target InfoNCE whose positives are every column of the row's class.

    ``sim (K, N)``, ``row_class (K,)``, ``col_class (N,)``. Each row's target is uniform
    over its class's columns (the "L_out" form of supervised contrastive), so same-class
    columns are never negatives. With one positive per row it equals the one-hot loss.
    Returns ``(loss, floor)``: ``floor`` is the smoothed target's entropy, the lowest
    value the loss can reach, logged so the excess is readable (KIM_RUN.md §28 NL3).
    """
    pos = (row_class[:, None] == col_class[None, :]).to(sim.dtype)
    counts = pos.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("class_infonce_loss: a row has no positive column")
    target = pos / counts
    if ignore is not None:
        return _masked_infonce(sim, target, pos.bool(), ignore, label_smoothing)
    loss = F.cross_entropy(sim, target, label_smoothing=label_smoothing)
    smoothed = (1.0 - label_smoothing) * target + label_smoothing / sim.shape[1]
    floor = -(smoothed * smoothed.clamp_min(torch.finfo(sim.dtype).tiny).log()).sum(1).mean()
    return loss, floor.detach()


def _masked_infonce(
    sim: Tensor, target: Tensor, pos: Tensor, ignore: Tensor, label_smoothing: float
) -> tuple[Tensor, Tensor]:
    """``class_infonce_loss`` with ``ignore (K, N)`` columns left out of each row (§30.1).

    An ignored column is neither positive nor negative: it is dropped from the softmax and
    from the label-smoothing mass. With no column ignored it equals ``F.cross_entropy``.
    """
    if bool((ignore & pos).any()):
        raise ValueError("class_infonce_loss: a positive column is also ignored")
    valid = ~ignore
    logp = torch.log_softmax(sim.masked_fill(ignore, float("-inf")), dim=1)
    n_valid = valid.sum(dim=1, keepdim=True).to(sim.dtype)
    smoothed = (1.0 - label_smoothing) * target + label_smoothing * valid.to(sim.dtype) / n_valid
    loss = -(smoothed * logp.masked_fill(ignore, 0.0)).sum(dim=1).mean()
    tiny = torch.finfo(sim.dtype).tiny
    floor = -(smoothed * smoothed.clamp_min(tiny).log()).sum(dim=1).mean()
    return loss, floor.detach()


class ClassContrastLoss(nn.Module):
    """``L_neg`` with class positives (KIM_RUN.md §28 NL2); the baseline form stays above.

    Rows are the text features of the abnormal classes present in the batch. Columns are
    every video's features aggregated under its anomaly curve, plus one mined "normal
    part" per abnormal video (lowest-scoring rows: over the anchor span when synthesized,
    over all valid rows when not). Similarity is cosine / ``cosine_temp``.
    """

    def __init__(
        self,
        mining: str = CONTRAST_TYPE_N3,
        temperature: float = constants.CONTRASTIVE_TEMP,
        cosine_temp: float = constants.LNEG_CLASS_TEMP,
    ) -> None:
        super().__init__()
        if mining not in (CONTRAST_TYPE_VANILLA, CONTRAST_TYPE_N3):
            raise ValueError(f"Unknown contrast_type: {mining!r}")
        self.mining = mining
        self.temperature = temperature
        self.cosine_temp = cosine_temp

    def _mine(
        self, logits: Tensor, lengths: Tensor, v_feats: Tensor, cls_idx: Tensor, pseudo: Tensor
    ) -> list[Tensor]:
        mined = []
        for b in torch.where(cls_idx != 0)[0].tolist():
            n = int(lengths[b])
            logit = logits[b, :n]
            prob = logit.sigmoid()
            if n == 0 or prob.max() - prob.min() < constants.N3_MIN_SCORE_RANGE:
                continue
            part = pseudo[b, :n] != 0
            if not bool(part.any()):  # un-synthesized: no anchor span, mine the whole window
                part = torch.ones_like(part)
            weights = torch.softmax(-logit[part] / self.temperature, dim=0)
            mined.append(weights @ v_feats[b, :n][part])
        return mined

    def forward(
        self,
        logits: Tensor,
        lengths: Tensor,
        v_feats: Tensor,
        class_feats: Tensor,
        cls_idx: Tensor,
        pseudo_frame_label: Tensor,
        dup: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``logits (B, T)`` pre-sigmoid, ``v_feats (B, T, E)``, ``class_feats (C, E)`` with
        row 0 = Normal, ``cls_idx (B,)`` with 0 = normal, ``pseudo_frame_label (B, T)``.
        ``dup (C, C)`` bool, optional (§30.1): ``dup[r, c]`` drops the columns of row ``c``
        from row ``r``'s softmax (near-duplicate captions); row / column 0 must be False.

        Returns ``loss``, ``floor``, ``mined`` (number of mined columns, detached) and
        ``masked_share`` (share of other-row abnormal columns dropped by ``dup``; 0 without).
        """
        present = torch.unique(cls_idx[cls_idx != 0])
        if present.numel() == 0:
            raise ValueError("ClassContrastLoss: the batch has no abnormal video")
        max_len = logits.shape[1]
        pad = torch.arange(max_len, device=logits.device)[None, :] >= lengths[:, None]
        masked = logits + torch.where(pad, constants.NEG_INF_MASK_VALUE, 0.0)
        attn = torch.softmax(masked / self.temperature, dim=1).unsqueeze(1)
        cols = (attn @ v_feats).squeeze(1)  # (B, E)
        col_class = cls_idx
        mined = (self._mine(logits, lengths, v_feats, cls_idx, pseudo_frame_label)
                 if self.mining == CONTRAST_TYPE_N3 else [])
        if mined:
            cols = torch.cat([cols, torch.stack(mined)], dim=0)
            col_class = torch.cat([cls_idx, cls_idx.new_zeros(len(mined))])
        rows = F.normalize(class_feats[present], dim=-1)
        sim = rows @ F.normalize(cols, dim=-1).T / self.cosine_temp  # (K, N)
        ignore = dup[present][:, col_class] if dup is not None else None
        loss, floor = class_infonce_loss(sim, present, col_class, ignore=ignore)
        if present.numel() > 1:  # video -> text only when the texts can be told apart
            ano = torch.where(cls_idx != 0)[0]
            v2t_ignore = dup[cls_idx[ano]][:, present] if dup is not None else None
            v2t, v2t_floor = class_infonce_loss(sim[:, ano].T, cls_idx[ano], present,
                                                ignore=v2t_ignore)
            loss, floor = (loss + v2t) / 2, (floor + v2t_floor) / 2
        masked = torch.zeros((), device=logits.device)
        if ignore is not None:
            other = (col_class[None, :] != 0) & (col_class[None, :] != present[:, None])
            if bool(other.any()):
                masked = (ignore & other).sum().to(sim.dtype) / other.sum()
        return {"loss": loss, "floor": floor, "masked_share": masked.detach(),
                "mined": torch.tensor(float(len(mined)), device=logits.device)}


def unique_target(key: str, dim: int, seed: int = constants.LNEG_UNIQ_TARGET_SEED) -> Tensor:
    """§28.3: a fixed unit vector for ``key`` (a source video id), meaningless by design.

    Drawn from its own generator, seeded by ``seed`` and the key's sha1, so it never touches
    the training RNG and is the same on every step, device and run.
    """
    digest = hashlib.sha1(f"{seed}:{key}".encode(), usedforsecurity=False).hexdigest()
    gen = torch.Generator().manual_seed(int(digest[:15], 16))
    return F.normalize(torch.randn(dim, generator=gen), dim=0)


def uniq_rows(cls_idx: Tensor, sources: list[str], dim: int) -> tuple[Tensor, Tensor]:
    """§28.3: ``(targets, idx)`` that make :class:`ClassContrastLoss` per source video.

    ``targets (1 + S, dim)``: row 0 is a placeholder for Normal (never a row of the loss),
    rows 1..S the :func:`unique_target` of each abnormal source in the batch, in first-seen
    order. ``idx (B,)``: 0 for a normal item, else its source's row, so abnormal windows of
    one source share a target and every other source is a negative.
    """
    if len(sources) != len(cls_idx):
        raise ValueError(f"{len(sources)} source ids for {len(cls_idx)} items")
    order: dict[str, int] = {}
    idx = torch.zeros_like(cls_idx)
    for b, (c, src) in enumerate(zip(cls_idx.tolist(), sources, strict=True)):
        if c != 0:
            idx[b] = order.setdefault(src, len(order) + 1)
    targets = torch.zeros(len(order) + 1, dim)
    for src, row in order.items():
        targets[row] = unique_target(src, dim)
    return targets.to(cls_idx.device), idx


CAPTION_SPACE = re.compile(r"\s+")
CAPTION_PUNCT = re.compile(r"[^\w\s-]")


def normalize_caption(text: str) -> str:
    """§30.1: lower case, punctuation and repeated spaces removed (merges identical captions)."""
    return CAPTION_SPACE.sub(" ", CAPTION_PUNCT.sub("", text.lower())).strip()


def caption_rows(
    cls_idx: Tensor, sources: list[str], captions: dict[str, str]
) -> tuple[list[str], Tensor]:
    """§30: ``(texts, idx)`` that make :class:`ClassContrastLoss` per caption.

    ``texts`` = the caption of each row 1..R, in first-seen order; abnormal windows whose
    sources share one normalized caption (one source, or identical captions, §30.1) share a
    row. ``idx (B,)``: 0 for a normal item, else its row. A source without a caption raises.
    """
    if len(sources) != len(cls_idx):
        raise ValueError(f"{len(sources)} source ids for {len(cls_idx)} items")
    order: dict[str, int] = {}
    texts: list[str] = []
    idx = torch.zeros_like(cls_idx)
    for b, (c, src) in enumerate(zip(cls_idx.tolist(), sources, strict=True)):
        if c == 0:
            continue
        if src not in captions:
            raise KeyError(f"source video {src!r} has an accident window but no caption "
                           "(KIM_RUN.md §30 C3)")
        key = normalize_caption(captions[src])
        if key not in order:
            order[key] = len(order) + 1
            texts.append(captions[src])
        idx[b] = order[key]
    return texts, idx


def same_source_negatives(cls_idx: Tensor, sources: list[str]) -> Tensor:
    """§30.1 (logged only): normal items whose source also has an abnormal item in the batch.

    Such a window is a negative column of its own video's caption row (kept on purpose, the
    same-scene hard negative). Mined parts are not counted: they are same-source by design.
    """
    if len(sources) != len(cls_idx):
        raise ValueError(f"{len(sources)} source ids for {len(cls_idx)} items")
    labels = cls_idx.tolist()
    abnormal = {src for c, src in zip(labels, sources, strict=True) if c != 0}
    count = sum(1 for c, src in zip(labels, sources, strict=True) if c == 0 and src in abnormal)
    return torch.tensor(float(count), device=cls_idx.device)


def near_duplicates(row_feats: Tensor, threshold: float = constants.LNEG_CAPTION_DUP_COS) -> Tensor:
    """§30.1: ``(R+1, R+1)`` bool, True where two different rows' text cosine >= threshold.

    Row 0 (Normal placeholder) and the diagonal are never duplicates.
    """
    unit = F.normalize(row_feats.detach().float(), dim=-1)
    dup = (unit @ unit.T) >= threshold
    dup.fill_diagonal_(False)
    dup[0, :] = False
    dup[:, 0] = False
    return dup


__all__ = [
    "CapContrastLoss",
    "ClassContrastLoss",
    "asymmetric_infonce_loss",
    "caption_rows",
    "class_infonce_loss",
    "near_duplicates",
    "normalize_caption",
    "same_source_negatives",
    "uniq_rows",
    "unique_target",
]
