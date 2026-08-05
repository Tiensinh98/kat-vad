"""Device resolution shared by every CLI entry point."""

from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger(__name__)

VALID_DEVICE_SPECS = ("auto", "cuda", "mps", "cpu")


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec (``auto``/``cuda``/``mps``/``cpu``) to a torch device.

    ``auto`` prefers CUDA, then MPS, then CPU. Explicit specs fail loudly if the
    backend is unavailable instead of silently falling back.
    """
    if spec not in VALID_DEVICE_SPECS and not spec.startswith("cuda:"):
        raise ValueError(f"Unknown device spec {spec!r}; expected one of {VALID_DEVICE_SPECS}")

    if spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        LOGGER.info("Auto-resolved device: %s", device)
        return device

    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")

    device = torch.device(spec)
    LOGGER.info("Using requested device: %s", device)
    return device
