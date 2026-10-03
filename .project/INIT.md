# Project Initialization — KAT-VAD

> Session-check anchor for CLAUDE.md §1. This file says **what the project is and where to look**;
> it does not hold state. Current state lives in `.project/memory-bank/activeContext.md` (TL;DR at the top).

## Identity

- **Project:** KAT-VAD — Kinematics-Aware, definition-conditioned Traffic Video Anomaly Detection (internship thesis).
- **Baseline:** LaGoVAD (weakly-supervised, language-definition-conditioned VAD). Reference code in
  `LaGoVAD-PreVAD/` — **read-only**.
- **Owner:** sinhpham (phamtiensinh010@gmail.com). Advisor reviews ("thầy") are labelled as such; since 2026-09-29
  the user authorizes pre-registration amendments directly (marked "not advisor-reviewed").

## Branches (check `git branch --show-current` first)

| Branch | Line | Memory bank |
|---|---|---|
| `main` | KAT-VAD v1 (KIP: PMG head + frozen-MLP shift + motion-score head) | v1 copy |
| `v2` | v1 code + the v2 build line (frozen VideoMAE V2 motion input + CRN, no KIP/RAFT) | v2 copy |
| `v3` | v1 + KIP gate rebuild (`gate_type`, ECMR) | v3 copy |

The memory bank is tracked per branch and the copies diverge on purpose — never merge them with "take theirs".

## Where things are

| What | Path |
|---|---|
| Implementation (only place to write code) | `core/` |
| Docs (source of truth) | `core/docs/` — v2: `core/docs/v2/` (`KAT_VAD_PROPOSAL_v2.md`, `KAT_VAD_v2_ARCHITECTURE.md`, `PREREG_ADDENDUM.md`) |
| Plans | `.project/plans/` — v2 live plan: `katvad-v2-e0-e2.md` (§0 = status table) |
| Frozen splits (v2) | `core/splits/v2/` + `SPLITS_MANIFEST.json` |
| Colab runbooks | `colab/` (v2: `colab/v2/`) |
| Memory bank | `.project/memory-bank/` (+ `lessons-learned/`) |
| Preferences | `.project/preference.md` |

## Environment

- Dev: macOS, CPU, `.venv` (`source .venv/bin/activate`); tests are data-free.
- Compute: Google Colab. Real-data runs ship as `.ipynb`; data never comes back to the local machine.
- Drive: `Thesis/` = data, caches, ckpts, v1 runs (read-only); `Thesis-V2/` = code (`kat-vad/`, uploaded, not a git
  checkout → `commit: UNKNOWN` is expected) + v2 outputs.

## Standing rules (short form — CLAUDE.md is authoritative)

- Never auto-commit; commit only the batch the user names.
- Pre-registered rules are fixed before the number is read; a defective rule becomes a numbered Amendment.
- DoTA is held out (never trained on); DoTA-eval is sealed until the final read.
- Quality gate before commit: `ruff`, `mypy`, `bandit`, `pycycle`, `pyright` (CLAUDE.md §11).
