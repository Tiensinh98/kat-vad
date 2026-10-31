# Plan — KAT-VAD v6 (LaGoVAD-AP), branch `v6`

Rules `core/docs/v6/PREREG_V6.md`. Runbook `colab/v6/ap_pilot.ipynb`. Drive `Thesis-V6/` (code `kat-vad/`, caches
`cache/{agents,tracks}/`, outputs `outputs/v6/`); `Thesis/`, `Thesis-V2/`, `Thesis-V5/` read-only.

## §0 Status

| step | what | state |
|---|---|---|
| P0 | prereg + deviations X1–X8 | written 2026-10-31 |
| P1 | code: `core/agents/{layout,geometry,tracking,features}.py`, `core/data/agent_cache.py`, `core/models/agent_branch.py`, `core/losses/agent_mil.py`, `agent.*` config, train / dataset / protocol-B wiring, `core/tools/v6_agents.py` | **built**, suite 1445 / 0 fail (2026-10-31), not committed |
| P2 | notebook step 1: CAP-dev agents | not run |
| P3 | Step 0 (go / no-go) | not run |
| P4 | Nexar agents (sharded) | not run (only after GO / NARROW) |
| P5 | train AP s2024 | not run |
| P6 | protocol B AP, logit dump, read-out vs v5 R1 s2024 (no v2 ckpt) | not run |
| P7 | if pilot passes: 3 seeds, CAP-eval (sealed, `--final`), `L_a` | not planned yet |

## §1 Risks

* **Detector recall on small / night agents** — Step 0 measures it before any training; NARROW drops strata.
* **Salient normal agent wins** (largest / central lead car) — dense negatives; box hit vs largest / centre printed.
* **Ultralytics API drift** — BOTSORT constructor / YAML loader handled both ways; version pinned `8.3.*` in the
  notebook and written to the manifest. If `update()` changes its output columns, `track_video` raises on reshape.
* **Extraction cost** — Nexar ≈ 1 decode + 10 fps detection + 1 VideoMAE clip / row + ≤ 16 crops / row per source;
  3 shards on one GPU. RAM per shard ≈ 0.8 GB of frames.
* **Host lag (X4)** — A2's causal row centres motion 0.75 s early; the agent RoI uses the centred clip. Paid by the
  control too; a shift test is not part of the pilot.

## §2 Next

User: upload branch `v6` to `Thesis-V6/kat-vad`, run steps 1–2, paste `REPORTS/step0.md`; then steps 3–6 if GO /
NARROW, paste `REPORTS/v6_readout.md`.
