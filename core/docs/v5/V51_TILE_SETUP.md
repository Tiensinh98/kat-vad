# v5.1 tile-scored MQR — run setup (PREREG_V5 §3, signed 2026-10-29)

Runbook: `colab/v5/v51_tile_pilot.ipynb` (A100 High-RAM, `/content` ≥ 80 GB). Upload this branch (`v5-probe`) to
`Thesis-V5/kat-vad` first.

| step | tools | output (Drive, `Thesis-V5/`) |
|---|---|---|
| 1 grids | `v5_stage1 extract --tile` (Nexar train s8, val s3), `v5_stage0 extract --tile` (DoTA-CAP s3, 5 CAP groups), `v5_stage1 head` | `cache/video_grid/<V2-S>/{Nexar_s8,Nexar_s3,DoTA_CAP_s3}_grid4x7/` (≈ 31 GB) |
| 2 train | `core.train` × R1c / R3c / R6c, s2024; `v5_stage1 provenance` | `outputs/v5/v5_tile/<row>/s2024/` |
| 3 reads | `nexar_eval`, `protocol_b_eval` (+ `--reverse-time`), `v5_maps dump` | `outputs/v5/v5_tile/{eval,maps}/` |
| 4 read-out | `v5_maps read`, `v5_tile_read` | `outputs/v5/v5_tile/REPORTS/v51_pilot_readout.md` |

**Gates that stop the notebook:** S1-G1 length (Nexar grids), G-c (every CAP-dev clip stored), J10 + S1-G4
provenance, finite last-step losses. The pilot outcome (PILOT-PASS / FAIL per candidate) is in the read-out.

**Parallelism.** Steps 1–3 run independent jobs as separate processes on the one GPU (`MAX_GPU_JOBS`, default 3;
`OMP_NUM_THREADS` split between them). Grids are written to VM disk and mirrored to Drive every `MIRROR_EVERY`
seconds; after a disconnect step 1 restores Drive's partial copy and the extractors skip finished videos. CUDA OOM →
set `MAX_GPU_JOBS = 1`, re-run the cell (finished jobs skip on their records).

**What the read-out prints (proposal §4.4).** DoTA-CAP-dev clip-macro AUC, Δ vs R1c (95 % cluster bootstrap) and TOST
±0.01; position-controlled AUC (a) cross-clip same-position pairs and (b) `z(y) + 2 z(p)` minus position alone;
shift test; time-reversed − forward; AP (clip macro + pooled min-max); AUC and (b) by ego-involved / ego-uninvolved
and box-share bin; maps: hit − chance / centre / change map, mass-in-box, STAUC beside uniform / centre / change maps;
`nexar_val` crop endpoint (guardrail); mechanics (losses finite, `p` abnormal vs normal, `T`, entropy / log 196).
All numbers are **PRELIMINARY (n = 1 seed)**.
