# KAT-VAD v6 — LaGoVAD-AP (Agent Prompt branch): pre-registration

Branch `v6`. Proposal: `core/docs/final/Thesis proposal Deep learning-based traffic video anomaly detection.md`
(§3.2 branch, §3.3 training, §4.3 Step 0, §4.4 metrics). Plan: `.project/plans/katvad-v6-agent-prompt.md`.
Runbook: `colab/v6/ap_pilot.ipynb` (Drive `Thesis-V6/`). Written 2026-10-31, **before any v6 number**.

User decisions (2026-10-31): DoTA-CAP (pilot = CAP-dev, test = CAP-eval, sealed); **Nexar-only** training;
**protocol B at s3**; ego / non-ego reported beside every DoTA number; gate = the proposal's `tanh(α_o)`;
control = v5 **R1 s2024** (VideoMAE only, no MQR, 0.8102), read from its existing protocol-B output; **no v2
checkpoint or v2 eval is loaded** (user, 2026-10-31: "đừng làm gì liên quan tới v2 nữa"); zero-shot on DoTA-CAP-dev.
v2 artefacts are used only as **input data** (Nexar windows / census / KNN, the A2-row bakes, the CAP alignment).

## 0. Deviations from the proposal (all fixed here, before the read)

| # | proposal | v6 pilot | why |
|---|---|---|---|
| X1 | Step 0 + pilots on a DoTA-**train** pilot subset; test on DoTA's 1,305 list | Step 0 + pilot read on **DoTA-CAP-dev** (569); CAP-eval sealed for the final read | DoTA pixels exist only as MM-AU CAP frames (memory: DoTA frames unrecoverable); the project's DoTA endpoint is DoTA-CAP |
| X2 | P3 trains on CCD + BDD100K and Nexar | **Nexar `window` only** | user; the source shortcut control (proposal §3.3) becomes the main row |
| X3 | frame scores at native 10 fps, pooled metrics | **protocol B**: `s1[::3]` rows, scores interpolated to native frames, macro over CAP-dev, cluster bootstrap by source video | user; every v2 / v5 DoTA number is protocol B |
| X4 | centred VideoMAE window for every model | the **host** keeps A2's baked **causal** V2-S row (`W_u`, zero-init); the **centred** window is used for the agent RoI motion only | no host re-extraction; R1 reads the same causal V2-S motion, so the lag is paid by the control too |
| X5 | agent text alignment `L_a` (λ_a, `T_a`) | **not built** (λ_a = 0) | staged: detection + box hit first; `L_a` only if the branch passes the pilot |
| X6 | λ, K, β, crop chosen once on held-out P3 / the pilot subset | **fixed**: λ = 0.5, K = 1, β = 0.01, crop 1.5×, K_max = 16 | one-seed pilot; no grid is read on any test list |
| X7 | 3 seeds per row | **one seed (s2024)**, paired per clip with R1 s2024 | pilot; label PILOT (n = 1) |
| X8 | candidate set: agents + ego + no-agent | as proposal; the ego token = camera warp of each slot of the centred clip (6 × 8) + a fixed bottom-centre box | BoT-SORT's GMC gives the warp; no ego-speed sensor |

## 1. Step 0 — go / no-go (proposal §4.3, applied mechanically by `v6_agents step0`)

On DoTA-CAP-dev, agent caches built by §2.1. VO / OO (`obstacle`, `leave_to_left`, `leave_to_right`) and clips with
no annotated box are excluded from D0 / D2 and counted.

* **D0** track recall: a DoTA anomalous-agent track (`obj_track_id`) is *found* if a kept track's box (linear between
  detections) overlaps it at IoU ≥ 0.3 (and ≥ 0.5, printed) on ≥ 50 % of its annotated frames. Strata: ego / non-ego
  (`ego_involve`), night, size bin of the track's median box (< 2,267 px at 1280 × 720, < 2 %, 2–8 %, > 8 %),
  category. *kept recall* = the same with only the top-16 tracks of each frame.
* **D1** load: agents per row, share of rows at K_max, kept | found.
* **D2** training-free ranking: per abnormal annotated row (inside `[anomaly_start, anomaly_end]`), hit = the top agent
  under a score overlaps a GT box at IoU ≥ 0.3 (0.5 printed). Scores: VideoMAE RoI change over 6 rows (18 frames,
  non-overlapping windows), CLIP crop change over 1 row, CLIP crop · (abnormal − normal text), looming ρ, approach
  (−d gap / dt); priors centre-nearest, largest, lowest; chance (share of overlapping agents) and ceiling (any agent
  overlaps). Per-clip means, cluster bootstrap 10,000 by source video, seed 2026; Δ vs centre paired per clip.
* **GO**: overall recall@0.3 ≥ 0.80, non-ego ≥ 0.70, smallest bin ≥ 0.70, **and** ≥ 1 D2 score − centre with 95 % low > 0.
  **NARROW**: overall ∈ [0.60, 0.80), or a stratum < 0.70, or no D2 score beats centre (then H2 rests on learning);
  strata < 0.50 are dropped from H1 / H2. **NO-GO**: overall < 0.60 → the branch is not trained (notebook stops).

## 2. Build (fixed)

### 2.1 Agent cache (`core/agents/`, `core/tools/v6_agents.py extract-{dota,nexar}`)

* Detector COCO **YOLOv9-C** (`yolov9c.pt`), imgsz 1280, conf 0.10, classes person / bicycle / car / motorcycle / bus /
  truck; tracker **BoT-SORT**, GMC `sparseOptFlow`, no ReID, lost after 10 detection frames; detection at **10 fps**
  (DoTA every frame, Nexar every `round(fps / 10)`-th). Tracks < 3 detections dropped; a person overlapping a
  two-wheeler (IoU or containment > 0.5 on ≥ 50 % of co-visible frames) is merged into it as `rider`. Track ids
  renumbered 1..n. The ultralytics version is recorded in `track_manifest.json`.
* Rows: DoTA-CAP at `s1[::3]` (= protocol B), Nexar at A2's `s1[::8]`; the row count must equal A2's bake (Nexar) /
  `DoTA_s1_ncc` frames (DoTA), else the video is reported and not stored.
* Per row ≤ 16 agents alive at the row frame, by track-mean confidence. Per agent (913 columns): valid, tracker box
  (normalized xyxy), track-mean confidence, class, track id, 9 kinematics (ego-compensated velocity and acceleration in
  box heights / s, ρ = d log h / dt, age, nearest-agent gap and its rate, ego gap; capped at ±10), **V2-S RoI motion**
  (centred 16-frame clip, step 3 Nexar / 1 DoTA; per slot 2 × 2 RoIAlign on the 14 × 14 tokens over the union of the
  slot's two boxes, ≥ one token wide; slot mean), **CLIP crop** `image_embeds` (square 1.5 × long side, ≥ 32 px,
  zero-padded, squash 224). Ego block 48. float16. `agent_manifest.json` binds stride / clip step.

### 2.2 Model (`core/models/agent_branch.py`, `agent.*`)

Host = v2 Nexar **A2** exactly (`v2.motion=V2-S`, `model.motion_dim=384`, no CRN, KIP off, DVS `span`, kernel /
top-k of T2). Branch: d = 256, 4 heads; tokens = Fourier(box corners) + type + class + MLP(log-area, aspect, conf,
ego distance) + `W_a LN(crop)` + `W_m LN(motion)` + MLP(κ); ego and no-agent tokens; one instance self-attention with
an MLP pairwise bias (Δbox, IoU, centre gap); relation module 12 queries × 2 blocks (q ← agents, q ← scene
`LN([x; u])`, agents ← q), `z_t` = query mean. Fusion `h = x + W_u u + tanh(α_o) W_o LN(z_t)`, **α_o = 0** (asserted:
the AP model with A2's weights reproduces A2's logits exactly, `test_alpha_zero_is_a2`). Scorer 256 → 128 → 1, last
layer zero (p = 0.5 at init). α_o and the queries: no weight decay. AdamW / LR / schedule / steps (1,749) = A2's.

### 2.3 `L_agent` (`core/losses/agent_mil.py`), λ = 0.5

`S_t = max_i p_{t,i}` over valid candidates; ŝ = mean of the top-K (K = 1) candidates of the top-k steps
(k = max(⌊L/16⌋, 1)). Abnormal (incl. DVS-synthesized): BCE(ŝ, 1) + β · mean p (β = 0.01); a synthesized sequence's
pseudo-normal steps (minus 3 rows = 8 × 3 frames / stride 8 around each inserted boundary) enter the dense negatives.
Normal: mean BCE(p, 0) over every candidate of every valid step. No GT box or frame label.

## 3. Read-out (`v6_agents readout`, printed; nothing below changes the build)

* Detection on DoTA-CAP-dev, protocol B, rows **AP s2024** and **v5 R1 s2024** (same harness and clips). Per row × {all, ego, non-ego}: macro (per-clip AUC, cluster CI), micro (pooled per-clip
  min-max), **strat-50** (50 relative-position bins, pooled AUC within bins, mean over two-class bins; the primary
  position-controlled number). Paired macro Δ **AP − R1** (primary contrast).
* Box-level hit of AP's top candidate (an ego / no-agent argmax is a miss) at IoU 0.3 / 0.5 vs centre / largest /
  lowest priors and chance, unconditional and given detected; paired AP − prior.
* Mechanics from `metrics.jsonl`: `agent`, `agent_alpha`, `agent_share`, top-candidate p on abnormal vs normal,
  ego / no-agent pick shares.
* **Pilot pass** (to 3 seeds + CAP-eval): AP − R1 macro ≥ 0 on all **and** non-ego strat-50 ≥ R1's, **and** box
  hit − centre 95 % low > 0 (proposal §4.5 decision gates). Otherwise PILOT-FAIL, reported as a negative result.

## 4. Post-hoc read: v5 R1 s2024 zero-shot on the official DADA-2000 test set (printed only)

Written 2026-11-01, **after** the §3 read-out (PILOT-FAIL: AP − R1 macro −0.014 [−0.027, −0.002]) and **before
any DADA number of R1**. User decisions (2026-11-01): set = the official list used by the papers (option a), **R1
only** (AP is not read on DADA: no agent cache exists there). Nothing in this section changes §3's verdict or
licenses an AP claim; it answers "what does the control score on DADA-2000 zero-shot".

| # | choice |
|---|---|
| Z1 | **Set** = SimpleTAD's (CVPR 2025) DADA-2000 evaluation list, `DADA2K_my_split/validation.txt` in `D2K.zip` (`constants.V6_DADA_OFFICIAL_URL`): every clip of the original release **not** in LOTVS's training list (`simple-tad/data_tools/dada/prepare_anno_dada2000.py`). `{type}/{video}` → `t{type:02d}_v{video:03d}`. Not this project's T2-test (370 sources, a seeded split), and not "dev + eval" (that is DoTA-CAP). |
| Z2 | **Coverage gate**: a listed clip is read iff it has an annotation row in `dada标注.xlsx`, frames on disk (census), an anomaly start inside them (`make_record`), and the DADA2000_orig CLIP s8 + V2-S s8 caches. Every exclusion is named by reason (`v6_r1_dada ids`); **coverage < 0.95 stops** (no number). |
| Z3 | **Model** = v5 R1 s2024 with its parameter-free mean readout removed (`v6_r1_dada strip`): same weights, `v5.readout = none`, reading A2's baked `u` (the same term; PREREG_V5 S1-G2(b)). Inputs = DADA rows baked with **R1's own Nexar A2 statistics** (`build_v2_inputs apply`, `A2_S/Nexar_s8`). |
| Z4 | **Gate G-S** (before DADA): stripped R1 on DoTA-CAP-dev (protocol B, A2 bake) vs R1's stored protocol-B scores: \|Δ macro\| ≤ 1e-3 **and** max per-frame \|Δ score\| ≤ 1e-2. FAIL stops. |
| Z5 | **Protocol** = `dada_whole_eval` unchanged (§29 D-N20): whole source video, every s8 row in one forward, labels = the annotation span at s8 (Gate D0 rule), macro over two-class videos (95 % bootstrap over videos), micro pooled, `t/N` and middle rulers beside. One seed → no seed interval. |
| Z6 | **Wording.** One sentence: "R1 (Nexar-trained, zero-shot) scores macro X / micro Y on the official DADA-2000 test list (N of M clips), against a middle-position ruler of R". SimpleTAD's D2K numbers are **fully supervised on DADA** and use their own frame annotation: printed in another column with that note, never as a comparison. Not licensed: any AP claim, any DADA claim for v2's `B`. |
