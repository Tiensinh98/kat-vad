# Nexar collision prediction — setup and runbook (v2)

Plan: `.project/plans/katvad-v2-nexar-feasibility.md` (scope chosen 2026-10-09: **N3 zero-shot read, then training;
the 750 negative videos are in scope behind the N-X gate**).

## 0. What the release is (measured from the HF repo listing, 2026-10-09)

| Path | Content |
|---|---|
| `train/positive/{id}.mp4` + `metadata.csv` | 750 videos (README: collision or near-miss; the label is the **folder**) |
| `train/negative/{id}.mp4` + `metadata.csv` | 750 videos, regular driving |
| `test-public/`, `test-private/` | ~10 s clips ending 0.5 / 1.0 / 1.5 s **before** the event (anticipation) |
| `solution.csv`, `time_to_accident_test_map.csv`, `evaluate_submission.py` | official test labels + mAP script (clip-level) |

`metadata.csv` follows the `videofolder` layout (`file_name` + `time_of_event`, `time_of_alert`, `light_conditions`,
`weather`, `scene`; the README example shows no collision / near-miss column — the census records the real columns).
Train mp4 ≈ 25.5 GB; whole repo ≈ 31.4 GB. Access is gated: accept the license on the dataset page with the account
whose read token is the Colab secret `HF_TOKEN`.

**Not used for detection:** the official test clips stop before the event, so they hold no anomalous frame.

## 1. N0 — mirror + census + split

| Step | Where | Command / notebook | Output |
|---|---|---|---|
| Mirror + census | Colab (CPU) | `colab/v2/nexar_n0_census.ipynb` | `Thesis/data/Nexar/raw/` (whole repo), `outputs/v2/REPORTS/nexar_n0/census.{json,md,log}` |
| Freeze split | local | `python -m core.tools.nexar_splits --census outputs/v2/REPORTS/nexar_n0/census.json` | `core/splits/v2/nexar_{train,val,test}.txt` + `NEXAR_MANIFEST.json` |
| Verify | local / CI | `... --check` | raises if the committed files differ |

**Census** (`core.tools.nexar_census`): metadata × container headers (PyAV, no decode) + `--verify-decode K` full
decodes. Prints length by class and the length ruler AUC (C28), fps / resolution / codec by class, header-vs-decode
frames (C10), `t_event / duration`, alert → event gap, the abnormal share under `[t_alert, t_event + d]` for
`d ∈ {0, 1, 2} s` (input to N1's span rule, not a choice), and counted annotation issues. Nothing is dropped.

**Split rule** (`core.tools.nexar_splits`, fixed before the census was read): one video = one group; stratum =
label × duration half (pooled median) × tercile of `t_event / duration` for positives (`pna` if the annotation is
unusable); `nexar_test` = 25 % (**sealed**, `load_split(..., final=True)` only), `nexar_val` = 15 %, `nexar_train` =
60 %; seed `V2_SPLIT_SEED`. Every video is placed, including those with issues; which a corpus keeps is N1's call.

### 1.1 Census read-out (run 2026-10-09, `outputs/v2/REPORTS/nexar_n0/census.md`)

| Quantity | Measured | Consequence |
|---|---|---|
| Videos | 1,500 (750 / 750), 0 probe failures, 0 annotation issues | every video placed in a split |
| Columns | `file_name, light_conditions, scene, time_of_alert, time_of_event, time_to_accident, weather` | **no collision / near-miss field**; `time_to_accident` empty for all → positives are one class |
| Frames | 1,701,166 header frames; header vs duration × fps mismatched 0; 20 full decodes max diff 0 | header counts can be trusted (C10) |
| Length | median 40.1 s both classes; length ruler AUC **0.5366** | small whole-video leak; windows close it |
| fps | 1,406 / 1,500 in [29.5, 30.7]; 31 below 29, 13 below 28 (min 23.6) | stride in frames drifts on ~2 %; rule in N1 |
| Resolution / codec | 1280×720 h264, all 1,500 | no camera-model proxy by class |
| `t_event / duration` | median **0.496**, IQR 0.482–0.511, **730 / 750 in [0.4, 0.6)** | a position prior is near-perfect on whole videos → whole-video macro is a position test |
| alert → event | median **1.43 s** (p5 0.40, p95 3.28) | ≈ DADA's start → accident median 1.50 s (all 1,945 rows) |
| Context | pre-alert median 18.3 s, post-event median 20.2 s (p5 8.5 s both) | room for in-video negatives and shifted crops |
| Abnormal frame share | `[t_alert, t_event + d]`: d = 0 → 2.1 %, 1 s → 3.4 %, 2 s → 4.8 % (corpus) | input to the span rule, not a choice |
| Metadata × label | Urban 475 vs 309, Sub-urban 75 vs 196, Rain 60 vs 23 | scene / weather separate the classes at clip level — relevant to N-X and micro, not to macro |

### 1.2 Frozen split (2026-10-10)

`python -m core.tools.nexar_splits --census outputs/v2/REPORTS/nexar_n0/census.json` (census sha1 `58a4aaf0…`), `--check`
passes. Seed 2024, position terciles at 0.4869 / 0.5055, median duration 40.13 s.

| Split | Videos | neg / pos | sha1 |
|---|---:|---|---|
| `nexar_train` | 900 | 449 / 451 | `6768538a…` |
| `nexar_val` | 225 | 113 / 112 | `cdbc0aaf…` |
| `nexar_test` (**sealed**) | 375 | 188 / 187 | `198e0889…` |

## 2. N2–N5 — extract, build, train `whole` (then `window` iff triggered), read val (`colab/v2/nexar_train.ipynb`)

Rules: `PREREG_ADDENDUM.md` §21 (D-N1…D-N10). One notebook, resumable at every step:

| Step | Tool | Output (Drive) |
|---|---|---|
| N2 extract (GPU) | `core.tools.nexar_extract` — one PyAV decode per mp4, one resize, CLIP and V2-S normalizations (bit-exact to `preprocess_frames(center_crop=False)` / `squash_frames`) | `Thesis-V2/cache/clip/Nexar_s1_ncc/`, `Thesis-V2/cache/video/<V2-S>/Nexar_s1_squash/`, `REPORTS/nexar_n2/` |
| N4 corpora | `core.tools.nexar_build corpus` — `Nexar_whole` (all `nexar_train`, video labels) and `Nexar_window` (20-row windows at s8, hop 8, positives only, no cap); epochs per construction for the 1,740-step budget | `Thesis-V2/data/Nexar_{whole,window}/`, `REPORTS/nexar_n4/build_report.json` |
| A0 input | `core.tools.nexar_build subsample` (`s1[::8]`) | `Thesis-V2/cache/clip/Nexar_s8_ncc/` |
| A3 input | `core.tools.build_v2_inputs fit-ids` — R2 + V2-S statistics on `nexar_train`, train + val baked at `s1[::8]` | `Thesis-V2/cache/v2/A3_R2_S/Nexar_s8/` |
| KNN | `core.data.knn_cache --dataset Nexar` on `Nexar_s8_ncc` | `Thesis-V2/cache/knn/Nexar_{c}/` |
| N5 train | `core.train`, E3 recipe (`data.is_egocentric=false` and `L_neg` off, as E3), `data.dataset=Nexar`; `whole` adds `dvs.syn_max_num_clips=3`; only the notebook's `CONSTRUCTIONS` (default `['whole']`) | `Thesis-V2/outputs/v2_nexar/<c>/<arm>/s<seed>/` |
| val read | `core.tools.nexar_eval` — shifted 8 s crops (endpoint) + whole videos + middle ruler | `REPORTS/nexar_val/<tag>_<c>_<arm>/` |
| zero-shot | `nexar_eval` on E3's five T2-trained A3 ckpts (`outputs/v2_e3/A3/s*`) with their T2 stats (`cache/v2/A3_R2_S/DADA2000_orig`) — W2's bar | `REPORTS/nexar_val/zeroshot_e3_A3/` |
| trigger | `core.tools.nexar_trigger` (pilot only): W1 edge drop > 0.05, W2 endpoint ≤ zero-shot, W3 clip AUC ≥ 0.95 and A3 < A0 | `REPORTS/nexar_val/trigger_pilot/window_trigger.{json,md}` |
| 2nd look | `build_v2_inputs apply` (Nexar stats on DoTA-CAP-dev) + `protocol_b_eval` | `REPORTS/nexar_dota_cap/<tag>_<c>_<arm>/` |

Pilot first (`SEEDS = [2099]`, `whole` × {A0, A3} = 2 runs) → trigger. Passes → `SEEDS = RUN_SEEDS` on `whole` (10 runs). Fires →
add `'window'` to `CONSTRUCTIONS`, pilot it, then `RUN_SEEDS` for both (20 runs; D-N9(b)). `nexar_test` is opened only by
`nexar_eval --split nexar_test --final`, once, after §21 is signed.
