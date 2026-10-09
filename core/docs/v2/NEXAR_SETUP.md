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

## 2. Next phases

N1 (pre-registration: span rule, protocol N-B, gates, arms) is written after the census and **before** any feature is
scored. N2–N6 are in the plan.
