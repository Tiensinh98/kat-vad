# Kill-switch K (T2 only) — read-out

Pre-registration: `core/docs/v2/PREREG_ADDENDUM.md` §6.1 · commit `UNKNOWN (git said: 'fatal: not a git repository (or any parent up to mount point /content)\nStopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).')`
Sources: 295 two-class (12409 frames), ids sha1 `634a105d7175593f01aeb64fb1d8f0b8d6176732` · dims x 512 / u 768
Encoder cache manifest: `{"assumed_fps": 30, "clip_frame_step": 3, "clip_frames": 16, "encoder": "vit_b_k710_dl_from_giant", "feature": "fc_norm(mean over tokens)", "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225], "pad": "clamp to frame 0 (repeat first frame)", "repo_id": "OpenGVLab/VideoMAE2", "revision": "706cc172d65ebd4dedbee3f9c0183a93df9fa125", "stride": 8, "transform": "squash224_imagenet", "weights_sha256": "8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec"}`

| probe | x | u | [x;u] | Δ([x;u] - x) |
|---|---|---|---|---|
| source-grouped CV (in-domain) | +0.6147 [+0.5909, +0.6372] | +0.7597 [+0.7401, +0.7806] | +0.7472 [+0.7264, +0.7681] | +0.1325 [+0.1093, +0.1551] |
| type-grouped CV (held-out accident types; NOT a domain transfer) | +0.6138 [+0.5887, +0.6373] | +0.7596 [+0.7390, +0.7819] | +0.7401 [+0.7184, +0.7627] | +0.1263 [+0.1010, +0.1517] |

Macro = mean per-source frame AUC; 95 % percentile bootstrap over sources (B = 2000).
x-only (i') beside Gate D0 0.6518 (different subset, same label convention) — printed, not gated.

## Verdict: **GO**

KILL iff both Δ have upper < +0.03; positive control = u under i_source with CI lower > 0.5 (fails → SUSPECT_PIPELINE, no KILL).
