# KAT-VAD — Implementation Spec (for Claude Code)

Implementation-only. No motivation, no comparisons. This document tells you exactly what to build, with per-step tensor shapes.

**What you already have:** the **LaGoVAD** source code (baseline). Build base on it with organizing NOT COPY CODE.
**What you must implement from scratch (no source available):** the **KIP** module (derived from Pi-VAD PMG + RefineVAD MoTAR) and its losses — these are described step-by-step below.
**What to reuse from LaGoVAD as-is:** frozen CLIP encoders, temporal encoder, co-attention fusion `U`, detection head `H_bin`, classification head `H_mul`, `L_MIL`, `L_MIL-align`, `L_dvs` (dynamic video synthesis), `L_neg` (hard-negative mining).

Everything is in feature space (CLIP features per sampled frame). Notation: `L` = number of sampled frames in a video (variable), `D = 512` (LaGoVAD hidden size), `C` = number of categories in the anomaly definition `Z`.

---

## 1. Offline flow-target extraction (train-time only) — `flow/raft_extract.py`

Run once, before training. Produces the ground-truth flow embeddings `e_O` that supervise the PMG head. **Not used at inference.**

**Per training video:**

1. Load raw video, sample frames identically to the RGB pipeline (every 8 frames; DoTA at its provided 10 fps). Result: `L` frames.
2. For each adjacent sampled pair `(frame_i, frame_{i+1})`, run **RAFT** → dense flow field `flow_i ∈ ℝ^{2×H×W}`. For the last frame, duplicate the previous flow (so you get `L` fields).
3. Pool each flow field into a fixed vector. **Pooling method:** resize flow to `H'×W'` (e.g. 24×24), flatten, project with a small **frozen random-init or PCA** projection to `d_O = 256`. (Simplest robust choice: compute per-field statistics — mean/std/max of magnitude and of angle histograms — then a fixed linear map to 256-d. Keep it deterministic so targets are stable across epochs.)
   - Input: `flow_i ∈ ℝ^{2×H×W}` → Output: `e_{O,i} ∈ ℝ^{256}`.
4. Stack: `e_O ∈ ℝ^{L×256}`. Save to disk keyed by video id: `cache/flow/{video_id}.npy`.

**Output of this stage:** one `e_O ∈ ℝ^{L×256}` per training video, cached. Dataloader loads it alongside the RGB features during training only.

> Implementation note: use the official RAFT (`princeton-vl/RAFT`) with pretrained weights. Batch the pair-wise passes. This is the only heavy step and it is amortized (run once).

---

## 2. Pseudo-flow generation head (PMG) — `kip/pmg.py`

Derived from Pi-VAD PMG, single modality (optical flow). Autoencoder-style translator: RGB-temporal features → flow embedding.

**Module:** `PMGFlowHead(nn.Module)`

**Input:** `v^t ∈ ℝ^{L×512}` (output of LaGoVAD temporal encoder).
**Output:** `ê_O ∈ ℝ^{L×256}` (pseudo-flow embedding).

**Layers (exact):**

| #   | Layer             | Config                           | Input shape   | Output shape  |
| --- | ----------------- | -------------------------------- | ------------- | ------------- |
| 1   | Conv1d encoder    | in=512, out=128, kernel=3, pad=1 | `(B, 512, L)` | `(B, 128, L)` |
| 2   | GELU              | —                                | `(B,128,L)`   | `(B,128,L)`   |
| 3   | Linear translator | 128→128 (applied per time step)  | `(B, L, 128)` | `(B, L, 128)` |
| 4   | GELU              | —                                | `(B,L,128)`   | `(B,L,128)`   |
| 5   | Conv1d decoder    | in=128, out=256, kernel=3, pad=1 | `(B, 128, L)` | `(B, 256, L)` |

Notes:

- Conv1d operates on `(B, channels, time)`; transpose to `(B, L, C)` for the Linear, transpose back for the decoder conv.
- `B` = batch of videos (LaGoVAD pads/masks variable `L` — reuse its collate/mask).
- Return `ê_O` as `(B, L, 256)`.

```python
class PMGFlowHead(nn.Module):
    def __init__(self, d_in=512, d_lat=128, d_flow=256):
        super().__init__()
        self.enc = nn.Conv1d(d_in, d_lat, 3, padding=1)
        self.translator = nn.Linear(d_lat, d_lat)
        self.dec = nn.Conv1d(d_lat, d_flow, 3, padding=1)
        self.act = nn.GELU()
    def forward(self, vt):                 # vt: (B, L, 512)
        h = self.enc(vt.transpose(1, 2))   # (B,128,L)
        h = self.act(h).transpose(1, 2)    # (B,L,128)
        h = self.act(self.translator(h))   # (B,L,128)
        eo = self.dec(h.transpose(1, 2))   # (B,256,L)
        return eo.transpose(1, 2)          # (B,L,256)
```

---

## 3. Kinematic gate + adaptive temporal shift — `kip/gate_shift.py`

Derived from RefineVAD MoTAR steps 2–4, with the gate signal driven by the **pseudo-flow norm** instead of feature variance. Shape-preserving.

**Module:** `KinematicShift(nn.Module)`

**Inputs:**

- `v^t ∈ ℝ^{L×512}` (features to shift)
- `ê_O ∈ ℝ^{L×256}` (from PMG; drives the gate)

**Output:** `v^k ∈ ℝ^{L×512}` (recalibrated features; same shape as `v^t`).

**Hyperparameter:** folding factor `K = 4` → max channels shiftable per direction `D/K = 128`.

**Step-by-step:**

| Step | Operation           | Formula                                                | Input           | Output              |
| ---- | ------------------- | ------------------------------------------------------ | --------------- | ------------------- |
| 1    | Kinematic intensity | `m_t = ‖ê_{O,t}‖₂` (L2 over the 256 dims)              | `ê_O (B,L,256)` | `m (B,L)`           |
| 2    | Normalize m         | per-video min-max to [0,1] (over time)                 | `m (B,L)`       | `m̂ (B,L)`           |
| 3    | Shift-ratio MLP     | `r_t = σ(W₃ GELU(W₂ GELU(W₁ m̂_t)))` ; widths 1→16→16→1 | `m̂ (B,L)`       | `r (B,L)`, in [0,1] |
| 4    | Channel count       | `s_t = floor(r_t · 128)`                               | `r (B,L)`       | `s (B,L)` int       |
| 5    | Bidirectional shift | see below                                              | `v^t, s`        | `v^k (B,L,512)`     |

**Step 5 detail (per time index `t`, per video):**

```
v^k_t[0    : s_t ]   = v^t_{t-1}[0    : s_t ]     # from PAST neighbor
v^k_t[s_t  : 2s_t]   = v^t_{t+1}[s_t  : 2s_t]     # from FUTURE neighbor
v^k_t[2s_t : 512 ]   = v^t_t   [2s_t : 512]       # keep PRESENT
```

- Boundary: `t=0` has no past → zero-fill that slice; `t=L-1` has no future → zero-fill.
- Because `s_t` varies per time step, implement with a vectorized gather or a per-video loop over `t` (L is small, typically < 400; a masked-index implementation is fine). Respect the LaGoVAD padding mask so padded positions are untouched.

```python
class KinematicShift(nn.Module):
    def __init__(self, d=512, K=4):
        super().__init__()
        self.d, self.max_s = d, d // K            # 128
        self.mlp = nn.Sequential(nn.Linear(1,16), nn.GELU(),
                                 nn.Linear(16,16), nn.GELU(),
                                 nn.Linear(16,1))
    def forward(self, vt, eo, mask=None):         # vt:(B,L,512) eo:(B,L,256)
        m = eo.norm(dim=-1)                        # (B,L)
        mmin = m.amin(1, keepdim=True); mmax = m.amax(1, keepdim=True)
        mhat = (m - mmin) / (mmax - mmin + 1e-6)   # (B,L)
        r = torch.sigmoid(self.mlp(mhat.unsqueeze(-1))).squeeze(-1)  # (B,L)
        s = (r * self.max_s).floor().long()        # (B,L)
        B, L, D = vt.shape
        past = torch.zeros_like(vt); fut = torch.zeros_like(vt)
        past[:,1:,:] = vt[:,:-1,:]; fut[:,:-1,:] = vt[:,1:,:]
        vk = vt.clone()
        for b in range(B):                         # L small; simple + correct
            for t in range(L):
                st = int(s[b,t].item()); st2 = min(2*st, D)
                if st>0:              vk[b,t,:st]      = past[b,t,:st]
                if st2>st:            vk[b,t,st:st2]   = fut[b,t,st:st2]
        if mask is not None: vk = vk * mask.unsqueeze(-1)
        return vk
```

> Optimize the double loop later with a scatter/gather; correctness first.

---

## 4. Motion-only score head — `kip/motion_head.py`

Reads the pseudo-flow alone and produces a per-frame anomaly curve used only by `L_kin`.

**Module:** `MotionScoreHead(nn.Module)`
**Input:** `ê_O ∈ ℝ^{L×256}`.
**Output:** `ŷ_O ∈ ℝ^{L}` in [0,1].

| Layer   | Config  | Input       | Output      |
| ------- | ------- | ----------- | ----------- |
| Linear  | 256→128 | `(B,L,256)` | `(B,L,128)` |
| GELU    | —       | —           | —           |
| Linear  | 128→1   | `(B,L,128)` | `(B,L,1)`   |
| Sigmoid | —       | —           | `(B,L)`     |

```python
class MotionScoreHead(nn.Module):
    def __init__(self, d_flow=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_flow,128), nn.GELU(), nn.Linear(128,1))
    def forward(self, eo):                 # (B,L,256)
        return torch.sigmoid(self.net(eo)).squeeze(-1)   # (B,L)
```

---

## 5. KIP module + losses — `kip/kip_module.py`, `kip/losses.py`

### 5.1 `KIP(nn.Module)` — wires §2–4 together

**Input:** `v^t ∈ ℝ^{L×512}`, optional `mask`.
**Outputs:** `v^k ∈ ℝ^{L×512}`, `ê_O ∈ ℝ^{L×256}`, `ŷ_O ∈ ℝ^{L}`.

```python
class KIP(nn.Module):
    def __init__(self, d=512, d_flow=256, K=4):
        super().__init__()
        self.pmg   = PMGFlowHead(d, 128, d_flow)
        self.shift = KinematicShift(d, K)
        self.mhead = MotionScoreHead(d_flow)
    def forward(self, vt, mask=None):
        eo  = self.pmg(vt)                 # (B,L,256)
        vk  = self.shift(vt, eo, mask)     # (B,L,512)
        yo  = self.mhead(eo)               # (B,L)
        return vk, eo, yo
```

### 5.2 Losses — `kip/losses.py`

**(a) `L_KIP_rec`** — MSE between pseudo-flow and cached RAFT target `e_O`.
Input: `ê_O (B,L,256)`, `e_O (B,L,256)`, `mask (B,L)`. Output: scalar.

```
L_KIP_rec = mean over valid (b,l) of  mean_k ( ê_O[b,l,k] − e_O[b,l,k] )²
```

**(b) `L_KIP_align`** — bidirectional snippet-level InfoNCE between `ê_O` and `v^t`.
Inputs: `ê_O (B,L,256)`, `v^t (B,L,512)`. First project both to a common dim `d_c=128` with two Linear layers (add them to `KIP.__init__`: `proj_flow: 256→128`, `proj_rgb: 512→128`). Then, **per video** (treat the `L` frames as the contrastive set), temperature `τ=0.07`:

```
a = normalize(proj_flow(ê_O))     # (L,128)
b = normalize(proj_rgb (v^t))     # (L,128)
S = a @ b.T / τ                   # (L,L)   S[i,j] = sim(flow_i, rgb_j)
targets = arange(L)               # positive = same index
L_f2r = CrossEntropy(S,   targets)      # rows: flow → rgb
L_r2f = CrossEntropy(S.T, targets)      # cols: rgb  → flow
L_KIP_align = 0.5 * (L_f2r + L_r2f)
```

Mask out padded indices (drop those rows/cols before the softmax).

**(c) `L_kin`** — kinematic-consistency MIL.
Inputs: `ŷ_O (B,L)`, main detection scores `y^bin (B,L)` (detach it), video-level labels `ŷ ∈ {0,1}^B`, top-k `k` (use LaGoVAD's k), `β=0.5`.

```
# term 1: standard top-k MIL on the motion curve
for each video: topk_mean_O = mean(topk(ŷ_O, k))
L_mil_motion = BCE( topk_mean_O , ŷ )          # abnormal→1, normal→0

# term 2: consistency on ABNORMAL videos only
for each abnormal video:
    idx_main = topk_indices(y^bin.detach(), k)  # where the main detector fires
    idx_mot  = topk_indices(ŷ_O,          k)    # where motion fires
    # compare the score profiles on the union of indices
    U = union(idx_main, idx_mot)
    L_cons += smooth_l1( ŷ_O[U] , y^bin.detach()[U] )
L_cons = L_cons / (#abnormal videos)

L_kin = L_mil_motion + β * L_cons
```

---

## 6. Integration into LaGoVAD forward pass

The only change to LaGoVAD's model is **one splice**: insert KIP between the temporal encoder and the fusion module, and route `v^k` (not `v^t`) to both the fusion module and `H_bin`'s pre-fusion (language-agnostic) pathway.

**Patched forward (pseudocode; edit LaGoVAD's model `forward`):**

```
v^t = temporal_encoder(clip_feats)               # (B,L,512)   UNCHANGED
v^k, ê_O, ŷ_O = kip(v^t, mask)                    # NEW
z^t = text_encoder(Z)                             # (B,C,512)   UNCHANGED
v^u, z^u = fusion_U(v^k, z^t)                     # was fusion_U(v^t, ...)  ← use v^k
y^bin = H_bin(pre=v^k, post=v^u)                  # pre-fusion path uses v^k (was v^t)
y^mul = H_mul(v^u, z^u)                           # UNCHANGED
return y^bin, y^mul, ê_O, ŷ_O                     # extra outputs for KIP losses
```

**`L_neg` compatibility:** LaGoVAD builds `ṽ_pos/ṽ_neg` from "temporal features". Feed it `v^k` (the new temporal features). No formula change.

### 6.1 Dynamic Video Synthesis patch (`synthesis.py`) — training only

Keep LaGoVAD's DVS as-is, with one addition for ego-centric datasets:

- **Motion-aware KNN key.** LaGoVAD's KNN filler matches on CLIP central-frame features. For ego-centric datasets (DoTA, DADA), **append a coarse ego-motion descriptor** to the KNN key: from the cached `e_O` of the candidate clip, compute mean magnitude and mean direction (2–4 numbers), L2-normalize, concat to the CLIP key before the cosine-similarity nearest-neighbor lookup. For fixed-camera datasets, leave the key unchanged.
- **Camera-geometry-conditioned `θ`.** Add a per-dataset flag `is_egocentric`. If true, raise the no-synthesis probability (use `θ_ego = 0.85` instead of `0.7`) and cap `δ_m = 2` (fewer splices). If false, keep `θ = 0.7`, `δ_m = 5`.
- **Pseudo-label passthrough to KIP:** DVS already returns `y^p ∈ {0,1}^L`. Expose it to the loss module so `L_kin`'s consistency term can, on synthesized abnormal videos, use `y^p` in place of `y^bin.detach()` as the anchor (known-window supervision). Flag: `use_yp_anchor=True` for synthesized samples.

---

## 7. Dataset preprocessing

For **every** dataset the pipeline is: (1) sample frames, (2) extract frozen CLIP features → `.npy` per video, (3) if in the training split, extract & cache RAFT flow embeddings `e_O` (§1), (4) build the label file. Sampling rule everywhere: **every 8 frames**, except **DoTA/DADA at their provided fps** (DoTA 10 fps). CLIP = ViT-B/16, 512-d, no 5/10-crop, no score smoothing.

Common feature-extraction step (all datasets):

```
frames = sample(video, stride=8 or dataset_fps)      # (L,3,H,W)
resize/centercrop → 224×224
F = CLIP_image_encoder(frames)                        # (L,512)   save cache/clip/{id}.npy
```

Per-dataset specifics:

### 7.1 PreVAD (primary WS training)

- **Source:** LaGoVAD's own dataset; reuse its download script + splits.
- **Labels:** video-level (abnormal/normal) for training; frame-level on the val split only.
- **Definitions `Z`:** use PreVAD's per-video anomaly descriptions + the 7 first-level / 35 sub-category names. During training randomly pick class-names or descriptions per batch (LaGoVAD convention).
- **Flow:** extract `e_O` for all training videos. `is_egocentric=False` for traffic-cam feeds; `True` for the dashcam/vlog subset if separable (else False).
- **Output files:** `clip/{id}.npy`, `flow/{id}.npy`, `labels_train.json` (id→0/1), `defs.json` (id→list of description strings).

### 7.2 MSAD (WS training + eval, fixed-camera)

- **Source:** official MSAD; request access; download the (blurred) videos + provided I3D/Swin features if you prefer, but for our pipeline extract **CLIP** features yourself for consistency.
- **Protocol ii (weakly-supervised):** train = 360 normal + 120 abnormal (video-level labels); test = 120 normal + 120 abnormal (frame-level labels provided).
- **Traffic slice:** filter by scenario ∈ {highway, road, street highview, parking lot, pedestrian street, sidewalk} for the traffic-focused experiments; keep the full set for the general benchmark.
- **`Z`:** MSAD 11 main anomaly-type names (+ optionally the sub-types in Appendix B). `is_egocentric=False`.
- **Frame-level test annotations:** convert MSAD's per-frame anomaly labels to `{id: [0/1 per sampled frame]}` aligned to the stride-8 sampling.
- **Output:** `clip/`, `flow/` (train only), `labels_train.json`, `frame_labels_test.json`, `defs.json`.

### 7.3 UCF-Crime (WS training bridge)

- **Source:** official UCF-Crime; use standard train/test split (Sultani et al.).
- **Labels:** video-level for training; frame-level (test) from the provided temporal annotations.
- **`Z`:** 13 class names; for the traffic-only sub-experiment restrict to `RoadAccidents` vs normal.
- **Flow:** `e_O` for training videos. `is_egocentric=False`.
- **Output:** same four file types.

### 7.4 DoTA (ego-centric eval; also the headline KIP metric)

- **Source:** official DoTA. Sample at provided **10 fps**.
- **Labels:** DoTA provides temporal anomaly windows → build frame-level test labels `{id:[0/1]}`. If used for any training, derive video-level labels (1 if any anomalous frame).
- **`Z`:** traffic definitions — e.g. `["vehicle collision", "vehicle-pedestrian collision", "out-of-control vehicle", "vehicle driving against traffic"]`; DoTA's ego-involved/non-ego categories can map into these.
- **`is_egocentric=True`** (affects DVS only, and only if DoTA is in a training split; for pure eval no DVS).
- **Output:** `clip/`, `frame_labels.json`, `defs.json`. Flow not needed for pure eval.

### 7.5 DADA-2000 (ego-centric cross-dataset eval)

- **Source:** official DADA-2000. Sample at provided fps (treat like DoTA).
- **Labels:** attention/anomaly temporal annotations → frame-level test labels.
- **`Z`:** same traffic definition set as DoTA.
- Eval only (zero-shot). `clip/`, `frame_labels.json`, `defs.json`.

### 7.6 TAD (fixed-camera traffic eval)

- **Source:** official TAD benchmark. Sample every 8 frames.
- **Labels:** frame-level test annotations.
- **`Z`:** traffic definition set. Eval only.

### 7.7 (optional) A3D / DAD / CCD — auxiliary dashcam accident sets

- Dashcam accident clips with clip-level positive/negative structure → derive video-level labels for extra WS training augmentation, or frame-level (where available) for eval. Same feature+flow pipeline; `is_egocentric=True`.

### 7.8 (optional) BDD100K normal driving — Stage-0 warm-up only

- Normal-only driving clips. Extract CLIP features + `e_O`. No anomaly labels needed.

**Label file formats (standardize across all datasets):**

```
labels_train.json      : { video_id: 0|1 }                       # video-level
frame_labels_test.json : { video_id: [0,0,1,1,...] }             # per sampled frame
defs.json              : { video_id: ["desc1", ...] }  or  global list of class names
```

---

## 8. Training pipeline — `train.py`

Frozen throughout: CLIP image + text encoders. Trainable: temporal encoder, KIP, fusion `U`, `H_bin`, `H_mul`, projections. RAFT is **never** in the training graph — `e_O` is loaded from cache.

Optimizer/schedule (LaGoVAD): AdamW, LR 5e-5, batch 64, ~40 epochs, single GPU. Sample every 8 frames.

**Stage 0 (optional) — masked warm-up on normal driving video (BDD100K / PreVAD-normal).**

- For each clip: mask 75% of the `v^t` sequence positions; train temporal encoder + KIP.pmg to reconstruct (a) the masked `v^t` tokens and (b) their cached `e_O`.
- Loss: `MSE(recon_vt, vt_masked) + MSE(pmg(vt), e_O)` on masked positions.
- Keep temporal encoder + PMG weights; discard the reconstruction decoder.

**Stage 0.5 (optional; skip if using stock-CLIP default) — Traffic Hard-Negative Tuning.**

- Build hard-negative pairs: (i) same-scene — an accident clip's own pre-crash normal portion; (ii) semantic — curated text pairs ("car brakes hard and stops safely" vs "car collides").
- Fine-tune only a thin projection on top of frozen CLIP with LaGoVAD's `L_neg` contrastive objective. CLIP body stays frozen. Save the projection; load it in later stages.

**Stage 1 — KIP warm-up.** Freeze CLIP; train **KIP only**:

```
L_stage1 = L_KIP_rec + L_KIP_align
```

Run a few epochs (e.g. 5) until `L_KIP_rec` plateaus. Purpose: make `ê_O` faithful + RGB-aligned before task training.

**Stage 2 — Task training (full objective).** Train temporal encoder + KIP + fusion + heads:

```
L_total =  L_MIL + L_MIL_align + L_dvs + L_neg           # LaGoVAD, unchanged (reuse)
         + λ_rec·L_KIP_rec + λ_al·L_KIP_align + γ·L_kin  # KIP (ours)
```

Weights: `λ_rec = 1.0` (leave unweighted, keep pseudo-flow bounded to targets), `λ_al = 0.1`, `γ = 0.2`, `β (inside L_kin) = 0.5`. Tune on the PreVAD val split (frame labels available).

- DVS runs on-the-fly inside the dataloader (LaGoVAD), producing `v` + `y^p`; apply §6.1 patch.
- `y^bin.detach()` is used inside `L_kin` (stop-gradient), except on synthesized abnormal videos where `y^p` is the anchor (`use_yp_anchor=True`).

**Stage 3 (optional) — reasoning head.** Freeze the whole detector. Train an ATS sampler + LoRA adapter (r=64, α=128) on instruction data (HIVAU-style + CUVA traffic captions). The frozen `y^bin` is the sampler signal. This does not touch the detector weights and can be built last.

**Per-step training loop (Stage 2):**

```
for batch in loader:                       # batch = clip_feats, e_O, ŷ (video label), Z, y^p, mask
    v^t          = temporal_encoder(clip_feats, mask)
    v^k, ê_O, ŷ_O = kip(v^t, mask)
    z^t          = text_encoder(Z)
    v^u, z^u     = fusion_U(v^k, z^t)
    y^bin        = H_bin(v^k, v^u)
    y^mul        = H_mul(v^u, z^u)
    loss  = L_MIL(y^bin, ŷ) + L_MIL_align(y^mul, ...) + L_dvs(y^bin, y^p) + L_neg(v^k, z^t, y^bin)
    loss += 1.0*L_KIP_rec(ê_O, e_O, mask) + 0.1*L_KIP_align(ê_O, v^t, mask) \
          + 0.2*L_kin(ŷ_O, y^bin.detach(), ŷ, y^p, k, β=0.5)
    loss.backward(); opt.step(); opt.zero_grad()
```

---

## 9. Inference pipeline — `infer.py`

RGB + text only. **RAFT and the LLM are not on the scoring path.** No DVS.

**Per test video, given definition `Z`:**

```
1. frames   = sample(video, stride=8 | dataset_fps); resize 224
2. F        = CLIP_image_encoder(frames)              # (L,512)
3. v^t      = temporal_encoder(F)                     # (L,512)
4. v^k, ê_O, _ = kip(v^t)      # PMG regenerates ê_O from v^t ALONE; RAFT NOT loaded
5. z^t      = CLIP_text_encoder(Z)                    # (C,512)   (cache per Z)
6. v^u, z^u = fusion_U(v^k, z^t)
7. y^bin    = H_bin(v^k, v^u)                         # (L,)  ← anomaly curve (main output)
   y^mul    = H_mul(v^u, z^u)                         # (L,C) ← per-frame category (triage)
8. localize : smooth y^bin (small Gaussian) + threshold → incident windows
9. (optional) ATS sampler over y^bin → MLLM → incident report   # asynchronous, off critical path
```

`ŷ_O` from step 4 is discarded (or kept as a diagnostic curve). Changing `Z` = one text-encoder forward + cached; no retraining.

---

## 10. Evaluation

Compute on frame-level scores `y^bin` vs `frame_labels_test.json`:

- **Frame-level AUC** (all datasets) — primary. Report DoTA/DADA/TAD/MSAD/UCF separately.
- **AP** (PreVAD-val, MSAD).
- **AUC_A / AnoAUC** — AUC restricted to abnormal videos.
- **MCC, AUCMCC, MCC@0.5** — for DoTA/DADA (imbalance).
- **mAP@IoU {0.1..0.5} + AVG** — temporal localization (MSAD, UCF).
- **Cross-dataset** — train PreVAD/MSAD/UCF, eval zero-shot on DoTA/DADA/TAD.
- Report **GFLOPs / FPS** of the inference path; reasoning latency separately.

**Required ablations (toggle flags in `config.py`):**

1. LaGoVAD baseline (KIP off).
2. - PMG head only (no gate-shift, no `L_kin`) — `v^k = v^t`, only `L_KIP_rec + L_KIP_align` active.
3. - gate-shift (KIP full, `L_kin` off).
4. - `L_kin` (full KIP).
5. gate signal: flow-norm (ours) vs feature-variance (RefineVAD original) — swap step-1 of §3.
6. KIP without temporal encoder (splice KIP directly on `F`) — expect `L_KIP_rec` to stay high.
7. - Stage-0 warm-up, + Stage-0.5 tuning (each on/off).

---

## 11. Config flags (`config.py`)

```
backbone         = "clip_vitb16"        # | "clip_vitb16+hn" (Stage0.5) | "openclip_vitl" | "alertclip" (if weights)
K                = 4                     # folding factor
d_flow           = 256
lambda_rec       = 1.0
lambda_align     = 0.1
gamma_kin        = 0.2
beta_cons        = 0.5
tau_align        = 0.07
topk             = <LaGoVAD default>
dvs_theta        = 0.7                   # 0.85 if is_egocentric
dvs_delta_m      = 5                     # 2  if is_egocentric
use_yp_anchor    = True
stage0_warmup    = False
stage05_hntune   = False
```

---

## 12. Build order (suggested for Claude Code)

1. `flow/raft_extract.py` → cache `e_O` on a small PreVAD subset (verify shapes `(L,256)`).
2. `kip/pmg.py`, `kip/gate_shift.py`, `kip/motion_head.py`, `kip/kip_module.py` → unit-test shapes on random `v^t (2,50,512)`.
3. `kip/losses.py` → unit-test each loss returns a finite scalar.
4. Patch LaGoVAD model `forward` (§6) + dataloader to yield `e_O`, `y^p`, `mask`.
5. Stage 1 warm-up → confirm `L_KIP_rec` decreases.
6. Stage 2 full training on PreVAD → sanity AUC on PreVAD-val.
7. Add dataset preprocessors (§7) one at a time; run eval (§10).
8. (optional) Stage 0, Stage 0.5, Stage 3.
