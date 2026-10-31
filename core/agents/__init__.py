"""KAT-VAD v6 (LaGoVAD-AP): detected and tracked road agents as MIL instances.

``layout``   -- the per-row column layout the agent cache appends to a model input row.
``geometry`` -- box / track / camera-affine arithmetic (numpy, no model, no detector).
``tracking`` -- the off-the-shelf detector + tracker wrapper (ultralytics, Colab GPU).
``features`` -- per-row agent blocks: selection, VideoMAE RoI motion, CLIP crops, kinematics.
"""
