"""Exploratory data analysis for the KAT-VAD corpora.

Every number this package computes is a *dataset* property, measured from the
files on disk — never a model result. It exists because
``core/docs/v3/RESULTS_DADA.md`` established that several headline metrics were
artifacts of corpus geometry (clip length against the score head's receptive
field, and an all-normal-clip-dominated test split) rather than of the model,
and those properties were only discovered after a full training campaign.

Layout mirrors the questions:

* :mod:`core.eda.corpus` — how many clips, how long, how they subgroup, and
  what the score head and the MIL loss can physically see of them.
* :mod:`core.eda.labels` — where the positives are: spans, counts, and the
  windows that round away at a given stride.
* :mod:`core.eda.protocol` — what a micro AUC on this label distribution is
  actually measuring, including the constant-score-per-clip oracle (lesson C12).
* :mod:`core.eda.features` — what the cached frozen-CLIP features contain,
  including the frame-level linear probe of ``RESULTS_DADA.md`` §10-B.
* :mod:`core.eda.report` — assembly, Markdown rendering, cross-corpus diff.

Import the submodules directly (``from core.eda import corpus``); this package
deliberately re-exports nothing, so a section can be loaded without pulling in
matplotlib or scikit-learn.
"""

from __future__ import annotations
