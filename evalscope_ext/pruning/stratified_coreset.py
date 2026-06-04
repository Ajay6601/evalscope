"""stratified_coreset - the Part A strategy.

Works on any benchmark: it pulls a few cheap, generic features off each Sample
(prompt length, #choices, #images) and reads an optional difficulty prior. The
real selection logic lives in coreset.py.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from .base import PruneResult, PruningStrategy, register_pruner
from .calibration import CalibrationTable
from .coreset import select_coreset


def _sample_features(sample) -> List[float]:
    """Generic, model-free features that exist for basically any benchmark."""
    text_len, n_img = 0, 0
    msgs = sample.input if isinstance(sample.input, list) else [sample.input]
    for m in msgs:
        c = getattr(m, 'content', m)
        if isinstance(c, str):
            text_len += len(c)
        elif isinstance(c, list):
            for part in c:
                if getattr(part, 'type', None) == 'image':
                    n_img += 1
                else:
                    text_len += len(getattr(part, 'text', '') or '')
    # input_tokens shows up in some benchmarks' metadata (e.g. long-context); use it if there
    tok = (getattr(sample, 'metadata', {}) or {}).get('input_tokens')
    return [
        math.log1p(text_len),
        float(len(sample.choices or [])) if getattr(sample, 'choices', None) else 0.0,
        float(n_img),
        math.log1p(tok) if isinstance(tok, (int, float)) else 0.0,
    ]


@register_pruner('stratified_coreset')
class StratifiedCoresetPruner(PruningStrategy):
    """Knobs (via extra_params): n_difficulty_bins, n_content_bins, floor."""

    def __init__(self, prune_ratio=0.1, seed=0,
                 n_difficulty_bins=4, n_content_bins=3, floor=1, **kwargs):
        super().__init__(prune_ratio=prune_ratio, seed=seed, **kwargs)
        self.n_difficulty_bins = int(n_difficulty_bins)
        self.n_content_bins = int(n_content_bins)
        self.floor = int(floor)
        self.calibration: Optional[CalibrationTable] = None  # set by the adapter

    def prune(self, samples: Sequence, subset: str) -> PruneResult:
        n = len(samples)
        if n == 0:
            return PruneResult(keep_indices=[])

        feats = np.array([_sample_features(s) for s in samples], dtype=float)
        difficulty = np.full(n, 0.5)
        joined = 0
        if self.calibration is not None:
            for i in range(n):
                d, n_obs = self.calibration.get(str(i))  # positional join
                difficulty[i] = d
                joined += n_obs > 0

        keep, bin_of = select_coreset(
            feats, difficulty, self.prune_ratio,
            n_difficulty_bins=self.n_difficulty_bins,
            n_content_bins=self.n_content_bins, floor=self.floor,
        )
        return PruneResult(
            keep_indices=keep,
            bin_of={i: bin_of[i] for i in keep},
            info={'n_full': n, 'n_kept': len(keep), 'subset': subset,
                  'calibration_join_rate': round(joined / n, 3)},
        )
