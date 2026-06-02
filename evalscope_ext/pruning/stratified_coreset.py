"""``stratified_coreset`` -- the Part A pruning strategy.

The algorithm lives in :mod:`coreset`; this class adapts it to evalscope's
``Sample`` objects.  Benchmark-specific glue (how to turn a sample into intrinsic
features, and how to join a sample to its calibration row) is injected by the
pruned *adapter* through :meth:`configure`, so the same algorithm serves coding,
long-context, or any future benchmark without edits here.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from .base import PruneResult, PruningStrategy, register_pruner
from .calibration import CalibrationTable
from .coreset import select_coreset


@register_pruner('stratified_coreset')
class StratifiedCoresetPruner(PruningStrategy):
    """Stratified discriminative coreset (see :func:`coreset.select_coreset`).

    Tunables (via ``--dataset-args`` ``extra_params``):
        alpha:              discrimination vs coverage blend in [0,1] (default 0.5)
        n_difficulty_bins:  fixed-width difficulty strata (default 4)
        n_content_bins:     content sub-strata when budget allows (default 3)
        floor:              minimum picks per occupied stratum (default 1)
    """

    def __init__(self, prune_ratio: float = 0.1, seed: int = 0,
                 alpha: float = 0.5, n_difficulty_bins: int = 4,
                 n_content_bins: int = 3, floor: int = 1, **kwargs):
        super().__init__(prune_ratio=prune_ratio, seed=seed, **kwargs)
        self.alpha = float(alpha)
        self.n_difficulty_bins = int(n_difficulty_bins)
        self.n_content_bins = int(n_content_bins)
        self.floor = int(floor)
        # injected by the adapter
        self._feature_fn: Optional[Callable] = None
        self._key_fn: Optional[Callable] = None
        self._calibration: Optional[CalibrationTable] = None

    def configure(self, *, feature_fn: Callable, key_fn: Callable,
                  calibration: Optional[CalibrationTable]) -> 'StratifiedCoresetPruner':
        """Inject benchmark-specific feature/key extractors and the prior."""
        self._feature_fn = feature_fn
        self._key_fn = key_fn
        self._calibration = calibration
        return self

    def prune(self, samples: Sequence, subset: str) -> PruneResult:
        if self._feature_fn is None or self._key_fn is None:
            raise RuntimeError('StratifiedCoresetPruner.configure() must be called first')
        n = len(samples)
        if n == 0:
            return PruneResult(keep_indices=[])

        feats: List[List[float]] = []
        difficulty = np.empty(n)
        discrimination = np.empty(n)
        joined = 0
        for i, s in enumerate(samples):
            feats.append(list(self._feature_fn(s)))
            if self._calibration is not None:
                d, disc, n_obs = self._calibration.get(self._key_fn(s, i))
                joined += int(n_obs > 0)
            else:
                d, disc = 0.5, 0.5
            difficulty[i] = d
            discrimination[i] = disc

        features = np.array(feats, dtype=float)
        keep, stratum_of, stratum_weight = select_coreset(
            features, difficulty, discrimination, self.prune_ratio,
            n_difficulty_bins=self.n_difficulty_bins,
            n_content_bins=self.n_content_bins,
            alpha=self.alpha, floor=self.floor, seed=self.seed,
        )
        return PruneResult(
            keep_indices=keep,
            stratum_of={i: stratum_of[i] for i in keep},
            stratum_weight=stratum_weight,
            diagnostics={
                'n_full': n, 'n_kept': len(keep),
                'calibration_join_rate': round(joined / n, 3),
                'subset': subset,
            },
        )
