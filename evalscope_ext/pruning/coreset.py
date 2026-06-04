"""The coreset selection, kept as plain numpy so it's easy to test.

Bin items by difficulty (and, if there's budget for it, a content axis), keep
items from each bin in proportion to the bin size, and inside a bin spread the
picks out across the content axis instead of clustering. Proportional sampling
keeps the mean estimate roughly unbiased; the difficulty bins are what cut the
variance. Difficulty comes from an offline prior, never the model under test.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _rank_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin by rank (handles ties / skew better than fixed-width on raw features)."""
    x = np.asarray(x, dtype=float)
    if n_bins <= 1 or np.allclose(x, x[0]):
        return np.zeros(len(x), dtype=int)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    return np.minimum((ranks / len(x) * n_bins).astype(int), n_bins - 1)


def _difficulty_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Difficulty is already 0..1, so fixed-width bins make sense; the number of
    occupied bins then adapts to how finely the prior is resolved."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if n_bins <= 1:
        return np.zeros(len(x), dtype=int)
    return np.minimum((x * n_bins).astype(int), n_bins - 1)


def _allocate(counts: np.ndarray, budget: int, floor: int) -> np.ndarray:
    """Hand out `budget` slots across bins ~proportional to size, min `floor` each."""
    alloc = np.zeros(len(counts), dtype=int)
    alloc[counts > 0] = floor
    left = budget - int(alloc.sum())
    if left <= 0:
        alloc[:] = 0
        for i in np.argsort(-counts)[:max(budget, 0)]:
            alloc[i] = 1
        return alloc
    quota = counts / counts.sum() * left
    base = np.floor(quota).astype(int)
    alloc += base
    for i in np.argsort(-(quota - base))[: left - int(base.sum())]:
        alloc[i] += 1
    return alloc


def _spread(sorted_idx: List[int], k: int) -> List[int]:
    """Pick k items evenly spaced along an already-sorted list."""
    n = len(sorted_idx)
    k = min(k, n)
    if k <= 0:
        return []
    if k == n:
        return list(sorted_idx)
    pos = np.linspace(0, n - 1, k).round().astype(int)
    return [sorted_idx[p] for p in pos]


def select_coreset(
    features: np.ndarray,
    difficulty: np.ndarray,
    ratio: float,
    *,
    n_difficulty_bins: int = 4,
    n_content_bins: int = 3,
    floor: int = 1,
) -> Tuple[List[int], Dict[int, str]]:
    """Pick a subset of items. Returns (kept indices, bin label per kept item).

    features    (n, d) cheap per-item features; column 0 is used as the content axis
    difficulty  (n,) 0..1, higher = harder
    ratio       fraction to keep
    """
    features = np.asarray(features, dtype=float)
    difficulty = np.asarray(difficulty, dtype=float)
    n = len(features)
    budget = max(1, round(ratio * n))

    content = features[:, 0] if features.shape[1] else np.zeros(n)
    diff_bin = _difficulty_bins(difficulty, n_difficulty_bins)

    # only add the (weaker) content split when each difficulty bin can spare a few
    # picks for it, otherwise it just fragments a small budget
    n_diff = len(set(diff_bin.tolist()))
    if budget / max(n_diff, 1) < 4:
        n_content_bins = 1
    content_bin = _rank_bins(content, n_content_bins)

    labels = np.array([f'd{d}c{c}' for d, c in zip(diff_bin, content_bin)])
    uniq = sorted(set(labels))
    counts = np.array([(labels == u).sum() for u in uniq], dtype=float)
    alloc = np.minimum(_allocate(counts, budget, floor), counts.astype(int))

    keep, bin_of = [], {}
    for u, k in zip(uniq, alloc):
        if k <= 0:
            continue
        members = np.where(labels == u)[0]
        ordered = [int(i) for i in members[np.argsort(content[members], kind='stable')]]
        for g in _spread(ordered, int(k)):
            keep.append(g)
            bin_of[g] = u
    keep.sort()
    return keep, bin_of
