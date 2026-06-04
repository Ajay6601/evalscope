"""The actual coreset selection, kept as plain numpy so it's easy to test.

Plan: bin items by difficulty (and one content axis), keep items from each bin
in proportion to the bin size, and inside a bin pick the items that best separate
models while still spreading across the feature space. Proportional sampling
keeps the mean estimate roughly unbiased; the spread + discrimination part is
what makes the subset useful for a go/no-go call. None of this looks at the model
being tested - difficulty/discrimination come from a prior built offline.
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
    """Difficulty is already on a 0..1 scale, so fixed-width bins make sense here.
    The number of *occupied* bins then adapts to how finely the prior is resolved.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if n_bins <= 1:
        return np.zeros(len(x), dtype=int)
    return np.minimum((x * n_bins).astype(int), n_bins - 1)


def _allocate(counts: np.ndarray, budget: int, floor: int) -> np.ndarray:
    """Hand out `budget` slots across bins, ~proportional to bin size, with a
    minimum of `floor` per non-empty bin (largest-remainder rounding)."""
    alloc = np.zeros(len(counts), dtype=int)
    nonempty = counts > 0
    alloc[nonempty] = floor
    left = budget - int(alloc.sum())
    if left <= 0:
        # not enough budget even for the floor -> just take the biggest bins
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


def _pick_within_bin(feats, disc, k, alpha, rng):
    """Greedy farthest-point pick, seeded by the most discriminating item.
    Score each candidate as alpha*discrimination + (1-alpha)*distance-to-picked."""
    n = len(feats)
    k = min(k, n)
    if k <= 0:
        return []
    if k == n:
        return list(range(n))
    picked = [int(np.argmax(disc + rng.uniform(0, 1e-9, n)))]
    d2 = ((feats - feats[picked[0]]) ** 2).sum(1)
    while len(picked) < k:
        spread = d2 / (d2.max() + 1e-12)
        score = alpha * disc + (1 - alpha) * spread
        score[picked] = -np.inf
        nxt = int(np.argmax(score + rng.uniform(0, 1e-9, n)))
        picked.append(nxt)
        d2 = np.minimum(d2, ((feats - feats[nxt]) ** 2).sum(1))
    return picked


def select_coreset(
    features: np.ndarray,
    difficulty: np.ndarray,
    discrimination: np.ndarray,
    ratio: float,
    *,
    n_difficulty_bins: int = 4,
    n_content_bins: int = 3,
    alpha: float = 0.5,
    floor: int = 1,
    seed: int = 0,
) -> Tuple[List[int], Dict[int, str], Dict[str, float]]:
    """Pick a subset of items. Returns (kept indices, bin per item, bin weights).

    features        (n, d) cheap per-item features (standardised here)
    difficulty      (n,) 0..1, higher = harder
    discrimination  (n,) 0..1, higher = separates models more
    ratio           fraction to keep
    """
    features = np.asarray(features, dtype=float)
    difficulty = np.asarray(difficulty, dtype=float)
    discrimination = np.asarray(discrimination, dtype=float)
    n = len(features)
    budget = max(1, round(ratio * n))
    rng = np.random.default_rng(seed)

    # standardise features, then collapse to one "content" number via PCA so the
    # content bins don't depend on how many features we happen to have
    mu, sd = features.mean(0), features.std(0)
    sd[sd == 0] = 1.0
    z = (features - mu) / sd
    if z.shape[1] == 0:
        content = np.zeros(n)
    else:
        _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
        pc1 = vt[0]
        if pc1[np.argmax(np.abs(pc1))] < 0:
            pc1 = -pc1
        content = z @ pc1

    diff_bin = _difficulty_bins(difficulty, n_difficulty_bins)

    # only split on the (weaker) content axis if each difficulty bin can spare a
    # few picks for it - otherwise we just fragment a small budget
    n_diff = len(set(diff_bin.tolist()))
    if budget / max(n_diff, 1) < 4:
        n_content_bins = 1
    content_bin = _rank_bins(content, n_content_bins)

    labels = np.array([f'd{d}c{c}' for d, c in zip(diff_bin, content_bin)])
    uniq = sorted(set(labels))
    counts = np.array([(labels == u).sum() for u in uniq], dtype=float)
    weights = counts / counts.sum()
    alloc = np.minimum(_allocate(counts, budget, floor), counts.astype(int))

    keep, bin_of = [], {}
    for u, k in zip(uniq, alloc):
        if k <= 0:
            continue
        local = np.where(labels == u)[0]
        for li in _pick_within_bin(z[local], discrimination[local], int(k), alpha, rng):
            g = int(local[li])
            keep.append(g)
            bin_of[g] = u
    keep.sort()
    return keep, bin_of, {u: float(w) for u, w in zip(uniq, weights)}
