"""Stratified discriminative coreset selection (numpy-only core).

This module contains the algorithm with no evalscope or benchmark coupling so it
can be unit-tested and reused by the offline validation harness.  See
``stratified_coreset.py`` for the evalscope-facing strategy that wires features
and a difficulty prior into :func:`select_coreset`.

Idea
----
We want a small sample set whose **mean accuracy estimates the full-benchmark
mean** for a *new* model (low bias + low variance), while still containing the
items that **separate a good model from a bad one** near a decision threshold.

Two well-grounded ingredients:

1. *Stratify on item-side properties* — a smoothed difficulty prior (latent item
   difficulty, IRT-style) crossed with an intrinsic content axis — and allocate
   the budget **proportionally** to stratum mass.  A proportional stratified
   estimator is unbiased for the full mean and has variance no larger than
   uniform random sampling (strictly lower when accuracy varies between strata),
   and the strata are defined without ever looking at the candidate model.

2. *Within each stratum, pick by discrimination + coverage*, not by score rank.
   We greedily grow the kept set using ``alpha * discrimination + (1-alpha) *
   distance-to-already-picked``.  Discrimination (variance of correctness across
   the calibration pool) surfaces items that tell strong and weak models apart;
   the coverage term spreads picks across the feature space so the subset still
   represents items a fourth model might find hard for idiosyncratic reasons.

Because difficulty/discrimination are treated as *item properties* (estimated as
a shrunk prior over a model pool) and only used to lay out strata and rank
within them, the method does not memorise "which items model X failed" and so is
defensible for an unseen model.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a quantile bin in ``[0, n_bins)`` (stable, dedup-safe)."""
    x = np.asarray(x, dtype=float)
    if n_bins <= 1 or np.allclose(x, x[0]):
        return np.zeros(len(x), dtype=int)
    # rank-based binning is robust to ties and skew (unlike fixed-width bins).
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x))
    return np.minimum((ranks / len(x) * n_bins).astype(int), n_bins - 1)


def _fixed_width_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin values in ``[0,1]`` into fixed-width buckets.

    Difficulty is a *calibrated* quantity on a fixed scale, so equal-width bins
    are the right tool: occupied-bin count then adapts to however many distinct
    difficulty levels the calibration pool resolves (e.g. a 2-model pool yields a
    coarse prior, a 10-model pool a fine one) without degenerating the way
    quantile bins do on a low-cardinality prior.
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if n_bins <= 1:
        return np.zeros(len(x), dtype=int)
    return np.minimum((x * n_bins).astype(int), n_bins - 1)


def _largest_remainder_alloc(weights: np.ndarray, budget: int, floor: int) -> np.ndarray:
    """Allocate ``budget`` integer slots across strata ~proportional to ``weights``.

    Every non-empty stratum first receives ``floor`` slots (coverage guarantee),
    then the remainder is distributed by the largest-remainder method.  Allocation
    is clipped to stratum capacity (handled by the caller).
    """
    n_strata = len(weights)
    alloc = np.zeros(n_strata, dtype=int)
    nonempty = weights > 0
    alloc[nonempty] = floor
    used = int(alloc.sum())
    remaining = budget - used
    if remaining <= 0:
        # Budget smaller than #strata*floor: keep the heaviest strata only.
        keep = np.argsort(-weights)[:max(budget, 0)]
        alloc[:] = 0
        alloc[keep] = 1
        return alloc
    quota = weights / weights.sum() * remaining
    base = np.floor(quota).astype(int)
    alloc += base
    frac = quota - base
    leftover = remaining - int(base.sum())
    for i in np.argsort(-frac)[:leftover]:
        alloc[i] += 1
    return alloc


def _greedy_cover(
    feats: np.ndarray,
    utility: np.ndarray,
    k: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[int]:
    """Pick ``k`` rows of ``feats`` maximising ``alpha*utility + (1-alpha)*spread``.

    A farthest-point/k-center style greedy seeded by the highest-utility item.
    ``feats`` are standardised; ``utility`` is a 0..1 discrimination score.
    Returns indices into the local arrays.
    """
    n = len(feats)
    k = min(k, n)
    if k <= 0:
        return []
    if k == n:
        return list(range(n))

    # Seed with the most discriminative item (deterministic; rng only tie-breaks).
    util = utility + rng.uniform(0, 1e-9, size=n)
    first = int(np.argmax(util))
    selected = [first]
    # min squared distance from every point to the selected set
    d2 = np.sum((feats - feats[first]) ** 2, axis=1)
    while len(selected) < k:
        # normalise spread to 0..1 so it is comparable to utility
        spread = d2 / (d2.max() + 1e-12)
        gain = alpha * utility + (1 - alpha) * spread
        gain[selected] = -np.inf
        nxt = int(np.argmax(gain + rng.uniform(0, 1e-9, size=n)))
        selected.append(nxt)
        d2 = np.minimum(d2, np.sum((feats - feats[nxt]) ** 2, axis=1))
    return selected


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
    """Select a stratified discriminative coreset.

    Parameters
    ----------
    features:
        ``(n, d)`` intrinsic, model-independent item features (already cleaned;
        standardisation is done here).
    difficulty:
        ``(n,)`` difficulty prior in ``[0, 1]`` (higher = harder), from calibration.
    discrimination:
        ``(n,)`` discrimination score in ``[0, 1]`` (higher = separates models),
        from calibration.
    ratio:
        Fraction of items to keep.

    Returns
    -------
    (keep_indices, stratum_of, stratum_weight)
    """
    features = np.asarray(features, dtype=float)
    n = len(features)
    difficulty = np.asarray(difficulty, dtype=float)
    discrimination = np.asarray(discrimination, dtype=float)
    budget = max(1, int(round(ratio * n)))
    rng = np.random.default_rng(seed)

    # Standardise features; collapse to a 1-D content coordinate via PCA so the
    # content strata are stable regardless of feature count.
    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    sd[sd == 0] = 1.0
    z = (features - mu) / sd
    if z.shape[1] == 0:
        content_coord = np.zeros(n)
    else:
        # first principal component (sign-stabilised)
        _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
        pc1 = vt[0]
        if pc1[np.argmax(np.abs(pc1))] < 0:
            pc1 = -pc1
        content_coord = z @ pc1

    diff_bin = _fixed_width_bins(difficulty, n_difficulty_bins)

    # Budget guard: only stratify on the (secondary, possibly weak) content axis
    # when each occupied difficulty bin can afford >=4 picks. Otherwise the
    # content split fragments a small budget and adds variance instead of removing
    # it. This keeps difficulty -- the axis proven to transfer across models --
    # as the workhorse at tight prune ratios.
    n_diff_occupied = len(set(diff_bin.tolist()))
    if budget / max(n_diff_occupied, 1) < 4:
        n_content_bins = 1
    content_bin = _quantile_bins(content_coord, n_content_bins)
    labels = np.array([f'd{d}_c{c}' for d, c in zip(diff_bin, content_bin)])

    uniq = sorted(set(labels))
    counts = np.array([(labels == u).sum() for u in uniq], dtype=float)
    weights = counts / counts.sum()
    alloc = _largest_remainder_alloc(counts, budget, floor)
    alloc = np.minimum(alloc, counts.astype(int))  # cannot exceed stratum size

    keep: List[int] = []
    stratum_of: Dict[int, str] = {}
    for u, k in zip(uniq, alloc):
        if k <= 0:
            continue
        local = np.where(labels == u)[0]
        chosen_local = _greedy_cover(z[local], discrimination[local], int(k), alpha, rng)
        for li in chosen_local:
            g = int(local[li])
            keep.append(g)
            stratum_of[g] = u
    keep.sort()
    stratum_weight = {u: float(w) for u, w in zip(uniq, weights)}
    return keep, stratum_of, stratum_weight
