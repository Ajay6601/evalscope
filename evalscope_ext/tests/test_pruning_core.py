"""Unit tests for the numpy-only coreset core (no evalscope needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalscope_ext.pruning.calibration import beta_binomial_mean, compute_item_stats
from evalscope_ext.pruning.coreset import select_coreset


def test_budget_and_weights():
    rng = np.random.default_rng(0)
    n = 200
    feats = rng.normal(size=(n, 3))
    diff = rng.uniform(size=n)
    disc = rng.uniform(size=n)
    for ratio in (0.05, 0.1, 0.25):
        keep, strat, w = select_coreset(feats, diff, disc, ratio, seed=0)
        assert len(keep) == round(ratio * n)
        assert len(set(keep)) == len(keep)              # no duplicates
        assert abs(sum(w.values()) - 1.0) < 1e-9        # stratum weights normalised
        assert keep == sorted(keep)                     # stable order
    print('budget/weights OK')


def test_determinism():
    rng = np.random.default_rng(1)
    feats, diff, disc = rng.normal(size=(120, 2)), rng.uniform(size=120), rng.uniform(size=120)
    a = select_coreset(feats, diff, disc, 0.2, seed=7)[0]
    b = select_coreset(feats, diff, disc, 0.2, seed=7)[0]
    assert a == b
    print('determinism OK')


def test_beats_random_when_difficulty_predicts_label():
    """In the regime the method targets -- a *discrete* difficulty prior estimated
    from a small model pool, where accuracy is ~homogeneous within a difficulty
    level -- proportional difficulty stratification gives an unbiased, lower-
    variance estimate of an UNSEEN model's accuracy than uniform random sampling.
    (This mirrors the real benchmarks: a 2-3 model pool yields a few-valued prior;
    see analysis/results for the on-data leave-one-model-out win.)"""
    rng = np.random.default_rng(2)
    n = 400
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    difficulty = rng.choice(levels, size=n)
    # an UNSEEN model: P(correct) set by difficulty level (homogeneous within level)
    p = 0.95 - 0.8 * difficulty
    y = (rng.uniform(size=n) < p).astype(float)
    truth = y.mean()
    feats = np.column_stack([difficulty, rng.normal(size=n)])
    disc = difficulty * (1 - difficulty) * 4  # peaked mid-difficulty

    core_err, rand_err = [], []
    for s in range(60):
        keep, _, _ = select_coreset(feats, difficulty, disc, 0.1,
                                    n_difficulty_bins=5, n_content_bins=1, seed=s)
        core_err.append(abs(y[keep].mean() - truth))
        idx = np.random.default_rng(1000 + s).choice(n, size=len(keep), replace=False)
        rand_err.append(abs(y[idx].mean() - truth))
    cm, rm = np.mean(core_err), np.mean(rand_err)
    assert cm < rm, (cm, rm)
    print(f'variance-reduction OK: coreset MAE={cm:.4f} < random MAE={rm:.4f}')


def test_judge_noise_increases_shrinkage():
    """Higher judge noise -> difficulty pulled harder toward the global mean."""
    clean = compute_item_stats({'a': 1, 'b': 1, 'c': 1}, global_pass=0.5, judge_noise=0.0)
    noisy = compute_item_stats({'a': 1, 'b': 1, 'c': 1}, global_pass=0.5, judge_noise=0.5)
    # all-correct item -> low difficulty; noise should make it *less* extreme
    assert noisy['difficulty'] > clean['difficulty']
    # discrimination of a unanimous item is ~0 and noise can only reduce it
    assert noisy['discrimination'] <= clean['discrimination'] + 1e-9
    assert beta_binomial_mean(3, 3, 0.5, 100) < 0.55  # strong prior dominates
    print('judge-noise shrinkage OK')


if __name__ == '__main__':
    test_budget_and_weights()
    test_determinism()
    test_beats_random_when_difficulty_predicts_label()
    test_judge_noise_increases_shrinkage()
    print('\nALL CORE UNIT TESTS PASSED')
