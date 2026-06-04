"""Unit tests for the numpy-only coreset core (no evalscope needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalscope_ext.pruning.calibration import beta_binomial_mean, item_stats
from evalscope_ext.pruning.coreset import select_coreset


def test_budget_and_uniqueness():
    rng = np.random.default_rng(0)
    n = 200
    feats = rng.normal(size=(n, 3))
    diff = rng.uniform(size=n)
    for ratio in (0.05, 0.1, 0.25):
        keep, bin_of = select_coreset(feats, diff, ratio)
        assert len(keep) == round(ratio * n)
        assert len(set(keep)) == len(keep)
        assert keep == sorted(keep)
    print('budget/uniqueness OK')


def test_determinism():
    rng = np.random.default_rng(1)
    feats, diff = rng.normal(size=(120, 2)), rng.uniform(size=120)
    a, _ = select_coreset(feats, diff, 0.2)
    b, _ = select_coreset(feats, diff, 0.2)
    assert a == b
    print('determinism OK')


def test_beats_random_when_difficulty_predicts_label():
    """When difficulty is a (discrete) prior that predicts accuracy - the real
    case - proportional difficulty stratification estimates an unseen model's
    score with less error than uniform random."""
    rng = np.random.default_rng(2)
    n = 400
    difficulty = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n)
    p = 0.95 - 0.8 * difficulty          # unseen model: harder -> lower accuracy
    y = (rng.uniform(size=n) < p).astype(float)
    truth = y.mean()
    feats = np.column_stack([difficulty, rng.normal(size=n)])

    keep, _ = select_coreset(feats, difficulty, 0.1, n_difficulty_bins=5, n_content_bins=1)
    core_err = abs(y[keep].mean() - truth)
    rand_err = np.mean([abs(y[np.random.default_rng(s).choice(n, len(keep), replace=False)].mean() - truth)
                        for s in range(200)])
    assert core_err < rand_err, (core_err, rand_err)
    print(f'variance-reduction OK: coreset {core_err:.4f} < random {rand_err:.4f}')


def test_judge_noise_smooths_difficulty():
    """More judge noise -> difficulty pulled harder toward the global mean."""
    clean = item_stats({'a': 1, 'b': 1, 'c': 1}, global_pass=0.5, judge_noise=0.0)
    noisy = item_stats({'a': 1, 'b': 1, 'c': 1}, global_pass=0.5, judge_noise=0.5)
    assert noisy['difficulty'] > clean['difficulty']      # all-correct -> less extreme when noisy
    assert beta_binomial_mean(3, 3, 0.5, 100) < 0.55      # strong prior dominates
    print('judge-noise smoothing OK')


if __name__ == '__main__':
    test_budget_and_uniqueness()
    test_determinism()
    test_beats_random_when_difficulty_predicts_label()
    test_judge_noise_smooths_difficulty()
    print('\nALL CORE UNIT TESTS PASSED')
