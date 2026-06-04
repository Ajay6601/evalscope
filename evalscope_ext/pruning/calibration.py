"""Difficulty/discrimination prior used to lay out the strata.

It's built once offline (see tools/calibrate.py) from past runs and saved as a
small JSON file per benchmark. The pruner just reads it; the model under test
never enters here. Anything that doesn't join falls back to a neutral 0.5 prior,
so a missing/partial file degrades gracefully.

Keys are the sample's position in the dataset (str(index)), which is how the
shipped reviews are indexed and works the same way for any benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


def beta_binomial_mean(correct: float, n: float, prior_mean: float, prior_strength: float) -> float:
    """Posterior mean pass-rate with a Beta prior - smooths noisy small-n counts."""
    a = prior_mean * prior_strength
    b = (1 - prior_mean) * prior_strength
    return (correct + a) / (n + a + b)


def item_stats(correct_by_model: Dict[str, int], *, global_pass: float, judge_noise: float) -> Dict[str, float]:
    """Difficulty + discrimination for one item from the pool's 0/1 outcomes.

    judge_noise is the grader's per-label flip rate (0 for an exact grader, >0 for
    an LLM judge). More noise -> stronger prior (more smoothing) and we subtract
    that noise from the cross-model variance so jitter doesn't look like signal.
    """
    outcomes = list(correct_by_model.values())
    m = max(len(outcomes), 1)
    correct = float(sum(outcomes))
    prior_strength = 2.0 + 8.0 * judge_noise
    pass_rate = beta_binomial_mean(correct, m, global_pass, prior_strength)

    p = correct / m
    var = max(0.0, p * (1 - p) - judge_noise * 0.25)
    return {
        'difficulty': round(1.0 - pass_rate, 5),
        'discrimination': round(min(1.0, var / 0.25), 5),
        'n_obs': m,
    }


@dataclass
class CalibrationTable:
    benchmark: str
    items: Dict[str, Dict[str, float]]
    neutral_difficulty: float = 0.5
    neutral_discrimination: float = 0.5

    @classmethod
    def load(cls, path: Path | str) -> 'CalibrationTable':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(benchmark=d.get('benchmark', '?'), items=d['items'])

    def get(self, key: str) -> Tuple[float, float, int]:
        row = self.items.get(key)
        if row is None:
            return self.neutral_difficulty, self.neutral_discrimination, 0
        return float(row['difficulty']), float(row['discrimination']), int(row.get('n_obs', 0))
