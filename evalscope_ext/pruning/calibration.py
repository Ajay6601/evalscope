"""Difficulty / discrimination *priors* used to lay out pruning strata.

A calibration table is a small JSON artifact computed **once, offline** from any
historical runs (see ``tools/calibrate.py``).  It is a *prior over item
properties*, not the candidate model's scores — the pruner reads it but the model
under test never enters the selection.  Items that fail to join fall back to a
neutral prior, so a stale or partial table degrades gracefully rather than
breaking.

Two quantities per item:

* ``difficulty`` in [0,1] — 1 minus the shrunk pass-rate across the calibration
  model pool (higher = harder).
* ``discrimination`` in [0,1] — how strongly the item separates models in the
  pool (variance of correctness), with an explicit correction for LLM-judge noise.

Both use Beta-Binomial shrinkage toward the global mean.  The shrinkage strength
is increased for judge-graded benchmarks so that a single noisy 0/1 judgement
cannot make an item look spuriously easy/hard or discriminative.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

_WS = re.compile(r'\s+')


def prompt_key(text: str) -> str:
    """Stable join key: sha1 of whitespace-normalised prompt text.

    Used to join calibration rows to live ``Sample`` objects without relying on
    positional ordering (which is not guaranteed across dataset revisions).
    """
    norm = _WS.sub(' ', (text or '').strip()).lower()
    return hashlib.sha1(norm.encode('utf-8')).hexdigest()


def beta_binomial_mean(correct: float, n: float, prior_mean: float, prior_strength: float) -> float:
    """Posterior mean of a Beta(prior) + Binomial(observed) pass-rate."""
    a0 = prior_mean * prior_strength
    b0 = (1 - prior_mean) * prior_strength
    return (correct + a0) / (n + a0 + b0)


@dataclass
class CalibrationTable:
    benchmark: str
    judge_noise: float
    global_difficulty: float
    items: Dict[str, Dict[str, float]]
    neutral_difficulty: float = 0.5
    neutral_discrimination: float = 0.5

    @classmethod
    def load(cls, path: Path | str) -> 'CalibrationTable':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(
            benchmark=d['benchmark'],
            judge_noise=float(d.get('judge_noise', 0.0)),
            global_difficulty=float(d.get('global_difficulty', 0.5)),
            items=d['items'],
        )

    def get(self, key: str) -> Tuple[float, float, int]:
        """Return ``(difficulty, discrimination, n_obs)`` for ``key``.

        Missing keys return the neutral prior with ``n_obs=0`` so callers can
        track join coverage.
        """
        row = self.items.get(key)
        if row is None:
            return self.neutral_difficulty, self.neutral_discrimination, 0
        return float(row['difficulty']), float(row['discrimination']), int(row.get('n_obs', 0))


def compute_item_stats(
    correct_by_model: Dict[str, int],
    *,
    global_pass: float,
    judge_noise: float,
) -> Dict[str, float]:
    """Difficulty + discrimination for one item from its pool of 0/1 outcomes.

    ``judge_noise`` is the per-label flip variance of the grader (0 for an exact
    sandbox grader, >0 for an LLM judge).  It (a) inflates the shrinkage prior so
    noisy items are pulled toward the mean, and (b) is subtracted from the
    observed cross-model variance so judge jitter is not mistaken for genuine
    discrimination.
    """
    outcomes = list(correct_by_model.values())
    m = len(outcomes)
    correct = float(sum(outcomes))

    # More judge noise -> stronger prior -> more shrinkage toward the global mean.
    prior_strength = 2.0 + 8.0 * judge_noise
    pass_rate = beta_binomial_mean(correct, m, global_pass, prior_strength)
    difficulty = 1.0 - pass_rate

    # Discrimination = cross-model variance, judge-noise-corrected and rescaled to
    # [0,1] (max possible Bernoulli variance is 0.25).
    p = correct / m if m else 0.0
    var_obs = p * (1 - p)
    var_corrected = max(0.0, var_obs - judge_noise * 0.25)
    discrimination = min(1.0, var_corrected / 0.25)
    return {
        'difficulty': round(difficulty, 5),
        'discrimination': round(discrimination, 5),
        'n_obs': m,
    }
