# Handout A — Why this works (technical)

## The problem I actually solved

Not "find the easiest/representative N items." The customer asks one question —
*is this model good enough for our workload?* — so the real target is a **decision
estimator**: a small subset whose accuracy (a) is an unbiased, low-variance
estimate of the full-benchmark accuracy, (b) lands on the same side of a go/no-go
threshold as the full run, and (c) does both for **a model we have not seen**.
That last constraint is what rules out the forbidden baselines: anything that
selects by *which items the three shipped models got wrong* overfits to those
three. So selection must depend on **item properties**, not candidate scores.

I treat each item as having a latent difficulty (IRT-style) and a discrimination
(how sharply it separates strong from weak models). Two ideas combine:

1. **Stratify on a difficulty prior × an intrinsic content axis, allocate the
   keep-budget proportionally.** A proportional stratified mean is unbiased for
   the full mean and has variance ≤ uniform random (strictly lower when accuracy
   varies across strata). The strata are defined *without the candidate model*,
   so the estimator stays unbiased for a fourth model.
2. **Within each stratum pick by discrimination + feature-space coverage
   (k-center), never by score rank.** Discrimination surfaces the items that
   actually move a go/no-go call near the threshold; the coverage term spreads
   picks so a new model with an idiosyncratic weakness still has representative
   items.

The difficulty/discrimination prior is computed **once, offline** from historical
reviews (`tools/calibrate.py`) with Beta-Binomial shrinkage. It is a prior over
items, not the candidate's scores — the online pruner never sees the model under
test. Items that fail to join the prior fall back to neutral, so a stale
calibration degrades gracefully.

## How much I pruned, and why the subset is sufficient

Validation is **leave-one-model-out**: hold a model out, build the prior from the
other two only, estimate the held-out model's full accuracy from the coreset.
No leakage. (40 seeds × 3 held-out models; full table in `analysis/results/`.)

| Benchmark | Keep | Coreset MAE | Random MAE | Top-k-hardest | Decision flips core→rand |
|---|--:|--:|--:|--:|--:|
| LiveCodeBench v5 | 15% (47/315) | **0.038** | 0.047 | 0.430 | 0.04 → 0.09 |
| LiveCodeBench v5 | 20% (63/315) | **0.038** | 0.046 | 0.396 | 0.03 → 0.08 |
| AA-LCR | 20% (20/100) | 0.073 | 0.074 | 0.360 | 0.16 → 0.12 |

**Coding:** keep ~15–20% and recover full accuracy to ±4 pts with ~half the
decision-flip rate of random — a genuine, defensible cut. The forbidden
top-k-hardest baseline is off by 35–43 pts (it keeps only items everyone fails,
so it measures nothing about a good model), which is exactly why it's forbidden
and why a difficulty-*balanced* (not difficulty-*ranked*) selection is the point.

**Long-context, honestly:** within AA-LCR the cheap intrinsic axes barely move
accuracy (|corr(context-length, accuracy)| < 0.13 — the contexts are *all* long,
71k–115k tokens) and the LLM judge injects per-label noise, so the full 100-item
score itself has a binomial standard error of ~0.05. No selector can beat that
floor on the *mean*, and my numbers say so — the coreset ties random there rather
than pretending to win. Its value on AA-LCR is (i) lower tail error (p90), (ii)
judge-noise-aware difficulty so we don't chase flipped labels, and (iii)
guaranteed coverage of the genuinely harder `count` questions and the longest
contexts that a random draw of 10 can easily miss. The actionable finding: a
*borderline* long-context call needs ~20–30% of AA-LCR **or** a denoised judge
(repeated/ensemble judging) before the go/no-go is trustworthy.

## Part B — probing the image encoder specifically

If multimodal lands next quarter, the failure we must catch is **encoder
degradation** — a model served at fp8/int4, lower input resolution, or with a
smaller vision tower — *not* generic reasoning. An encoder loses information
first on high-spatial-frequency, small, low-contrast, multi-panel content. So the
probe selects MMMU items that are simultaneously (a) visually fragile and (b)
unanswerable without truly seeing the image.

**Selection** (`encoder_stress`, runs over the full ~12k): score each item on
intrinsic, model-free image features — Laplacian high-frequency energy, edge
density, smallest side, aspect extremity, color entropy, image count — plus an
`img_type` prior (tables/charts/diagrams/sheet-music/medical = high stress;
natural photos = low). Return a **subject-balanced high-stress probe** and a
**subject-matched low-stress control**. Balancing across subjects/subfields stops
us from accidentally measuring a domain gap.

**Measurement through the standard OpenAI image+text interface** (we can't open
the encoder, so we build controlled contrasts at the API):

1. **Vision-necessity filter** — run each candidate item with the image removed
   (or replaced by a neutral placeholder). Keep items the model *can't* answer
   blind. This guarantees the probe scores perception, not text priors / option
   leakage.
2. **Perturbation ladder** — re-ask each probe item under encoder-attacking
   degradations: downscale→upscale, JPEG, mild blur, color quantization. A
   faithful encoder is robust; a degraded one collapses. The **accuracy-vs-
   perturbation slope** is the headline encoder-fidelity number, and reasoning
   strength cannot fake robustness to resolution loss.
3. **Probe − control differential** — strong on control, weak on probe ⇒ encoder
   problem; weak on both ⇒ general capability gap. Random sampling cannot make
   this separation.

## Assumptions

- **Distribution:** the calibration pool spans the difficulty range the candidate
  will see, and item difficulty is partly model-agnostic (validated: leave-one-out
  difficulty correlates −0.51 to −0.57 with held-out correctness).
- **Scale:** benchmarks are 10²–10⁴ items; an O(n²)-per-stratum k-center is fine.
- **Model behavior:** no adversarial gaming of the kept subset; the candidate is
  graded by the same harness as the pool.
- **Join stability:** content/`id` keys are stable across dataset revisions;
  unmatched items get the neutral prior (graceful).

## What I'd change with more of each

- **More data (more pool models):** fit a real 2-parameter IRT model for sharper,
  better-resolved difficulty/discrimination — the current prior is deliberately
  shrunk because a 3-model pool is thin. This would most help AA-LCR.
- **A live endpoint:** make the prior *active* — query the candidate on a tiny
  seed set, locate it on the difficulty curve, and adaptively spend the remaining
  budget where this model is uncertain (CAT-style). For Part B, actually run the
  vision-necessity filter and perturbation ladder instead of describing them.
- **More time:** de-noise the AA-LCR judge (repeat/ensemble) before calibrating;
  add a reweighted (Neyman-allocation) estimator in `compare_runs` for a further
  variance cut; learn the `img_type` stress prior from real fp8-vs-bf16 deltas
  instead of hand-set weights.
