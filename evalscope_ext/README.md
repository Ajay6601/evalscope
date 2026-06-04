# evalscope_ext - benchmark-pruning extension for evalscope

A plugin-style extension that adds **sample pruning** to evalscope: select the
smallest subset of a benchmark that still estimates a candidate model's full
score, so an expensive suite gives a customer a go/no-go answer cheaply.

> **Developed against evalscope commit `de7b0b3f08c617f48a00ef09f7169dc74212a6d9`**
> (`master`, fetched 2026-06-01). evalscope's adapter/registry APIs are evolving;
> this is the SHA the extension points below were written against.

## Why it lives where it does (extension points)

Everything hangs off evalscope's *public* extension surfaces - no core files are
patched, so this drops in as a fork package or an installed plugin:

| Concern | evalscope surface used |
|---|---|
| Register pruned benchmark variants | `@register_benchmark(BenchmarkMeta(...))` |
| Configure selection from the CLI | standard `extra_params` / `--dataset-args` channel |
| Hook selection into loading | override `DefaultDataAdapter.load_dataset()` in a mixin |
| Reuse a benchmark verbatim | subclass its adapter; derive its `BenchmarkMeta` |

A pruned benchmark is just `class XPrunedAdapter(PrunableAdapterMixin, XAdapter)`
registered under `x_pruned`. The mixin calls `super().load_dataset()` (the normal
load + prompt formatting), then replaces each subset with the indices a
`PruningStrategy` selected. Selection parameters arrive through `extra_params`
exactly like every other benchmark knob, so the core CLI is untouched.

We mirror evalscope's own `register_*` pattern for strategies:

```python
from evalscope_ext.pruning import register_pruner, PruningStrategy

@register_pruner('my_strategy')
class MyStrategy(PruningStrategy):
    def prune(self, samples, subset): ...
```

## One universal pruned adapter

There's no per-benchmark pruned class. `adapters/pruned_adapter.py` has a single
`register_pruned(name)` factory that takes any already-registered benchmark and
builds a `<name>_pruned` variant which loads exactly like the original, then drops
samples with the chosen strategy. Adding a benchmark is one line:

```python
register_pruned('live_code_bench')
register_pruned('aa_lcr')
register_pruned('mmmu', strategy='encoder_stress', ratio=0.05)
# register_pruned('gsm8k')   # <- any other benchmark, same line
```

The strategy pulls generic, model-free features off each Sample (prompt length,
#choices, #images) so nothing is hardcoded to one benchmark, and joins the
difficulty prior by sample position.

## Layout

```
pruning/
  base.py               PruningStrategy base class + registry (register_pruner)
  coreset.py            the stratified coreset selection (plain numpy, no evalscope dep)
  calibration.py        Beta-Binomial difficulty/discrimination prior + judge-noise term
  stratified_coreset.py Part A strategy: generic features + the prior -> coreset
  encoder_stress.py     Part B strategy: pick high-stress + control MMMU images
  image_ops.py          Part B: image degradations for the probe (the stress ladder)
adapters/
  pruned_adapter.py     universal PrunedMixin + register_pruned() factory
calibration/            shipped priors: live_code_bench.json, aa_lcr.json
tools/
  calibrate.py          offline: build priors from historical reviews
  compare_runs.py       compare a full run vs a pruned run -> go/no-go verdict
```

## The two strategies

**`stratified_coreset` (Part A).** Stratify items on a *fixed-width difficulty
prior* (latent, IRT-style, estimated once over a model pool - never the candidate)
crossed with an intrinsic content axis, allocate the keep-budget proportionally
(unbiased + variance-reducing stratified estimator), and pick within each stratum
by discrimination + feature-space coverage (k-center), not by score rank. The
difficulty prior uses Beta-Binomial shrinkage with a judge-noise term so a single
noisy LLM-judge label can't make an item look spuriously hard or discriminative.
Generalises to an unseen model because selection uses *item properties*, not which
items a known model failed. See `coreset.py` for the full rationale.

**`encoder_stress` (Part B).** Scores each MMMU item on intrinsic image-fragility
features (Laplacian high-frequency energy, edge density, small side, aspect
extremity, color entropy, image count, `img_type` prior) and returns a
subject-balanced **high-stress probe** plus a matched **low-stress control**, so
the eval can read a *differential*: a model that holds on the control but
collapses on the probe has an encoder problem, not a reasoning problem.

The measurement runs through the normal image+text API, with two extra knobs the
pruned adapter applies after selection (so it works end to end in evalscope):

- `perturb_level` (0-3): degrade the kept images with the stress ladder in
  `image_ops.py` (downscale, JPEG, blur, fewer colours). Run the probe at level 0
  vs level 2 and compare accuracy - the drop is the encoder-fidelity signal.
- `drop_images` (bool): remove the images (blind run) to confirm the questions
  actually need vision.

```bash
python -m evalscope_ext eval --model <m> --datasets mmmu_pruned \
  --dataset-args '{"mmmu_pruned": {"extra_params": {"perturb_level": 2}}}' --output ./p2
```

## Calibration is offline and model-agnostic

`tools/calibrate.py` reads historical per-sample reviews (here: the shipped
`Evals/`) and writes a small JSON prior per benchmark, keyed by sample position.
The online pruner reads that prior and **never sees the candidate model's
scores**. Items that don't join fall back to a neutral prior, so a stale or
partial calibration degrades gracefully instead of breaking.
