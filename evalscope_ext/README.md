# evalscope_ext - benchmark-pruning extension for evalscope

A plugin-style extension that adds **sample pruning** to evalscope: select the
smallest subset of a benchmark that still estimates a candidate model's full
score, so an expensive suite gives a customer a go/no-go answer cheaply.

> **This fork is based on evalscope commit `de7b0b3f08c617f48a00ef09f7169dc74212a6d9`**
> (`master`, fetched 2026-06-01). evalscope's adapter/registry APIs are evolving;
> this is the SHA the extension points below were written against.

## Install & run (inside this fork)

`evalscope_ext/` lives at the repo root next to `evalscope/`. evalscope's own
`pyproject.toml` discovers packages with the glob `evalscope*`, which already
matches `evalscope_ext`, so one editable install ships both:

```bash
pip install -e .                         # installs evalscope AND evalscope_ext
# run a pruned variant via the shim that registers it, then delegates to evalscope:
python -m evalscope_ext eval --model <model> --datasets live_code_bench_pruned \
    --dataset-args '{"live_code_bench_pruned": {"extra_params": {"pruning_strategy": "stratified_coreset", "prune_ratio": 0.15}}}' \
    --output ./results_pruned
python -m evalscope_ext.tools.compare_runs --full ./results_full --pruned ./results_pruned --threshold 0.7
```

Self-contained unit tests: `python evalscope_ext/tests/test_pruning_core.py`.
Handouts and the leave-one-model-out validation results are under `docs/`.

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

## Layout

```
pruning/
  base.py               PruningStrategy ABC + PRUNER_REGISTRY + register_pruner
  coreset.py            numpy-only stratified coreset (the algorithm; no evalscope dep)
  calibration.py        Beta-Binomial difficulty/discrimination prior + judge-noise term
  stratified_coreset.py Part A strategy (wraps coreset.py onto Sample objects)
  encoder_stress.py     Part B strategy (image-encoder stress probe selector)
adapters/
  pruning_mixin.py      PrunableAdapterMixin (the load_dataset hook) + make_pruned_meta
  live_code_bench_pruned.py / aa_lcr_pruned.py / mmmu_pruned.py
calibration/            shipped priors: live_code_bench_v5.json, aa_lcr.json
tools/
  calibrate.py          offline: build priors from historical reviews
  compare_runs.py        compare a full run vs a pruned run -> go/no-go verdict
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
collapses on the probe has an encoder problem, not a reasoning problem. The
measurement protocol (vision-necessity contrast + perturbation ladder over the
OpenAI image+text interface) is in `../HANDOUT_A.md`.

## Calibration is offline and model-agnostic

`tools/calibrate.py` reads historical per-sample reviews (here: the shipped
`Evals/`) and writes a small JSON prior per benchmark. The online pruner reads
that prior and **never sees the candidate model's scores**. Items that fail to
join fall back to a neutral prior, so a stale or partial calibration degrades
gracefully rather than breaking. Join keys: positional index (LCB), question
content-hash (AA-LCR), stable `id` (MMMU).

## Tests

`../tests/test_integration.py` reconstructs samples from the shipped data,
instantiates each pruned adapter through `get_benchmark`, and exercises the
pruning hook (feature extraction, calibration join, selection, dataset
replacement) with no network or model calls.
