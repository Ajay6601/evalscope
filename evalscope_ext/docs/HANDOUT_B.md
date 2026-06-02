# Handout B — Why this matters and how to use it

*For developers, test engineers, product, and the customer team.*

## What changes for the customer conversation

Today, answering "is this model good enough for your workload?" means running the
whole coding and long-context suite on every candidate — slow and expensive, so
it happens late and rarely. These pruners let us answer the **same question from
~15–20% of the work**. On coding we reproduce the full-benchmark score within
about 4 points and, more importantly, we land on the **same go/no-go verdict** as
the full run with roughly half the error rate of just sampling questions at
random. Concretely: a go/no-go that used to take a full overnight sweep becomes a
same-meeting answer, and we can screen several candidate models in the time one
used to take.

It also makes us honest where honesty matters. For long-context, the tool shows
that within an already-long benchmark the cheap signal is weak and the grader
(an LLM judge) is noisy — so for a *borderline* customer we say plainly "this
needs a bigger sample or a cleaner judge," instead of shipping a confident number
that's really a coin-flip. That candor protects the relationship.

## How to run it tomorrow (go/no-go in three commands)

```bash
# full reference run (once, or reuse an existing one)
evalscope eval --model <candidate> --datasets live_code_bench --output ./full

# the cheap pruned run — 15% of the samples
python -m evalscope_ext eval --model <candidate> --datasets live_code_bench_pruned \
    --dataset-args '{"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.15}}}' \
    --output ./pruned

# the verdict, against the customer's bar (e.g. 70%)
python -m evalscope_ext.tools.compare_runs --full ./full --pruned ./pruned --threshold 0.70
```

The last command prints a one-line verdict per benchmark: full score, pruned
score, the gap, and whether both clear the customer's threshold — i.e. **GO /
NO-GO, and whether the cheap run agrees with the full one.** Swap
`live_code_bench` for `aa_lcr` for long-context; raise `prune_ratio` to 0.3 when a
result is borderline and you want more confidence.

## What the multimodal probe gives you that random sampling cannot

When the customer adds images next quarter, the question becomes "can this model
actually *see* well enough?" A random MMMU sample blends "can't see" with "can't
reason," so a model can score poorly for the wrong reason — or score *well* by
reasoning around a weak image encoder. Our probe deliberately picks the images
that stress an encoder (dense tables, charts, diagrams, fine medical detail) and
pairs every hard item with an easy visual control. Then a model that does fine on
the controls but falls apart on the stress set is flagged as having an **image-
encoder problem specifically** — exactly the regression that appears when a model
is compressed for fast serving. Random sampling cannot isolate that; it just
gives a blurry average.

## Why a customer-facing PM should care

- **Faster, cheaper qualification** → we can say yes/no to a prospect in a
  meeting, and re-qualify when a new model or a faster-serving variant ships.
- **Defensible numbers** → the subset is chosen by item difficulty and content,
  not cherry-picked, and it's validated to hold up on a model we'd never seen, so
  the answer survives customer scrutiny.
- **Catches the regression that bites in production** → the multimodal probe
  surfaces "the optimized model can't read the chart anymore" *before* the
  customer does.
