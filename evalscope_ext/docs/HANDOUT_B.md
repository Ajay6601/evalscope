# Handout B: Why this matters and how to use it

For developers, test engineers, product, and the customer team.

## What it changes for customer conversations

Right now, telling a prospect "yes, this model is good enough" means running the
whole coding and long-context test suite on every model we're considering. It's
slow and expensive, so it happens late. These pruners let us get the same answer
from about 15-20% of the work. On coding we land within about 4 points of the full
score and, more importantly, we usually reach the same yes/no decision, with about
half the mistakes you'd get from just testing random questions. In practice: a call
that used to need an overnight run can happen in the meeting, and we can check
several models in the time one used to take.

It also keeps us honest. For long-context, the tool tells us when the cheap test
isn't trustworthy (the grader is noisy and the questions don't separate models
well), so for a borderline customer we can say "we need a bigger sample or a better
grader" instead of handing over a confident number that's really a guess.

## How to run it tomorrow

```bash
# full reference run (once, or reuse an existing one)
evalscope eval --model <candidate> --datasets live_code_bench --output ./full

# the cheap pruned run, 15% of the questions
python -m evalscope_ext eval --model <candidate> --datasets live_code_bench_pruned \
    --dataset-args '{"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.15}}}' \
    --output ./pruned

# the verdict, against the customer's bar (say 70%)
python -m evalscope_ext.tools.compare_runs --full ./full --pruned ./pruned --threshold 0.70
```

The last command prints, per benchmark: the full score, the pruned score, the gap,
and whether both clear the customer's bar. In other words, go or no-go, and whether
the cheap run agrees with the full one. Use `aa_lcr` for long-context. If a result
is close to the line, bump `prune_ratio` to 0.3 for more confidence.

## What the image probe gives you that random sampling doesn't

When images come into play, the real question is "can the model actually see well
enough?" A random sample of image questions mixes up "can't see" and "can't
reason," so a model can look bad for the wrong reason, or look fine by reasoning
around a weak encoder. Our probe deliberately picks the images that are genuinely
hard to see (dense tables, charts, diagrams, fine medical detail) and pairs each
one with an easy image. If a model does fine on the easy ones but falls apart on
the hard ones, that points straight at the image encoder, which is the exact thing
that breaks when a model is squeezed for faster serving. Random sampling just gives
you a blurry average.

## Why a PM should care

- **Faster answers:** yes or no in a meeting, and easy to re-check whenever a new
  or faster model shows up.
- **Defensible:** the sample is chosen by the questions' own properties, not
  cherry-picked, and it was checked on a model it had never seen, so it holds up
  under scrutiny.
- **Catches real regressions:** the image probe flags "the faster model can't read
  the chart anymore" before the customer hits it.
