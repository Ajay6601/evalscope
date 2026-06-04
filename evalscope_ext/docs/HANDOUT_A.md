# Handout A: Why this works

## What I was actually trying to do

The goal isn't to grab a "representative" slice of the benchmark. The customer
asks one thing: is this model good enough for our workload, yes or no? So what I
really need is a small set of questions whose score still tells me three things:

- it gives about the same accuracy as running the full benchmark,
- it lands on the same side of the customer's pass/fail line, and
- it does all of that for a model I haven't seen yet.

That last point is the hard one, and it's what rules out the easy shortcuts. If I
pick questions based on which ones the three given models got wrong, I've just fit
to those three models and the answer won't hold for a fourth. So the selection has
to be based on facts about the questions themselves, never on the model's answers.

## The approach

I treat every question as having two hidden properties: how hard it is, and how
well it separates strong models from weak ones. I estimate both once, offline,
from past runs and save them to a small file. The pruner reads that file. It never
looks at the model being tested.

Then two steps:

1. Group the questions by difficulty (plus one cheap content feature), and keep
   questions from each group in proportion to how common that group is. Sampling
   proportionally like this gives an honest estimate of the full score with less
   noise than picking at random, and since the groups come from the questions'
   own properties, that still holds for a new model.
2. Inside each group, don't just take the hardest or the easiest. Take the
   questions that best tell models apart, and spread the picks out so that an odd
   weak spot in some future model still gets caught.

The difficulty number is smoothed (Beta-Binomial shrinkage) so a single noisy
grade can't swing it, and any question that doesn't match the file just falls back
to "average." A stale file makes the result a bit weaker, it doesn't break it.

## How much I cut, and why it's enough

I tested this the honest way: hold one model out, build the difficulty file from
the other two only, then check how well the kept set predicts the held-out model's
real score. No peeking at the model being predicted.

| Benchmark | Keep | Coreset error | Random error | "Hardest-only" error | Wrong go/no-go calls (ours vs random) |
|---|--:|--:|--:|--:|--:|
| LiveCodeBench v5 | 15% (47/315) | **0.038** | 0.047 | 0.430 | 0.04 vs 0.09 |
| LiveCodeBench v5 | 20% (63/315) | **0.038** | 0.046 | 0.396 | 0.03 vs 0.08 |
| AA-LCR | 20% (20/100) | 0.073 | 0.074 | 0.360 | 0.16 vs 0.12 |

For coding, keeping 15-20% gets within about 4 points of the true score and roughly
halves the rate of wrong yes/no calls compared to random. The banned "keep the
hardest" trick is off by 35-43 points, which is exactly why it's banned: it only
keeps questions everyone fails, so it tells you nothing about a good model.

For long-context I'll be straight about it. The cheap signals barely predict
accuracy here (the contexts are all long to begin with), and the LLM grader is
noisy enough that even the full 100-question score wobbles by about 0.05. No method
can beat that on the average, and mine doesn't pretend to: it matches random. What
it does add is steadier worst-case error and a guarantee that the hard "counting"
questions and the longest contexts actually show up. The practical takeaway: a
borderline long-context call needs more like 30% of the set, or a cleaner grader
(grade each answer a few times and take the majority).

## Part B: testing the image encoder, not the brain

If they add images later, the thing to watch for is the image encoder getting worse
when a model is served cheaply (lower precision, smaller vision part, lower
resolution). That's a different problem from the model just not being smart. An
encoder loses fine detail first: small text, dense tables, thin lines, subtle color.

So the probe picks MMMU questions that (a) lean on that kind of fragile visual
detail and (b) can't be answered without actually looking. I score each image on
simple things: how much fine detail and how many edges it has, how small it is, odd
aspect ratios, color complexity, number of images, plus a prior on image type
(tables, charts, diagrams, and medical images are high stress; plain photos are
low). It returns a high-stress set and a matched easy "control" set.

We can't open up the encoder, so we measure it through the normal image+text API in
three ways, and the first two are shipped as code on the pruned adapter, not just
described:

1. Drop the image and ask anyway (`drop_images`). Keep only the questions the model
   can't answer blind, so we know we're testing seeing and not guessing.
2. Damage the image on purpose and re-ask (`perturb_level`, the ladder in
   `image_ops.py`: shrink-then-grow, JPEG, blur, fewer colors). Watch how fast
   accuracy drops between level 0 and level 2. A good encoder barely notices; a
   degraded one falls off a cliff. Being smart can't fake this.
3. Compare the hard set against the easy control. Fine on control but bad on hard
   means an encoder problem. Bad on both means the model just isn't good. Random
   sampling can't tell those two apart.

So running the probe is just: run `mmmu_pruned` at `perturb_level` 0 and 2 and look
at the gap, then compare the probe half against the control half.

## Assumptions

- The models used to build the difficulty file cover the same range of hard and
  easy that a new model will see, and difficulty carries across models. It did:
  difficulty correlated -0.51 to -0.57 with the held-out model's results.
- Benchmarks are a few hundred to a few thousand items, so selection cost doesn't
  matter.
- The new model is graded the same way as the others and isn't gaming the subset.
- The join keys are stable; anything that doesn't match just uses the average prior.

## What I'd do with more

- **More models in the pool:** fit a proper item-response model for sharper
  difficulty. This would help long-context the most.
- **A live model to query:** probe it on a tiny warm-up set first, find where it's
  shaky, and spend the budget there. The image damage/blind runs already work; with
  a live endpoint I'd wire up the level-0-vs-level-2 sweep into one command.
- **More time:** grade the long-context answers a few times to kill the judge
  noise; add the reweighted estimator for a bit more precision; and learn the
  image-type stress prior from real low-precision vs full-precision gaps instead of
  setting it by hand.
