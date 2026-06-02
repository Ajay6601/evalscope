"""Build difficulty/discrimination calibration artifacts from historical reviews.

This is the **offline** half of the pruning pipeline.  It consumes the shipped
``Evals/.../reviews`` (per-sample scores for a pool of models) and emits a small
JSON prior per benchmark into ``evalscope_ext/calibration/``.  Run it once; the
online pruner then reads the artifact and never needs the candidate model's
scores.

Usage
-----
    python -m evalscope_ext.tools.calibrate \
        --evals /path/to/Evals --out evalscope_ext/calibration

The pool models used here are *not* the model under test — they only supply a
smoothed estimate of latent item difficulty.  Using >=3 diverse models keeps the
prior from tracking any single model's idiosyncrasies.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evalscope_ext.pruning.calibration import compute_item_stats, prompt_key

# (benchmark name, review filename glob, score field, judge_noise, key builder)
#   judge_noise: 0.0 for the exact LCB sandbox grader; ~0.06 for the AA-LCR LLM
#   judge (an order-of-magnitude estimate of per-label flip rate -- see README).
PART1 = 'Part 1/reviews'


def _read(path: Path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _key_lcb(rev):
    # Positional index is the canonical LiveCodeBench sample id; the online
    # adapter joins by load position.  See live_code_bench_pruned.py.
    return str(rev['index'])


def _key_aalcr(rev):
    sm = rev['sample_score'].get('sample_metadata', {})
    return prompt_key(sm.get('question', ''))


def _key_mmmu(rev):
    sm = rev['sample_score'].get('sample_metadata', {})
    return sm.get('id') or str(rev['index'])


def _score(rev, field):
    val = rev['sample_score']['score']['value']
    return int(float(val.get(field, val.get('acc', val.get('pass', 0.0)))) > 0.5)


def build(benchmark, review_dir, files, score_field, judge_noise, key_fn):
    # gather correct-by-model per join key
    per_item = defaultdict(dict)  # key -> {model: 0/1}
    for model, fname in files:
        path = review_dir / fname
        if not path.exists():
            print(f'  (skip missing {fname})')
            continue
        for rev in _read(path):
            per_item[key_fn(rev)][model] = _score(rev, score_field)

    total_correct = sum(sum(v.values()) for v in per_item.values())
    total_obs = sum(len(v) for v in per_item.values())
    global_pass = total_correct / max(total_obs, 1)

    items = {
        key: compute_item_stats(by_model, global_pass=global_pass, judge_noise=judge_noise)
        for key, by_model in per_item.items()
    }
    return {
        'benchmark': benchmark,
        'judge_noise': judge_noise,
        'global_difficulty': round(1 - global_pass, 5),
        'n_pool_models': len(files),
        'n_items': len(items),
        'items': items,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evals', required=True, type=Path, help='path to the Evals/ root')
    ap.add_argument('--out', required=True, type=Path, help='output calibration dir')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pool = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
    specs = [
        ('live_code_bench_v5', args.evals / PART1,
         [(m, f'live_code_bench_v5__{m}.jsonl') for m in pool], 'pass', 0.0, _key_lcb),
        ('aa_lcr', args.evals / PART1,
         [(m, f'aa_lcr__{m}.jsonl') for m in pool], 'acc', 0.06, _key_aalcr),
    ]
    for benchmark, rdir, files, field, jn, key_fn in specs:
        print(f'[calibrate] {benchmark}')
        art = build(benchmark, rdir, files, field, jn, key_fn)
        out = args.out / f'{benchmark}.json'
        out.write_text(json.dumps(art, indent=1), encoding='utf-8')
        print(f'  {art["n_items"]} items  global_difficulty={art["global_difficulty"]}  -> {out}')


if __name__ == '__main__':
    main()
