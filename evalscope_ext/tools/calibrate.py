"""Build the difficulty prior for each benchmark from past per-sample reviews.

Run once, offline. Reads the shipped Evals/.../reviews (scores for a pool of
models) and writes calibration/<benchmark>.json. Keys are the sample index, so
the online pruner joins by position. The pool models are NOT the model you'll
test later - they just give a smoothed sense of which items are hard.

    python -m evalscope_ext.tools.calibrate --evals /path/to/Evals --out evalscope_ext/calibration
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evalscope_ext.pruning.calibration import item_stats

POOL = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']
# benchmark -> (review subdir, filename template, score field, judge noise estimate)
SPECS = {
    'live_code_bench': ('Part 1/reviews', 'live_code_bench_v5__{m}.jsonl', 'pass', 0.0),
    'aa_lcr': ('Part 1/reviews', 'aa_lcr__{m}.jsonl', 'acc', 0.06),
}


def _rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _score(row, field):
    v = row['sample_score']['score']['value']
    return int(float(v.get(field, v.get('acc', v.get('pass', 0.0)))) > 0.5)


def build(evals: Path, name: str, subdir, template, field, judge_noise):
    per_item = defaultdict(dict)  # index -> {model: 0/1}
    for m in POOL:
        path = evals / subdir / template.format(m=m)
        if not path.exists():
            print(f'  skip missing {path.name}')
            continue
        for row in _rows(path):
            per_item[str(row['index'])][m] = _score(row, field)

    n_correct = sum(sum(v.values()) for v in per_item.values())
    n_obs = sum(len(v) for v in per_item.values())
    global_pass = n_correct / max(n_obs, 1)
    items = {k: item_stats(v, global_pass=global_pass, judge_noise=judge_noise) for k, v in per_item.items()}
    return {'benchmark': name, 'judge_noise': judge_noise, 'n_items': len(items),
            'global_difficulty': round(1 - global_pass, 5), 'items': items}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evals', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (subdir, template, field, jn) in SPECS.items():
        art = build(args.evals, name, subdir, template, field, jn)
        (args.out / f'{name}.json').write_text(json.dumps(art, indent=1), encoding='utf-8')
        print(f'{name}: {art["n_items"]} items, global_difficulty={art["global_difficulty"]}')


if __name__ == '__main__':
    main()
