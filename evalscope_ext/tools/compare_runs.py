"""Compare a full evalscope run against a pruned run and emit a go/no-go verdict.

This is the third leg of the run contract::

    evalscope eval --model M --datasets live_code_bench       --output ./results_full
    evalscope eval --model M --datasets live_code_bench_pruned \
        --dataset-args '{"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.15}}}' \
        --output ./results_pruned
    python -m evalscope_ext.tools.compare_runs --full ./results_full --pruned ./results_pruned

It loads every report JSON under each output dir, matches the pruned dataset
``<base>_pruned`` to its full ``<base>`` for the same model, and reports the
score gap and whether both runs land on the same side of a go/no-go
``--threshold`` -- the number a sales engineer actually cares about.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple


def _load_reports(run_dir: str) -> List[dict]:
    """Return [{model, dataset, score}] for every evalscope report under ``run_dir``.

    Parses the report JSON directly (no evalscope import needed): an evalscope
    ``Report`` serialises ``dataset_name``, ``model_name`` and a top-level
    ``score`` (the first metric's micro-mean), which is exactly the headline
    accuracy we compare on.
    """
    files = glob.glob(os.path.join(os.path.normpath(run_dir), '**', '*.json'), recursive=True)
    out: List[dict] = []
    for p in files:
        if os.path.basename(p).startswith('_'):
            continue
        rec = _from_raw_json(p)
        if rec and rec.get('dataset'):
            rec['path'] = p
            out.append(rec)
    return out


def _from_raw_json(path: str) -> Optional[dict]:
    try:
        d = json.load(open(path, encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(d, dict) or 'dataset_name' not in d:
        return None
    score = d.get('score')
    if score is None and d.get('metrics'):
        score = d['metrics'][0].get('score')
    return {'model': d.get('model_name', '?'), 'dataset': d['dataset_name'], 'score': float(score or 0.0)}


def _index(reports: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r['model'], r['dataset']): r for r in reports}


def compare(full_dir: str, pruned_dir: str, threshold: float) -> dict:
    full = _index(_load_reports(full_dir))
    pruned = _index(_load_reports(pruned_dir))

    rows = []
    for (model, dataset), prep in sorted(pruned.items()):
        base = dataset[:-len('_pruned')] if dataset.endswith('_pruned') else dataset
        frep = full.get((model, base)) or full.get((model, dataset))
        if not frep:
            rows.append({'model': model, 'dataset': dataset, 'note': f'no full run for base {base!r}'})
            continue
        f, p = frep['score'], prep['score']
        f_go, p_go = f >= threshold, p >= threshold
        rows.append({
            'model': model, 'base_dataset': base,
            'full_score': round(f, 4), 'pruned_score': round(p, 4),
            'abs_error': round(abs(p - f), 4),
            'threshold': threshold,
            'full_decision': 'GO' if f_go else 'NO-GO',
            'pruned_decision': 'GO' if p_go else 'NO-GO',
            'decision_agrees': f_go == p_go,
        })
    return {'threshold': threshold, 'rows': rows}


def _print(result: dict) -> None:
    print(f"\nGo/No-Go threshold: {result['threshold']:.3f}\n")
    hdr = f"{'model':16} {'dataset':20} {'full':>7} {'pruned':>7} {'|err|':>7} {'full':>6} {'pruned':>7} {'agree':>6}"
    print(hdr); print('-' * len(hdr))
    for r in result['rows']:
        if 'note' in r:
            print(f"{r['model'][:16]:16} {r['dataset'][:20]:20}  -- {r['note']}")
            continue
        print(f"{r['model'][:16]:16} {r['base_dataset'][:20]:20} "
              f"{r['full_score']:>7.3f} {r['pruned_score']:>7.3f} {r['abs_error']:>7.3f} "
              f"{r['full_decision']:>6} {r['pruned_decision']:>7} {str(r['decision_agrees']):>6}")
    agree = [r for r in result['rows'] if 'decision_agrees' in r]
    if agree:
        n_ok = sum(r['decision_agrees'] for r in agree)
        print(f"\nDecision agreement: {n_ok}/{len(agree)} datasets  "
              f"(mean |err| = {sum(r['abs_error'] for r in agree)/len(agree):.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full', required=True, help='output dir of the full run')
    ap.add_argument('--pruned', required=True, help='output dir of the pruned run')
    ap.add_argument('--threshold', type=float, default=0.7,
                    help='go/no-go accuracy bar for the customer (default 0.7)')
    ap.add_argument('--json', dest='as_json', action='store_true', help='emit JSON only')
    args = ap.parse_args()
    result = compare(args.full, args.pruned, args.threshold)
    if args.as_json:
        print(json.dumps(result, indent=1))
    else:
        _print(result)


if __name__ == '__main__':
    main()
