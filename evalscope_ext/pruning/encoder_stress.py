"""encoder_stress - the multimodal probe (Part B).

Picks the MMMU items that lean hardest on fine visual detail, the stuff an image
encoder loses first when a model is served cheaply (low precision / resolution /
smaller vision tower). It scores each image on simple, model-free features and
returns a subject-balanced high-stress set plus a matched easy "control" set, so
you can compare the two and tell an encoder problem apart from a plain reasoning gap.

Run the probe (via mmmu_pruned) at perturb_level 0 vs 2 to see how fast accuracy
drops - that slope is the encoder-fidelity signal. The degradations live in
image_ops.py.
"""
from __future__ import annotations

import base64
import io
import math
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from .base import PruneResult, PruningStrategy, register_pruner

# how much each image type tends to stress an encoder (answer depends on dense/fine detail)
IMG_TYPE_PRIOR = {
    'Tables': 1.0, 'Diagrams': 0.9, 'Plots and Charts': 0.9, 'Charts': 0.9,
    'Chemical Structures': 1.0, 'Sheet Music': 1.0, 'Mathematical Notations': 0.9,
    'Geometric Shapes': 0.7, 'Maps': 0.8, 'Technical Drawings': 0.95,
    'Medical Images': 0.95, 'Microscopic Images': 0.95, 'Pathology': 0.95,
    'DNA Sequences': 0.9, 'Trees and Graphs': 0.85, 'Circuit Diagrams': 0.95,
    'Photographs': 0.3, 'Paintings': 0.4, 'Portraits': 0.3, 'Comics and Cartoons': 0.5,
    'Logos and Icons': 0.4, 'Sculpture': 0.35, 'Screenshots': 0.6, 'Icons': 0.4,
}
DEFAULT_PRIOR = 0.6
WEIGHTS = {'hf': 0.30, 'edges': 0.20, 'small': 0.15, 'aspect': 0.10,
           'color': 0.10, 'multi': 0.05, 'type': 0.10}


def _images(sample):
    from PIL import Image
    out = []
    msgs = sample.input if isinstance(sample.input, list) else []
    for m in msgs:
        content = getattr(m, 'content', None)
        if not isinstance(content, list):
            continue
        for part in content:
            if getattr(part, 'type', None) != 'image':
                continue
            raw = part.image
            try:
                if isinstance(raw, str) and raw.startswith('data:'):
                    raw = raw.split(',', 1)[1]
                out.append(Image.open(io.BytesIO(base64.b64decode(raw))).convert('RGB'))
            except Exception:
                pass
    return out


def _image_features(sample) -> Dict[str, float]:
    imgs = _images(sample)
    md = getattr(sample, 'metadata', {}) or {}
    types = md.get('img_type') or []
    if isinstance(types, str):
        types = [types]
    type_score = max((IMG_TYPE_PRIOR.get(t, DEFAULT_PRIOR) for t in types), default=DEFAULT_PRIOR)
    if not imgs:
        return {'hf': 0, 'edges': 0, 'small': 0, 'aspect': 0, 'color': 0, 'multi': 0, 'type': type_score}

    hf, edges, sides, aspect, color = [], [], [], [], []
    for im in imgs:
        w, h = im.size
        g = np.asarray(im.convert('L'), float) / 255
        lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1))
        hf.append(lap.var())
        edges.append((np.abs(lap) > 0.05).mean())
        sides.append(min(w, h))
        aspect.append(abs(math.log((w + 1e-6) / (h + 1e-6))))
        q = (np.asarray(im, int) // 64).reshape(-1, 3)
        hist = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(float)
        prob = hist / hist.sum()
        color.append(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum() / 6)
    return {
        'hf': float(np.mean(hf)),
        'edges': float(np.mean(edges)),
        'small': float(1 - min(np.mean(sides), 1024) / 1024),  # smaller image = more stress
        'aspect': float(min(np.mean(aspect), 2) / 2),
        'color': float(np.mean(color)),
        'multi': float(min(len(imgs) / 4, 1)),
        'type': type_score,
    }


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


@register_pruner('encoder_stress')
class EncoderStressPruner(PruningStrategy):
    """Knobs (via extra_params): with_control (default True), subject_key."""

    def __init__(self, prune_ratio=0.05, seed=0, with_control=True, subject_key='subfield', **kwargs):
        super().__init__(prune_ratio=prune_ratio, seed=seed, **kwargs)
        self.with_control = bool(with_control)
        self.subject_key = subject_key

    def _stress(self, samples) -> np.ndarray:
        keys = list(WEIGHTS)
        rows = np.array([[_image_features(s)[k] for k in keys] for s in samples], float)
        z = np.column_stack([_z(rows[:, j]) for j in range(rows.shape[1])])
        return z @ np.array([WEIGHTS[k] for k in keys])

    def prune(self, samples: Sequence, subset: str) -> PruneResult:
        n = len(samples)
        if n == 0:
            return PruneResult(keep_indices=[])
        budget = max(1, round(self.prune_ratio * n))
        stress = self._stress(samples)

        # balance picks across subjects so we measure seeing, not a subject we're weak at
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            groups[str((getattr(s, 'metadata', {}) or {}).get(self.subject_key, '_'))].append(i)

        probe_budget = budget if not self.with_control else max(1, budget // 2)
        control_budget = budget - probe_budget

        def split(total):
            tot = sum(len(v) for v in groups.values())
            a = {g: int(total * len(ix) / tot) for g, ix in groups.items()}
            for g in sorted(groups, key=lambda k: -len(groups[k]))[: total - sum(a.values())]:
                a[g] += 1
            return a

        probe_alloc, control_alloc = split(probe_budget), (split(control_budget) if control_budget else {})
        keep, role = [], {}
        for g, ix in groups.items():
            ranked = sorted(ix, key=lambda i: (-stress[i], i))  # high stress first
            hi = min(probe_alloc.get(g, 0), len(ranked))
            for i in ranked[:hi]:
                keep.append(i); role[i] = 'probe'
            lo = min(control_alloc.get(g, 0), len(ranked) - hi) if control_budget else 0
            for i in (ranked[len(ranked) - lo:] if lo else []):
                keep.append(i); role[i] = 'control'
        keep.sort()
        probe = [i for i in keep if role[i] == 'probe']
        return PruneResult(
            keep_indices=keep, bin_of=role,
            info={'n_full': n, 'n_kept': len(keep),
                  'n_probe': len(probe), 'n_control': len(keep) - len(probe),
                  'mean_probe_stress': float(np.mean([stress[i] for i in probe])) if probe else 0.0,
                  'subset': subset},
        )
