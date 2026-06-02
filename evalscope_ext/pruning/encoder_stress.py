"""``encoder_stress`` -- forward-looking multimodal probe selector (Part B).

Goal: pick the ~N MMMU items (out of the full ~12K) that most sharply expose
**image-encoder degradation** -- the failure mode when a vision-language model is
served at lower precision (fp8/int4), lower input resolution, or with a smaller
vision tower -- as opposed to generic reasoning gaps.

Why these items stress an *encoder* specifically
------------------------------------------------
An encoder loses information first on high-spatial-frequency, small, or
low-contrast visual content.  We score each image on intrinsic, model-free
features that proxy that fragility:

* ``hf_energy``      -- variance of the Laplacian: fine detail / thin lines / small text
* ``edge_density``   -- fraction of strong edges: dense schematics, tables, sheet music
* ``min_side``       -- small images get destroyed by patchification/resize
* ``aspect_extremity`` -- extreme aspect ratios force lossy resizes
* ``color_entropy``  -- subtle shading (histology, heatmaps) sensitive to quantization
* ``n_images``       -- multi-panel composition strains tiling / token budget
* ``type_prior``     -- per ``img_type`` prior (tables/charts/diagrams/medical high;
                        natural photographs low)

The selector returns a **stress probe** (high stress) and, when ``with_control``
is set, a **subject-matched low-stress control** so the evaluation can report a
*differential*: a model that holds up on the control but collapses on the probe
has an encoder problem, not a reasoning problem.  Selection is balanced across
subjects/subfields so the probe measures perception, not a domain skew.

The accompanying measurement protocol (vision-necessity contrast + a
perturbation ladder, both run over the standard OpenAI image+text interface) is
documented in Handout A; this module implements the *selection*.
"""
from __future__ import annotations

import base64
import io
import math
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from .base import PruneResult, PruningStrategy, register_pruner

# A-priori encoder-stress weight per MMMU img_type. High = the answer hinges on
# fine/dense visual structure that a degraded encoder drops first.
IMG_TYPE_PRIOR: Dict[str, float] = {
    'Tables': 1.0, 'Diagrams': 0.9, 'Plots and Charts': 0.9, 'Charts': 0.9,
    'Chemical Structures': 1.0, 'Sheet Music': 1.0, 'Mathematical Notations': 0.9,
    'Geometric Shapes': 0.7, 'Maps': 0.8, 'Technical Drawings': 0.95,
    'Medical Images': 0.95, 'Microscopic Images': 0.95, 'Pathology': 0.95,
    'DNA Sequences': 0.9, 'Trees and Graphs': 0.85, 'Circuit Diagrams': 0.95,
    'Photographs': 0.3, 'Paintings': 0.4, 'Portraits': 0.3, 'Comics and Cartoons': 0.5,
    'Logos and Icons': 0.4, 'Sculpture': 0.35, 'Screenshots': 0.6, 'Icons': 0.4,
}
DEFAULT_TYPE_PRIOR = 0.6

_STRESS_WEIGHTS = {
    'hf_energy': 0.30, 'edge_density': 0.20, 'small_side': 0.15,
    'aspect_extremity': 0.10, 'color_entropy': 0.10, 'multi_image': 0.05,
    'type_prior': 0.10,
}


def _decode_images(sample) -> List['np.ndarray']:
    """Return grayscale float arrays for every image attached to ``sample``."""
    try:
        from PIL import Image
    except Exception:  # pragma: no cover
        return []
    imgs = []
    messages = sample.input if isinstance(sample.input, list) else []
    for m in messages:
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
                data = base64.b64decode(raw) if isinstance(raw, str) else raw
                imgs.append(Image.open(io.BytesIO(data)).convert('RGB'))
            except Exception:
                continue
    return imgs


def _image_stress_features(sample) -> Dict[str, float]:
    """Intrinsic encoder-stress features for a sample's image(s)."""
    pil_imgs = _decode_images(sample)
    md = getattr(sample, 'metadata', {}) or {}
    img_types = md.get('img_type') or []
    if isinstance(img_types, str):
        img_types = [img_types]
    type_prior = max((IMG_TYPE_PRIOR.get(t, DEFAULT_TYPE_PRIOR) for t in img_types),
                     default=DEFAULT_TYPE_PRIOR)

    if not pil_imgs:
        return {'hf_energy': 0.0, 'edge_density': 0.0, 'small_side': 0.0,
                'aspect_extremity': 0.0, 'color_entropy': 0.0,
                'multi_image': 0.0, 'type_prior': type_prior}

    hf, edge, sides, aspects, entropies = [], [], [], [], []
    for im in pil_imgs:
        w, h = im.size
        g = np.asarray(im.convert('L'), dtype=float) / 255.0
        # Laplacian high-frequency energy (downscale-robust normalisation by area)
        lap = (-4 * g
               + np.roll(g, 1, 0) + np.roll(g, -1, 0)
               + np.roll(g, 1, 1) + np.roll(g, -1, 1))
        hf.append(float(lap.var()))
        edge.append(float((np.abs(lap) > 0.05).mean()))
        sides.append(min(w, h))
        aspects.append(abs(math.log((w + 1e-6) / (h + 1e-6))))
        # color entropy on a coarse 4x4x4 histogram
        arr = (np.asarray(im, dtype=int) // 64).reshape(-1, 3)
        idx = arr[:, 0] * 16 + arr[:, 1] * 4 + arr[:, 2]
        hist = np.bincount(idx, minlength=64).astype(float)
        p = hist / hist.sum()
        entropies.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 6.0))  # /log2(64)

    return {
        'hf_energy': float(np.mean(hf)),
        'edge_density': float(np.mean(edge)),
        # smaller side -> more stress, so invert (clamp at 1024 px)
        'small_side': float(1.0 - min(np.mean(sides), 1024) / 1024),
        'aspect_extremity': float(min(np.mean(aspects), 2.0) / 2.0),
        'color_entropy': float(np.mean(entropies)),
        'multi_image': float(min(len(pil_imgs) / 4.0, 1.0)),
        'type_prior': type_prior,
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


@register_pruner('encoder_stress')
class EncoderStressPruner(PruningStrategy):
    """Select a subject-balanced, high-image-stress probe set for MMMU.

    Tunables:
        with_control:  also reserve a subject-matched low-stress control half
                       (default True). Probe and control split the budget.
        subject_key:   sample.metadata field used for balancing (default 'subfield').
    """

    def __init__(self, prune_ratio: float = 0.05, seed: int = 0,
                 with_control: bool = True, subject_key: str = 'subfield', **kwargs):
        super().__init__(prune_ratio=prune_ratio, seed=seed, **kwargs)
        self.with_control = bool(with_control)
        self.subject_key = subject_key

    def _stress_scores(self, samples: Sequence) -> np.ndarray:
        rows = [_image_stress_features(s) for s in samples]
        keys = list(_STRESS_WEIGHTS)
        mat = np.array([[r[k] for k in keys] for r in rows], dtype=float)
        z = np.column_stack([_zscore(mat[:, j]) for j in range(mat.shape[1])])
        w = np.array([_STRESS_WEIGHTS[k] for k in keys])
        return z @ w

    def prune(self, samples: Sequence, subset: str) -> PruneResult:
        n = len(samples)
        if n == 0:
            return PruneResult(keep_indices=[])
        budget = max(1, int(round(self.prune_ratio * n)))
        stress = self._stress_scores(samples)

        # Balance across subjects so we measure perception, not a domain gap.
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            g = (getattr(s, 'metadata', {}) or {}).get(self.subject_key, '_')
            groups[str(g)].append(i)

        probe_budget = budget if not self.with_control else max(1, budget // 2)
        control_budget = budget - probe_budget

        def _alloc(total: int) -> Dict[str, int]:
            sizes = {g: len(ix) for g, ix in groups.items()}
            tot = sum(sizes.values())
            alloc = {g: int(total * sizes[g] / tot) for g in groups}
            # distribute remainder to largest groups
            rem = total - sum(alloc.values())
            for g in sorted(groups, key=lambda k: -sizes[k])[:rem]:
                alloc[g] += 1
            return alloc

        probe_alloc = _alloc(probe_budget)
        control_alloc = _alloc(control_budget) if control_budget else {}

        keep: List[int] = []
        role: Dict[int, str] = {}
        for g, ix in groups.items():
            order = sorted(ix, key=lambda i: (-stress[i], i))
            k_hi = min(probe_alloc.get(g, 0), len(order))
            for i in order[:k_hi]:
                keep.append(i); role[i] = 'probe'
            if control_budget:
                k_lo = min(control_alloc.get(g, 0), len(order) - k_hi)
                for i in order[len(order) - k_lo:] if k_lo else []:
                    keep.append(i); role[i] = 'control'
        keep.sort()
        return PruneResult(
            keep_indices=keep,
            stratum_of=role,
            diagnostics={
                'n_full': n, 'n_kept': len(keep), 'subset': subset,
                'n_probe': sum(v == 'probe' for v in role.values()),
                'n_control': sum(v == 'control' for v in role.values()),
                'mean_probe_stress': float(np.mean([stress[i] for i, r in role.items() if r == 'probe']))
                if any(r == 'probe' for r in role.values()) else 0.0,
            },
        )
