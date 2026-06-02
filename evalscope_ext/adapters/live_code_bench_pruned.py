"""``live_code_bench_pruned`` -- LiveCodeBench with stratified-coreset pruning.

Inherits all of :class:`LiveCodeBenchAdapter` (loading, sandbox grading, date
filtering) and only adds the pruning hook + the benchmark-specific feature/key
glue the coreset pruner needs.
"""
from __future__ import annotations

import math
from typing import List

from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter

from .pruning_mixin import PrunableAdapterMixin, make_pruned_meta


def _user_text(sample) -> str:
    """Concatenate the textual content of a sample's messages."""
    parts: List[str] = []
    msgs = sample.input if isinstance(sample.input, list) else [sample.input]
    for m in msgs:
        c = getattr(m, 'content', m)
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                t = getattr(p, 'text', None)
                if t:
                    parts.append(t)
    return '\n'.join(parts)


@register_benchmark(make_pruned_meta('live_code_bench', 'live_code_bench_pruned'))
class LiveCodeBenchPrunedAdapter(PrunableAdapterMixin, LiveCodeBenchAdapter):

    # difficulty/discrimination prior built offline by tools.calibrate
    calibration_file = 'live_code_bench_v5.json'

    def feature_fn(self, sample) -> List[float]:
        # Prompt length is the cheapest intrinsic complexity proxy for a coding
        # problem (longer statement -> more constraints / edge cases). The
        # difficulty prior supplies the rest of the signal via the strata.
        return [math.log1p(len(_user_text(sample)))]

    def calibration_key(self, sample, position: int) -> str:
        # Positional index is LiveCodeBench's canonical sample id; the shipped
        # calibration is keyed the same way (see tools/calibrate.py::_key_lcb).
        # Unmatched samples fall back to the neutral prior, so a reordering only
        # softens -- never breaks -- selection.
        return str(position)
