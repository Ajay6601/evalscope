"""``aa_lcr_pruned`` -- AA-LCR with stratified-coreset pruning.

Long-context reasoning difficulty is dominated by judge noise within an
already-long context set, so the calibration prior here is built with an explicit
judge-noise term (see tools/calibrate.py) and the features capture the axes that
*do* carry signal: question type (aggregation/count vs lookup) and context size.
"""
from __future__ import annotations

import math
import re
from typing import List

from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter

from ..pruning.calibration import prompt_key
from .pruning_mixin import PrunableAdapterMixin, make_pruned_meta

_LIST_RE = re.compile(r'\b(list|all|which .* mention|in alphabetical|every|each)\b', re.I)
_COUNT_RE = re.compile(r'\b(how many|number of|count|total)\b', re.I)


def _qtype_code(question: str) -> int:
    if _COUNT_RE.search(question):
        return 1  # count: cross-document tallying (empirically the hardest)
    if _LIST_RE.search(question):
        return 2  # aggregate: multi-doc retrieval + synthesis
    return 0      # lookup: single-fact retrieval


@register_benchmark(make_pruned_meta('aa_lcr', 'aa_lcr_pruned'))
class AALCRPrunedAdapter(PrunableAdapterMixin, AALCRAdapter):

    calibration_file = 'aa_lcr.json'

    def feature_fn(self, sample) -> List[float]:
        md = sample.metadata or {}
        urls = md.get('data_source_urls', '') or ''
        n_docs = len([u for u in urls.split(';') if u.strip()])
        question = md.get('question', '')
        return [
            math.log1p(md.get('input_tokens') or 0),
            float(n_docs),
            math.log1p(len(question)),
            float(_qtype_code(question)),
        ]

    def calibration_key(self, sample, position: int) -> str:
        # Content key on the question text -> robust to reordering.
        return prompt_key((sample.metadata or {}).get('question', ''))
