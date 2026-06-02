"""``mmmu_pruned`` -- MMMU with the forward-looking encoder-stress probe (Part B).

Defaults to the ``encoder_stress`` strategy, which needs no calibration prior:
it scores items by intrinsic image-fragility features and returns a
subject-balanced high-stress probe (plus an optional low-stress control).
"""
from __future__ import annotations

from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter

from .pruning_mixin import PrunableAdapterMixin, make_pruned_meta


@register_benchmark(
    make_pruned_meta('mmmu', 'mmmu_pruned',
                     default_strategy='encoder_stress', default_ratio=0.05)
)
class MMMUPrunedAdapter(PrunableAdapterMixin, MMMUAdapter):
    # encoder_stress works purely from image features + img_type metadata, so no
    # calibration file and no feature_fn/calibration_key glue are required.
    calibration_file = None
