"""Adapter mixin that applies a pruning strategy at dataset-load time.

This is the single, narrow extension point into evalscope.  A *pruned* benchmark
subclasses its normal adapter plus this mixin; the mixin overrides
:meth:`load_dataset` to (1) load + post-process exactly as the base adapter does,
then (2) replace each subset with the indices a registered
:class:`~evalscope_ext.pruning.PruningStrategy` chose.

Selection parameters arrive through evalscope's standard ``extra_params`` channel
(i.e. ``--dataset-args '{"<name>": {"extra_params": {...}}}'``), so nothing about
the core CLI changes.  A concrete pruned adapter only has to declare which
pruners it allows and -- for coreset-style pruners -- how to turn a sample into
features and how to join it to the calibration prior.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning import get_pruner
from evalscope_ext.pruning.base import PruningStrategy
from evalscope_ext.pruning.calibration import CalibrationTable
from evalscope_ext.pruning.stratified_coreset import StratifiedCoresetPruner

logger = get_logger()

# extra_params spec entries shared by every pruned benchmark. Declaring them here
# means dataset_args validation accepts them (see BenchmarkMeta._update_extra_params).
PRUNING_EXTRA_PARAMS: Dict[str, dict] = {
    'pruning_strategy': {
        'type': 'str', 'value': 'stratified_coreset',
        'description': 'Registered pruning strategy name (e.g. stratified_coreset, encoder_stress).',
    },
    'prune_ratio': {
        'type': 'float', 'value': 0.15,
        'description': 'Fraction of samples to keep, in (0, 1].',
    },
    'prune_seed': {
        'type': 'int', 'value': 0,
        'description': 'Deterministic seed for tie-breaking during selection.',
    },
    'pruner_args': {
        'type': 'dict', 'value': {},
        'description': 'Extra keyword args forwarded to the pruning strategy (e.g. {"alpha": 0.5}).',
    },
}


def make_pruned_meta(base_name: str, new_name: str, *,
                     default_strategy: str = 'stratified_coreset',
                     default_ratio: float = 0.15) -> BenchmarkMeta:
    """Derive a pruned-variant ``BenchmarkMeta`` from an existing benchmark.

    Copies the upstream metadata verbatim (dataset id, prompt template, metrics,
    subset list, ...), renames it, and merges in the pruning ``extra_params`` so
    the variant is configured exactly like its parent plus selection controls.
    """
    base = BENCHMARK_REGISTRY.get(base_name)
    if base is None:
        raise ValueError(f'base benchmark {base_name!r} is not registered; '
                         'import evalscope.benchmarks before building pruned variants')
    meta = copy.deepcopy(base)
    meta.name = new_name
    meta.pretty_name = f'{base.pretty_name} (pruned)'
    meta.data_adapter = None  # set by register_benchmark on the subclass
    extra = dict(meta.extra_params or {})
    merged = copy.deepcopy(PRUNING_EXTRA_PARAMS)
    merged['pruning_strategy']['value'] = default_strategy
    merged['prune_ratio']['value'] = default_ratio
    extra.update(merged)
    meta.extra_params = extra
    return meta


class PrunableAdapterMixin:
    """Mix in *before* the concrete data adapter in the MRO."""

    #: Calibration artifact filename (under evalscope_ext/calibration/), or None.
    calibration_file: Optional[str] = None

    def _calibration_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / 'calibration'

    # ----- hooks a concrete pruned adapter overrides for coreset pruners ----- #
    def feature_fn(self, sample) -> List[float]:
        """Intrinsic, model-independent features for one sample."""
        raise NotImplementedError

    def calibration_key(self, sample, position: int) -> str:
        """Stable key joining ``sample`` to its calibration row."""
        raise NotImplementedError

    # ------------------------------------------------------------------------ #
    def _build_pruner(self) -> PruningStrategy:
        name = self.extra_params.get('pruning_strategy', 'stratified_coreset')
        ratio = float(self.extra_params.get('prune_ratio', 0.15))
        seed = int(self.extra_params.get('prune_seed', 0))
        args = dict(self.extra_params.get('pruner_args') or {})
        pruner = get_pruner(name, prune_ratio=ratio, seed=seed, **args)

        # Coreset-style pruners need benchmark glue + the difficulty prior.
        if isinstance(pruner, StratifiedCoresetPruner):
            calibration = None
            if self.calibration_file:
                path = self._calibration_dir() / self.calibration_file
                if path.exists():
                    calibration = CalibrationTable.load(path)
                else:
                    logger.warning(f'[pruning] calibration not found: {path}; using neutral prior')
            pruner.configure(
                feature_fn=self.feature_fn,
                key_fn=self.calibration_key,
                calibration=calibration,
            )
        return pruner

    def load_dataset(self) -> DatasetDict:
        dataset_dict = super().load_dataset()  # full load + prompt formatting
        if not self.extra_params.get('pruning_strategy'):
            return dataset_dict

        pruner = self._build_pruner()
        for subset in list(dataset_dict.keys()):
            ds = dataset_dict[subset]
            samples = list(ds)
            result = pruner.prune(samples, subset)
            kept = [samples[i] for i in result.keep_indices]
            pruned = MemoryDataset(
                samples=kept,
                name=getattr(ds, 'name', subset),
                location=getattr(ds, 'location', None),
                shuffled=getattr(ds, 'shuffled', False),
            )
            pruned.reindex()
            dataset_dict[subset] = pruned
            logger.info(
                f'[pruning:{pruner.name}] subset={subset} '
                f'{len(samples)} -> {len(kept)} samples ({result.diagnostics})'
            )
        return dataset_dict
