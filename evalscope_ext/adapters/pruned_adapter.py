"""One universal "pruned" wrapper that works for any benchmark.

Instead of writing a new class per benchmark, `register_pruned("foo")` takes the
already-registered "foo" adapter, makes a "foo_pruned" variant that loads exactly
like the original and then drops samples according to the chosen pruning strategy.
Adding a new benchmark is a single line at the bottom of this file.

Selection settings ride on the normal extra_params / --dataset-args channel, so
nothing in evalscope's CLI changes.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset
from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning import get_pruner
from evalscope_ext.pruning.calibration import CalibrationTable
from evalscope_ext.pruning.image_ops import degrade_data_url
from evalscope_ext.pruning.stratified_coreset import StratifiedCoresetPruner

logger = get_logger()
CALIBRATION_DIR = Path(__file__).resolve().parents[1] / 'calibration'

# extra_params every pruned benchmark accepts (declaring them lets dataset_args validation pass)
PRUNING_PARAMS: Dict[str, dict] = {
    'pruning_strategy': {'type': 'str', 'value': 'stratified_coreset',
                         'description': 'registered strategy name (stratified_coreset, encoder_stress)'},
    'prune_ratio': {'type': 'float', 'value': 0.15, 'description': 'fraction of samples to keep (0,1]'},
    'prune_seed': {'type': 'int', 'value': 0, 'description': 'seed for tie-breaking'},
    'pruner_args': {'type': 'dict', 'value': {}, 'description': 'extra kwargs for the strategy, e.g. {"alpha": 0.5}'},
    # for the multimodal probe:
    'perturb_level': {'type': 'int', 'value': 0,
                      'description': 'image degradation level 0-3 for the encoder probe (0 = off)'},
    'drop_images': {'type': 'bool', 'value': False,
                    'description': 'remove images (blind run) to check vision-necessity'},
}


def _pruned_meta(base_name: str, new_name: str, strategy: str, ratio: float) -> BenchmarkMeta:
    base = BENCHMARK_REGISTRY.get(base_name)
    if base is None:
        raise ValueError(f'{base_name!r} not registered; import evalscope.benchmarks first')
    meta = copy.deepcopy(base)
    meta.name = new_name
    meta.pretty_name = f'{base.pretty_name} (pruned)'
    meta.data_adapter = None
    params = copy.deepcopy(PRUNING_PARAMS)
    params['pruning_strategy']['value'] = strategy
    params['prune_ratio']['value'] = ratio
    meta.extra_params = {**(meta.extra_params or {}), **params}
    return meta


def _strip_images(sample):
    for m in (sample.input if isinstance(sample.input, list) else []):
        c = getattr(m, 'content', None)
        if isinstance(c, list):
            m.content = [p for p in c if getattr(p, 'type', None) != 'image']


def _perturb_images(sample, level):
    for m in (sample.input if isinstance(sample.input, list) else []):
        c = getattr(m, 'content', None)
        if not isinstance(c, list):
            continue
        for p in c:
            if getattr(p, 'type', None) == 'image' and isinstance(p.image, str):
                p.image = degrade_data_url(p.image, level)


class PrunedMixin:
    """Mix in *before* the base adapter. Set `base_benchmark_name` on the subclass."""

    base_benchmark_name: str = ''

    def _make_pruner(self):
        strategy = self.extra_params.get('pruning_strategy', 'stratified_coreset')
        pruner = get_pruner(
            strategy,
            prune_ratio=float(self.extra_params.get('prune_ratio', 0.15)),
            seed=int(self.extra_params.get('prune_seed', 0)),
            **dict(self.extra_params.get('pruner_args') or {}),
        )
        # the coreset strategy can use a difficulty prior if we shipped one for this benchmark
        if isinstance(pruner, StratifiedCoresetPruner):
            path = CALIBRATION_DIR / f'{self.base_benchmark_name}.json'
            if path.exists():
                pruner.calibration = CalibrationTable.load(path)
            else:
                logger.info(f'[pruning] no calibration for {self.base_benchmark_name}, using neutral prior')
        return pruner

    def load_dataset(self) -> DatasetDict:
        dataset = super().load_dataset()
        if not self.extra_params.get('pruning_strategy'):
            return dataset

        pruner = self._make_pruner()
        level = int(self.extra_params.get('perturb_level', 0) or 0)
        drop = bool(self.extra_params.get('drop_images', False))

        for subset in list(dataset.keys()):
            samples = list(dataset[subset])
            result = pruner.prune(samples, subset)
            kept = [samples[i] for i in result.keep_indices]
            for s in kept:
                if drop:
                    _strip_images(s)
                elif level > 0:
                    _perturb_images(s, level)
            new_ds = MemoryDataset(samples=kept, name=getattr(dataset[subset], 'name', subset))
            new_ds.reindex()
            dataset[subset] = new_ds
            logger.info(f'[pruning:{pruner.name}] {subset}: {len(samples)} -> {len(kept)} '
                        f'(perturb={level}, drop_images={drop}) {result.info}')
        return dataset


def register_pruned(base_name: str, strategy='stratified_coreset', ratio=0.15):
    """Register `<base_name>_pruned` for any already-registered benchmark."""
    base = BENCHMARK_REGISTRY.get(base_name)
    meta = _pruned_meta(base_name, f'{base_name}_pruned', strategy, ratio)
    cls = type(
        ''.join(p.capitalize() for p in base_name.split('_')) + 'PrunedAdapter',
        (PrunedMixin, base.data_adapter),
        {'base_benchmark_name': base_name},
    )
    register_benchmark(meta)(cls)
    return cls


# benchmarks we ship pruned variants of (one line each)
register_pruned('live_code_bench')
register_pruned('aa_lcr')
register_pruned('mmmu', strategy='encoder_stress', ratio=0.05)
