"""Pruning strategy base class + a small registry, in the same spirit as
evalscope's own register_benchmark / register_metric.

A strategy takes the loaded samples for one subset and returns which ones to
keep. It only gets the items and an offline difficulty prior, never the model's
answers, so the choice stays fair for a model we haven't tested.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

try:
    from evalscope.api.dataset import Sample
except Exception:  # lets the numpy core be used without evalscope installed
    Sample = object  # type: ignore


@dataclass
class PruneResult:
    keep_indices: List[int]
    bin_of: Dict[int, str] = field(default_factory=dict)        # kept index -> bin/role label
    bin_weight: Dict[str, float] = field(default_factory=dict)  # bin -> share of full set
    info: Dict[str, object] = field(default_factory=dict)


class PruningStrategy(abc.ABC):
    """Base class. Construction kwargs come from the benchmark extra_params, so a
    strategy can expose its own knobs (alpha, bins, ...)."""

    name: str = ''

    def __init__(self, prune_ratio: float = 0.1, seed: int = 0, **kwargs):
        if not 0 < prune_ratio <= 1:
            raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')
        self.prune_ratio = float(prune_ratio)
        self.seed = int(seed)
        self.params = kwargs

    @abc.abstractmethod
    def prune(self, samples: Sequence['Sample'], subset: str) -> PruneResult:
        ...


PRUNER_REGISTRY: Dict[str, type] = {}


def register_pruner(name: str) -> Callable[[type], type]:
    def wrap(cls: type) -> type:
        if not issubclass(cls, PruningStrategy):
            raise TypeError(f'{cls!r} is not a PruningStrategy')
        if name in PRUNER_REGISTRY and PRUNER_REGISTRY[name] is not cls:
            raise KeyError(f'pruner {name!r} already registered')
        cls.name = name
        PRUNER_REGISTRY[name] = cls
        return cls
    return wrap


def get_pruner(name: str, **kwargs) -> PruningStrategy:
    if name not in PRUNER_REGISTRY:
        raise KeyError(f'unknown pruning_strategy {name!r}; have {sorted(PRUNER_REGISTRY)}')
    return PRUNER_REGISTRY[name](**kwargs)


def available_pruners() -> List[str]:
    return sorted(PRUNER_REGISTRY)
