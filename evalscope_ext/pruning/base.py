"""Pruning-strategy extension point for evalscope.

A *pruning strategy* takes the fully-loaded list of evaluation ``Sample`` objects
for one subset and returns the subset of sample indices to keep.  This mirrors
evalscope's own ``register_benchmark`` / ``register_metric`` registries so that a
maintainer reading the code finds a familiar pattern.

Strategies are intentionally model-agnostic: they receive the dataset and a
*difficulty prior* (a calibration artifact computed once, offline, from historical
runs), never the candidate model's scores.  This is what makes a pruned set
defensible for a model we have not seen.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

# Imported lazily inside type-checking only blocks to avoid a hard evalscope
# dependency when the pruning algorithms are unit-tested in isolation.
try:  # pragma: no cover - exercised indirectly
    from evalscope.api.dataset import Sample
except Exception:  # pragma: no cover
    Sample = object  # type: ignore


@dataclass
class PruneResult:
    """Outcome of a pruning pass for a single subset."""

    keep_indices: List[int]
    """Positions (into the input sample list) to retain, in stable order."""

    stratum_of: Dict[int, str] = field(default_factory=dict)
    """Map kept position -> stratum label, for diagnostics / reweighting."""

    stratum_weight: Dict[str, float] = field(default_factory=dict)
    """Full-set frequency of each stratum (sum to 1). Lets a downstream
    estimator reweight to stay unbiased even under non-proportional allocation."""

    diagnostics: Dict[str, object] = field(default_factory=dict)


class PruningStrategy(abc.ABC):
    """Base class for all sample-pruning strategies.

    Subclasses implement :meth:`prune`.  Construction kwargs come straight from
    the benchmark ``extra_params`` (i.e. from ``--dataset-args``), so a strategy
    may expose tunables such as ``alpha`` or ``n_difficulty_bins``.
    """

    #: Registry name; set by :func:`register_pruner`.
    name: str = ''

    def __init__(self, prune_ratio: float = 0.1, seed: int = 0, **kwargs):
        if not 0 < prune_ratio <= 1:
            raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')
        self.prune_ratio = float(prune_ratio)
        self.seed = int(seed)
        self.params = kwargs

    @abc.abstractmethod
    def prune(self, samples: Sequence['Sample'], subset: str) -> PruneResult:
        """Return the indices of ``samples`` to keep for ``subset``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
PRUNER_REGISTRY: Dict[str, type] = {}


def register_pruner(name: str) -> Callable[[type], type]:
    """Class decorator registering a :class:`PruningStrategy` under ``name``."""

    def _wrap(cls: type) -> type:
        if not issubclass(cls, PruningStrategy):
            raise TypeError(f'{cls!r} is not a PruningStrategy')
        if name in PRUNER_REGISTRY and PRUNER_REGISTRY[name] is not cls:
            raise KeyError(f'pruner {name!r} already registered')
        cls.name = name
        PRUNER_REGISTRY[name] = cls
        return cls

    return _wrap


def get_pruner(name: str, **kwargs) -> PruningStrategy:
    """Instantiate a registered pruning strategy."""
    if name not in PRUNER_REGISTRY:
        raise KeyError(
            f'unknown pruning_strategy {name!r}; registered: {sorted(PRUNER_REGISTRY)}'
        )
    return PRUNER_REGISTRY[name](**kwargs)


def available_pruners() -> List[str]:
    return sorted(PRUNER_REGISTRY)
