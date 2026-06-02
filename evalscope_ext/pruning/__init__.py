"""Pruning strategies for evalscope (registry + built-in strategies)."""
from .base import (  # noqa: F401
    PRUNER_REGISTRY,
    PruneResult,
    PruningStrategy,
    available_pruners,
    get_pruner,
    register_pruner,
)

# Import strategy modules so their @register_pruner decorators run on package import.
from . import stratified_coreset  # noqa: E402,F401
from . import encoder_stress  # noqa: E402,F401
