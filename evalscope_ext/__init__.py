"""evalscope_ext -- benchmark-pruning extension for evalscope.

Importing this package registers:
  * pruning strategies (``stratified_coreset``, ``encoder_stress``) in the
    pruner registry, and
  * pruned benchmark variants (``live_code_bench_pruned``, ``aa_lcr_pruned``,
    ``mmmu_pruned``) in evalscope's benchmark registry.

Use it by passing ``--datasets <name>_pruned`` to ``evalscope eval`` after the
extension is importable (e.g. installed, or on PYTHONPATH).
"""
from . import pruning  # noqa: F401  (registers pruners; pure-python, no evalscope dep)

# Registering the pruned benchmark variants needs evalscope itself. Import it
# lazily so the pruning algorithms (and the offline calibration/validation
# tooling) remain usable in an environment where evalscope is not installed.
try:
    from . import adapters  # noqa: F401  (registers pruned benchmarks)
    _ADAPTERS_AVAILABLE = True
except ModuleNotFoundError as e:  # pragma: no cover - depends on install
    if (e.name or '').split('.')[0] != 'evalscope':
        raise
    _ADAPTERS_AVAILABLE = False

__all__ = ['pruning']
__version__ = '0.1.0'
