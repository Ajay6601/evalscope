"""Thin CLI shim so the pruned benchmarks are registered before evalscope runs.

evalscope has no general plugin-discovery for third-party benchmarks, so the
extension's ``@register_benchmark`` decorators must execute before the eval
command parses ``--datasets``. Importing this module does exactly that and then
hands control to evalscope's own CLI, so usage mirrors ``evalscope`` 1:1::

    python -m evalscope_ext eval --model <model> --datasets live_code_bench_pruned \
        --dataset-args '{"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.15}}}' \
        --output ./results_pruned

Equivalently, ``import evalscope_ext`` at the top of any script that calls
``evalscope.run_task(...)``.
"""
import evalscope_ext  # noqa: F401  (side effect: registers pruned benchmarks + pruners)
from evalscope.cli.cli import run_cmd

if __name__ == '__main__':
    run_cmd()
