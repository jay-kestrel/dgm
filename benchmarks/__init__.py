"""
Benchmarks module forwarding to dgm.benchmarks.
"""

from dgm.benchmarks.empirical_validation import (
    METRIC_KEY_MAP,
    METRIC_LABELS,
    PRIMARY_METRICS,
    EmpiricalValidationBenchmark,
    compute_goodness_of_fit,
    run_empirical_benchmark,
)

__all__ = [
    "METRIC_KEY_MAP",
    "METRIC_LABELS",
    "PRIMARY_METRICS",
    "EmpiricalValidationBenchmark",
    "compute_goodness_of_fit",
    "run_empirical_benchmark",
]
