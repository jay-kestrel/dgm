"""
Benchmark suite module for DeltaGraphModel (DGM).
"""

from .empirical_validation import (
    METRIC_KEY_MAP,
    METRIC_LABELS,
    PRIMARY_METRICS,
    EmpiricalValidationBenchmark,
    compute_goodness_of_fit,
    run_empirical_benchmark,
)
from .mega_delta_benchmark import (
    DEFAULT_MEGA_DELTAS,
    MegaDeltaScaleBenchmark,
)

__all__ = [
    "DEFAULT_MEGA_DELTAS",
    "METRIC_KEY_MAP",
    "METRIC_LABELS",
    "PRIMARY_METRICS",
    "EmpiricalValidationBenchmark",
    "MegaDeltaScaleBenchmark",
    "compute_goodness_of_fit",
    "run_empirical_benchmark",
]
