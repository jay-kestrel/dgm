"""
Benchmark suite module for DeltaGraphModel (DGM).
"""

from .empirical_validation import (
    EmpiricalValidationBenchmark,
    run_empirical_benchmark,
    PRIMARY_METRICS,
    METRIC_LABELS,
    METRIC_KEY_MAP,
    compute_goodness_of_fit,
)
from .mega_delta_benchmark import (
    MegaDeltaScaleBenchmark,
    DEFAULT_MEGA_DELTAS,
)

__all__ = [
    "EmpiricalValidationBenchmark",
    "run_empirical_benchmark",
    "MegaDeltaScaleBenchmark",
    "DEFAULT_MEGA_DELTAS",
    "PRIMARY_METRICS",
    "METRIC_LABELS",
    "METRIC_KEY_MAP",
    "compute_goodness_of_fit",
]
