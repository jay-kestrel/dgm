"""
DeltaGraphModel (DGM)
=====================
A process-emergent, forward generative framework for synthesizing realistic
river delta channel networks (DCNs) and sampling large ensemble realizations.

Quickstart:
    >>> import dgm
    >>> G = dgm.generate(N=50, R_rt=0.4)
    >>> metrics = dgm.MetricsBattery().evaluate_all(G)
    >>> ensemble = dgm.sample_ensemble(N=50, R_rt=0.4, num_realizations=100)
"""

__version__ = "0.2.0"

from .config import DGMConfig, ParameterBounds
from .generator import (
    DeltaGenerator,
    EnsembleGenerator,
    generate,
    sample_ensemble,
    V2DeltaGenerator,
    V2EnsembleGenerator,
)
from .metrics import MetricsBattery, V2MetricsBattery
from .core.domain import PolarDomain
from .io.datasets import (
    DIDNTLoader,
    EmpiricalDelta,
    get_empirical_deltas,
    get_empirical_deltas_dict,
    DELTA_FORCING_REGIMES,
    CANONICAL_EMPIRICAL_DELTAS,
)
from .io.export import (
    to_graphml,
    to_geojson,
    metrics_to_csv,
    metrics_to_dataframe,
    summary_to_csv,
    summary_to_dataframe,
)
from .visualization import (
    plot_network,
    plot_delta,
    plot_tree_vs_network,
    plot_stages,
)
from .validation import (
    EmpiricalBenchmark,
    V2EmpiricalBenchmark,
    FULL_20D_METRIC_KEYS,
    FULL_20D_METRIC_LABELS,
    EXPANDED_METRIC_KEYS,
    EXPANDED_METRIC_LABELS,
    EMPIRICAL_5D_KEYS,
    EMPIRICAL_5D_LABELS,
    get_empirical_vector,
    get_empirical_5d_vector,
    compute_goodness_of_fit,
    compute_ks_test,
    compute_mahalanobis_validation,
    compute_mmd,
    compute_permanova,
)

# Backwards compatibility alias
V2Config = DGMConfig

__all__ = [
    "DGMConfig",
    "V2Config",
    "ParameterBounds",
    "DeltaGenerator",
    "EnsembleGenerator",
    "V2DeltaGenerator",
    "V2EnsembleGenerator",
    "generate",
    "sample_ensemble",
    "MetricsBattery",
    "V2MetricsBattery",
    "PolarDomain",
    "DIDNTLoader",
    "EmpiricalDelta",
    "get_empirical_deltas",
    "get_empirical_deltas_dict",
    "DELTA_FORCING_REGIMES",
    "CANONICAL_EMPIRICAL_DELTAS",
    "to_graphml",
    "to_geojson",
    "metrics_to_csv",
    "metrics_to_dataframe",
    "summary_to_csv",
    "summary_to_dataframe",
    "plot_network",
    "plot_delta",
    "plot_tree_vs_network",
    "plot_stages",
    "EmpiricalBenchmark",
    "V2EmpiricalBenchmark",
    "FULL_20D_METRIC_KEYS",
    "FULL_20D_METRIC_LABELS",
    "EXPANDED_METRIC_KEYS",
    "EXPANDED_METRIC_LABELS",
    "EMPIRICAL_5D_KEYS",
    "EMPIRICAL_5D_LABELS",
    "get_empirical_vector",
    "get_empirical_5d_vector",
    "compute_goodness_of_fit",
    "compute_ks_test",
    "compute_mahalanobis_validation",
    "compute_mmd",
    "compute_permanova",
]
