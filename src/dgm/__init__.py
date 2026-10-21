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
from .core.domain import PolarDomain
from .generator import (
    DeltaGenerator,
    EnsembleGenerator,
    V2DeltaGenerator,
    V2EnsembleGenerator,
    generate,
    sample_ensemble,
)
from .io.datasets import (
    CANONICAL_EMPIRICAL_DELTAS,
    DELTA_FORCING_REGIMES,
    DIDNTLoader,
    EmpiricalDelta,
    get_empirical_deltas,
    get_empirical_deltas_dict,
)
from .io.export import (
    metrics_to_csv,
    metrics_to_dataframe,
    summary_to_csv,
    summary_to_dataframe,
    to_geojson,
    to_graphml,
)
from .metrics import MetricsBattery, V2MetricsBattery
from .validation import (
    EMPIRICAL_5D_KEYS,
    EMPIRICAL_5D_LABELS,
    EXPANDED_METRIC_KEYS,
    EXPANDED_METRIC_LABELS,
    FULL_20D_METRIC_KEYS,
    FULL_20D_METRIC_LABELS,
    EmpiricalBenchmark,
    V2EmpiricalBenchmark,
    compute_goodness_of_fit,
    compute_ks_test,
    compute_mahalanobis_validation,
    compute_mmd,
    compute_permanova,
    get_empirical_5d_vector,
    get_empirical_vector,
)
from .visualization import (
    plot_delta,
    plot_network,
    plot_stages,
    plot_tree_vs_network,
)

# Backwards compatibility alias
V2Config = DGMConfig

__all__ = [
    "CANONICAL_EMPIRICAL_DELTAS",
    "DELTA_FORCING_REGIMES",
    "EMPIRICAL_5D_KEYS",
    "EMPIRICAL_5D_LABELS",
    "EXPANDED_METRIC_KEYS",
    "EXPANDED_METRIC_LABELS",
    "FULL_20D_METRIC_KEYS",
    "FULL_20D_METRIC_LABELS",
    "DGMConfig",
    "DIDNTLoader",
    "DeltaGenerator",
    "EmpiricalBenchmark",
    "EmpiricalDelta",
    "EnsembleGenerator",
    "MetricsBattery",
    "ParameterBounds",
    "PolarDomain",
    "V2Config",
    "V2DeltaGenerator",
    "V2EmpiricalBenchmark",
    "V2EnsembleGenerator",
    "V2MetricsBattery",
    "compute_goodness_of_fit",
    "compute_ks_test",
    "compute_mahalanobis_validation",
    "compute_mmd",
    "compute_permanova",
    "generate",
    "get_empirical_5d_vector",
    "get_empirical_deltas",
    "get_empirical_deltas_dict",
    "get_empirical_vector",
    "metrics_to_csv",
    "metrics_to_dataframe",
    "plot_delta",
    "plot_network",
    "plot_stages",
    "plot_tree_vs_network",
    "sample_ensemble",
    "summary_to_csv",
    "summary_to_dataframe",
    "to_geojson",
    "to_graphml",
]
