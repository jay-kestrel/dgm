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

from .config import DGMConfig
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
)
from .io.export import (
    to_graphml,
    to_geojson,
    metrics_to_csv,
    metrics_to_dataframe,
)

# Backwards compatibility alias
V2Config = DGMConfig

__all__ = [
    "DGMConfig",
    "V2Config",
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
    "to_graphml",
    "to_geojson",
    "metrics_to_csv",
    "metrics_to_dataframe",
]
