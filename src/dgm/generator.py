"""
Unified Facade Generator module for DeltaGraphModel (DGM).
Provides:
  - DeltaGenerator: single realization generator
  - EnsembleGenerator: stochastic cloud / ensemble sampling generator
  - generate(): convenient top-level function for single graphs
  - sample_ensemble(): convenient top-level function for ensembles
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
import networkx as nx

from .config import DGMConfig
from .core.domain import PolarDomain
from .core.backbone import generate_polar_tree_backbone
from .core.loops import apply_loop_mechanism
from .core.hydraulics import apply_emergent_hydraulics
from .metrics import MetricsBattery


_UNSET = object()


class DeltaGenerator:
    """
    Primary generator for single synthetic delta graph realizations.
    """

    def __init__(self, config: Optional[DGMConfig] = None, **overrides):
        import copy
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = DGMConfig.load_default()

        if overrides:
            for k, v in overrides.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
                self.config.distributions.pop(k, None)

        self.battery = MetricsBattery()

    def generate(self, seed_override: Any = _UNSET) -> nx.DiGraph:
        """
        Generates a single synthetic delta graph realization given N and R_rt.
        """
        seed = self.config.seed if seed_override is _UNSET else seed_override
        rng = np.random.default_rng(seed)
        sub_seed = int(rng.integers(0, 2**31 - 1)) if seed is None else seed

        # Get parameter set (sampled from distributions or deterministic defaults)
        params = self.config.get_active_parameters(rng)

        # 1. Initialize Polar Domain
        domain = PolarDomain(
            r_shore=params['r_shore'],
            theta_base=params['theta_base'],
            theta_delta=params['theta_delta'],
            R_rt=self.config.R_rt,
            confinement=params.get('confinement', self.config.confinement),
            lateral_valley_width=params.get('lateral_valley_width', self.config.lateral_valley_width)
        )

        # 2. Generate Polar Growth Tree Backbone
        G_tree = generate_polar_tree_backbone(
            N_target=self.config.N,
            R_rt=self.config.R_rt,
            domain=domain,
            params=params,
            rng=rng
        )

        # 3. Apply Emergent Loop Mechanism (Mechanism A or Mechanism B)
        G_looped = apply_loop_mechanism(
            G=G_tree,
            domain=domain,
            mechanism=self.config.loop_mechanism,
            params=params,
            rng=rng,
            seed=sub_seed
        )

        # 4. Apply Emergent Hydraulics & Channel Widening
        G_final = apply_emergent_hydraulics(
            G=G_looped,
            params=params,
            seed=sub_seed
        )

        # Attach run metadata to graph
        G_final.graph['N_input'] = self.config.N
        G_final.graph['R_rt'] = self.config.R_rt
        G_final.graph['confinement'] = domain.confinement
        G_final.graph['lateral_valley_width'] = domain.lateral_valley_width
        G_final.graph['coastal_node_decay'] = params.get('coastal_node_decay', self.config.coastal_node_decay)
        G_final.graph['coastal_sinuosity_decay'] = params.get('coastal_sinuosity_decay', self.config.coastal_sinuosity_decay)
        G_final.graph['loop_mechanism'] = self.config.loop_mechanism
        G_final.graph['params'] = params
        G_final.graph['seed'] = seed

        return G_final

    def evaluate_metrics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Evaluates full 20-metric battery on graph G."""
        params = G.graph.get('params', None)
        return self.battery.evaluate_all(G, params)


class EnsembleGenerator:
    """
    Ensemble generator for sampling a stochastic cloud of graph realizations
    conditioned on (N, R_rt) and parameter distributions.
    """

    def __init__(
        self,
        N: int = 50,
        R_rt: float = 0.5,
        num_realizations: int = 100,
        loop_mechanism: str = "potential_field",
        sample_parameters: bool = True,
        base_seed: int = 42,
        config: Optional[DGMConfig] = None,
        **overrides
    ):
        self.N = N
        self.R_rt = R_rt
        self.num_realizations = num_realizations
        self.loop_mechanism = loop_mechanism
        self.sample_parameters = sample_parameters
        self.base_seed = base_seed

        import copy
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = DGMConfig.load_default()

        self.config.N = self.N
        self.config.R_rt = self.R_rt
        self.config.loop_mechanism = self.loop_mechanism
        self.config.sample_parameters = self.sample_parameters

        if overrides:
            for k, v in overrides.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
                self.config.distributions.pop(k, None)

        self.battery = MetricsBattery()

    def generate_ensemble(self) -> List[nx.DiGraph]:
        """Generates an array of M graph realizations."""
        if self.base_seed is None:
            raise ValueError('base_seed must be an integer, got None')

        import copy
        graphs = []
        for i in range(self.num_realizations):
            seed = self.base_seed + i * 1000

            # Clone config with realization-specific seed
            cfg = copy.deepcopy(self.config)
            cfg.seed = seed

            gen = DeltaGenerator(cfg)
            G = gen.generate(seed_override=seed)
            graphs.append(G)

        return graphs

    def evaluate_ensemble(self) -> Dict[str, Any]:
        """
        Generates ensemble graphs and computes summary statistics across the cloud realization.
        """
        graphs = self.generate_ensemble()
        all_metrics = [self.battery.evaluate_all(g, g.graph.get('params')) for g in graphs]

        if not all_metrics:
            return {
                'N_input': self.N,
                'R_rt': self.R_rt,
                'num_realizations': 0,
                'metrics': {},
                'raw_realizations': []
            }

        metric_keys = list(all_metrics[0].keys())
        summary = {
            'N_input': self.N,
            'R_rt': self.R_rt,
            'num_realizations': self.num_realizations,
            'loop_mechanism': self.loop_mechanism,
            'metrics': {},
            'raw_realizations': all_metrics
        }

        for k in metric_keys:
            vals = [m[k] for m in all_metrics if isinstance(m[k], (int, float))]
            if vals:
                summary['metrics'][k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'p25': float(np.percentile(vals, 25)),
                    'p75': float(np.percentile(vals, 75))
                }

        return summary


def generate(
    N: int = 50,
    R_rt: float = 0.5,
    confinement: Optional[float] = None,
    lateral_valley_width: Optional[float] = None,
    coastal_node_decay: Optional[float] = None,
    coastal_sinuosity_decay: Optional[float] = None,
    config: Optional[Union[str, DGMConfig]] = None,
    seed: Any = _UNSET,
    **overrides
) -> nx.DiGraph:
    """
    Convenient top-level function to generate a single synthetic delta graph.
    """
    if isinstance(config, str):
        cfg = DGMConfig.from_yaml(config, N=N, R_rt=R_rt, **overrides)
    elif isinstance(config, DGMConfig):
        cfg = copy.deepcopy(config)
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        cfg = DGMConfig.load_default()
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if confinement is not None:
        cfg.confinement = float(confinement)

    if lateral_valley_width is not None:
        cfg.lateral_valley_width = float(lateral_valley_width)

    if coastal_node_decay is not None:
        cfg.coastal_node_decay = float(coastal_node_decay)

    if coastal_sinuosity_decay is not None:
        cfg.coastal_sinuosity_decay = float(coastal_sinuosity_decay)

    if seed is not _UNSET:
        cfg.seed = seed

    gen = DeltaGenerator(cfg)
    return gen.generate()


def sample_ensemble(
    N: int = 50,
    R_rt: float = 0.5,
    confinement: Optional[float] = None,
    lateral_valley_width: Optional[float] = None,
    coastal_node_decay: Optional[float] = None,
    coastal_sinuosity_decay: Optional[float] = None,
    num_realizations: int = 50,
    config: Optional[Union[str, DGMConfig]] = None,
    base_seed: int = 42,
    **overrides
) -> List[nx.DiGraph]:
    """
    Convenient top-level function to generate an ensemble of synthetic delta graphs.
    """
    if isinstance(config, str):
        cfg = DGMConfig.from_yaml(config, N=N, R_rt=R_rt, **overrides)
    elif isinstance(config, DGMConfig):
        cfg = copy.deepcopy(config)
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        cfg = DGMConfig.load_default()
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if confinement is not None:
        cfg.confinement = float(confinement)
        cfg.distributions.pop("confinement", None)

    if lateral_valley_width is not None:
        cfg.lateral_valley_width = float(lateral_valley_width)
        cfg.distributions.pop("lateral_valley_width", None)

    if coastal_node_decay is not None:
        cfg.coastal_node_decay = float(coastal_node_decay)
        cfg.distributions.pop("coastal_node_decay", None)

    if coastal_sinuosity_decay is not None:
        cfg.coastal_sinuosity_decay = float(coastal_sinuosity_decay)
        cfg.distributions.pop("coastal_sinuosity_decay", None)

    ens = EnsembleGenerator(
        N=N,
        R_rt=R_rt,
        num_realizations=num_realizations,
        base_seed=base_seed,
        config=cfg,
        **overrides
    )
    return ens.generate_ensemble()


# Backwards-compatibility aliases
V2DeltaGenerator = DeltaGenerator
V2EnsembleGenerator = EnsembleGenerator
