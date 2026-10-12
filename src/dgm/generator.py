"""
Unified Facade Generator module for DeltaGraphModel (DGM).
Provides:
  - DeltaGenerator: single realization generator
  - EnsembleGenerator: stochastic cloud / ensemble sampling generator
  - generate(): convenient top-level function for single graphs
  - sample_ensemble(): convenient top-level function for ensembles
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import copy
import numpy as np
import networkx as nx

from .config import DGMConfig
from .core.domain import PolarDomain
from .core.backbone import generate_polar_tree_backbone
from .core.loops import apply_loop_mechanism
from .core.pruning import prune_and_simplify_network
from .core.hydraulics import apply_emergent_hydraulics
from .core.channel_geometry import generate_channel_centerlines, apply_channel_proximity_constraints
from .metrics import MetricsBattery


_UNSET = object()


class DeltaGenerator:
    """
    Primary generator for single synthetic delta graph realizations.
    """

    def __init__(
        self,
        config: Optional[DGMConfig] = None,
        target_nodes: Optional[int] = None,
        **overrides
    ):
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = DGMConfig.load_default()

        if target_nodes is not None:
            self.config.N = int(target_nodes)
            self.config.distributions.pop('N', None)

        if overrides:
            for k, v in overrides.items():
                if k == 'target_nodes':
                    self.config.N = int(v)
                    self.config.distributions.pop('N', None)
                elif hasattr(self.config, k):
                    setattr(self.config, k, v)
                self.config.distributions.pop(k, None)

        self.config.validate()
        self.battery = MetricsBattery()

    def generate(
        self,
        seed_override: Any = _UNSET,
        return_stages: bool = False
    ) -> Union[nx.DiGraph, Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]]:
        """
        Generates a single synthetic delta graph realization given N and R_rt.

        Parameters
        ----------
        seed_override : Any, optional
            Optional seed to override instance configuration.
        return_stages : bool, default False
            If True, returns a tuple (G_final, stages) where stages is a dictionary
            mapping stage names ('tree', 'looped', 'pruned', 'final') to their graphs.

        Returns
        -------
        Union[nx.DiGraph, Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]]
            Generated graph G_final, or (G_final, stages) if return_stages is True.
        """
        seed = self.config.seed if seed_override is _UNSET else seed_override
        if seed is not None:
            try:
                seed = int(seed)
            except (ValueError, TypeError):
                pass
        rng = np.random.default_rng(seed)
        sub_seed = int(rng.integers(0, 2**31 - 1)) if seed is None else int(seed)

        # Get parameter set (sampled from distributions or deterministic defaults)
        params = self.config.get_active_parameters(rng)

        # 1. Initialize Polar Domain
        try:
            domain = PolarDomain(
                r_shore=params['r_shore'],
                theta_base=params['theta_base'],
                theta_delta=params['theta_delta'],
                R_rt=self.config.R_rt,
                confinement=params.get('confinement', self.config.confinement),
                lateral_valley_width=params.get('lateral_valley_width', self.config.lateral_valley_width)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PolarDomain: {e}") from e

        # 2. Generate Polar Growth Tree Backbone
        try:
            G_tree = generate_polar_tree_backbone(
                N_target=self.config.N,
                R_rt=self.config.R_rt,
                domain=domain,
                params=params,
                rng=rng
            )
        except Exception as e:
            raise RuntimeError(
                f"DGM polar tree backbone generation failed for N={self.config.N}, R_rt={self.config.R_rt}: {e}"
            ) from e
        G_tree_snapshot = G_tree.copy()

        # 3. Apply Emergent Loop Mechanism (Mechanism A or Mechanism B)
        try:
            G_looped = apply_loop_mechanism(
                G=G_tree,
                domain=domain,
                mechanism=self.config.loop_mechanism,
                params=params,
                rng=rng,
                seed=sub_seed
            )
        except Exception as e:
            raise RuntimeError(
                f"DGM loop mechanism '{self.config.loop_mechanism}' failed: {e}"
            ) from e
        G_looped_snapshot = G_looped.copy()

        # 4. Apply Pre-Hydraulic Network Pruning & Simplification
        if self.config.enable_pruning or params.get('enable_pruning', False):
            try:
                G_pruned = prune_and_simplify_network(
                    G=G_looped,
                    domain=domain,
                    params=params,
                    rng=rng
                )
            except Exception as e:
                raise RuntimeError(f"DGM network pruning and reach simplification failed: {e}") from e
        else:
            G_pruned = G_looped
        G_pruned_snapshot = G_pruned.copy()

        # 5. Apply Emergent Hydraulics & Channel Widening
        try:
            G_final = apply_emergent_hydraulics(
                G=G_pruned,
                params=params,
                seed=sub_seed
            )
        except Exception as e:
            raise RuntimeError(f"DGM emergent hydraulics and flux relaxation failed: {e}") from e

        # 6. Apply Channel Proximity Constraints & Curved Polyline Geometry
        curved_enabled = bool(self.config.enable_curved_centerlines or params.get('enable_curved_centerlines', False))
        clr_frac = params.get('min_channel_clearance', self.config.min_channel_clearance)
        pts_num = params.get('num_polyline_points', self.config.num_polyline_points)
        bends_num = params.get('meander_bends', self.config.meander_bends)

        if curved_enabled:
            G_final = generate_channel_centerlines(
                G=G_final,
                clearance_fraction=clr_frac,
                num_points=pts_num,
                num_bends=bends_num,
                seed=sub_seed
            )
            # Also synthesize centerlines for the tree backbone snapshot
            try:
                generate_channel_centerlines(
                    G=G_tree_snapshot,
                    clearance_fraction=clr_frac,
                    num_points=pts_num,
                    num_bends=bends_num,
                    seed=sub_seed
                )
            except Exception:
                pass
        else:
            G_final = apply_channel_proximity_constraints(
                G=G_final,
                clearance_fraction=clr_frac
            )

        # Attach run metadata to graph
        G_final.graph['N_input'] = self.config.N
        G_final.graph['R_rt'] = self.config.R_rt
        G_final.graph['confinement'] = domain.confinement
        G_final.graph['lateral_valley_width'] = domain.lateral_valley_width
        G_final.graph['coastal_node_decay'] = params.get('coastal_node_decay', self.config.coastal_node_decay)
        G_final.graph['coastal_sinuosity_decay'] = params.get('coastal_sinuosity_decay', self.config.coastal_sinuosity_decay)
        pruning_enabled = bool(self.config.enable_pruning or params.get('enable_pruning', False))
        G_final.graph['enable_pruning'] = pruning_enabled
        G_final.graph['prune_inland_dead_ends'] = params.get('prune_inland_dead_ends', self.config.prune_inland_dead_ends)
        G_final.graph['min_spur_length'] = params.get('min_spur_length', self.config.min_spur_length)
        G_final.graph['simplify_pass_through_nodes'] = params.get('simplify_pass_through_nodes', self.config.simplify_pass_through_nodes)
        G_final.graph['max_prune_iterations'] = params.get('max_prune_iterations', self.config.max_prune_iterations)
        G_final.graph['enable_curved_centerlines'] = curved_enabled
        G_final.graph['min_channel_clearance'] = clr_frac
        G_final.graph['num_polyline_points'] = pts_num
        G_final.graph['meander_bends'] = bends_num
        G_final.graph['loop_mechanism'] = self.config.loop_mechanism
        G_final.graph['params'] = params
        G_final.graph['seed'] = seed

        stages = {
            'tree': G_tree_snapshot,
            'looped': G_looped_snapshot,
            'pruned': G_pruned_snapshot,
            'final': G_final.copy()
        }
        G_final.graph['stages'] = stages

        if return_stages:
            return G_final, stages
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
        target_nodes: Optional[int] = None,
        **overrides
    ):
        if num_realizations < 1:
            raise ValueError(f"num_realizations must be >= 1 (received {num_realizations}).")

        self.N = int(target_nodes) if target_nodes is not None else N
        self.R_rt = R_rt
        self.num_realizations = num_realizations
        self.loop_mechanism = loop_mechanism
        self.sample_parameters = sample_parameters
        self.base_seed = int(base_seed) if base_seed is not None else None

        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = DGMConfig.load_default()

        self.config.N = self.N
        self.config.R_rt = self.R_rt
        self.config.loop_mechanism = self.loop_mechanism
        self.config.sample_parameters = self.sample_parameters

        if target_nodes is not None:
            self.config.N = int(target_nodes)
            self.config.distributions.pop('N', None)

        if overrides:
            for k, v in overrides.items():
                if k == 'target_nodes':
                    self.N = int(v)
                    self.config.N = int(v)
                    self.config.distributions.pop('N', None)
                elif hasattr(self.config, k):
                    setattr(self.config, k, v)
                self.config.distributions.pop(k, None)

        self.config.validate()
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
    enable_curved_centerlines: Optional[bool] = None,
    min_channel_clearance: Optional[float] = None,
    enable_pruning: Optional[bool] = None,
    prune_inland_dead_ends: Optional[bool] = None,
    min_spur_length: Optional[float] = None,
    simplify_pass_through_nodes: Optional[bool] = None,
    return_stages: bool = False,
    config: Optional[Union[str, Path, DGMConfig]] = None,
    seed: Any = _UNSET,
    target_nodes: Optional[int] = None,
    **overrides
) -> Union[nx.DiGraph, Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]]:
    """
    Convenient top-level function to generate a single synthetic delta graph.
    """
    if target_nodes is not None:
        N = int(target_nodes)

    if isinstance(config, (str, Path)):
        cfg = DGMConfig.from_yaml(config, N=N, R_rt=R_rt, **overrides)
    elif isinstance(config, DGMConfig):
        cfg = copy.deepcopy(config)
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if k == 'target_nodes':
                cfg.N = int(v)
                cfg.distributions.pop('N', None)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
            cfg.distributions.pop(k, None)
    else:
        cfg = DGMConfig.load_default()
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if k == 'target_nodes':
                cfg.N = int(v)
                cfg.distributions.pop('N', None)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
            cfg.distributions.pop(k, None)

    if target_nodes is not None:
        cfg.N = int(target_nodes)
        cfg.distributions.pop('N', None)

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

    if enable_curved_centerlines is not None:
        cfg.enable_curved_centerlines = bool(enable_curved_centerlines)

    if min_channel_clearance is not None:
        cfg.min_channel_clearance = float(min_channel_clearance)
        cfg.distributions.pop("min_channel_clearance", None)

    if enable_pruning is not None:
        cfg.enable_pruning = bool(enable_pruning)

    if prune_inland_dead_ends is not None:
        cfg.prune_inland_dead_ends = bool(prune_inland_dead_ends)

    if min_spur_length is not None:
        cfg.min_spur_length = float(min_spur_length)
        cfg.distributions.pop("min_spur_length", None)

    if simplify_pass_through_nodes is not None:
        cfg.simplify_pass_through_nodes = bool(simplify_pass_through_nodes)

    if seed is not _UNSET:
        cfg.seed = seed

    gen = DeltaGenerator(cfg)
    return gen.generate(return_stages=return_stages)


def sample_ensemble(
    N: int = 50,
    R_rt: float = 0.5,
    confinement: Optional[float] = None,
    lateral_valley_width: Optional[float] = None,
    coastal_node_decay: Optional[float] = None,
    coastal_sinuosity_decay: Optional[float] = None,
    enable_curved_centerlines: Optional[bool] = None,
    min_channel_clearance: Optional[float] = None,
    enable_pruning: Optional[bool] = None,
    prune_inland_dead_ends: Optional[bool] = None,
    min_spur_length: Optional[float] = None,
    simplify_pass_through_nodes: Optional[bool] = None,
    num_realizations: int = 50,
    config: Optional[Union[str, Path, DGMConfig]] = None,
    base_seed: int = 42,
    target_nodes: Optional[int] = None,
    **overrides
) -> List[nx.DiGraph]:
    """
    Convenient top-level function to generate an ensemble of synthetic delta graphs.
    """
    if target_nodes is not None:
        N = int(target_nodes)

    if isinstance(config, (str, Path)):
        cfg = DGMConfig.from_yaml(config, N=N, R_rt=R_rt, **overrides)
    elif isinstance(config, DGMConfig):
        cfg = copy.deepcopy(config)
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if k == 'target_nodes':
                cfg.N = int(v)
                cfg.distributions.pop('N', None)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
            cfg.distributions.pop(k, None)
    else:
        cfg = DGMConfig.load_default()
        cfg.N = N
        cfg.R_rt = R_rt
        for k, v in overrides.items():
            if k == 'target_nodes':
                cfg.N = int(v)
                cfg.distributions.pop('N', None)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
            cfg.distributions.pop(k, None)

    if target_nodes is not None:
        cfg.N = int(target_nodes)
        cfg.distributions.pop('N', None)

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

    if enable_curved_centerlines is not None:
        cfg.enable_curved_centerlines = bool(enable_curved_centerlines)

    if min_channel_clearance is not None:
        cfg.min_channel_clearance = float(min_channel_clearance)
        cfg.distributions.pop("min_channel_clearance", None)

    if enable_pruning is not None:
        cfg.enable_pruning = bool(enable_pruning)

    if prune_inland_dead_ends is not None:
        cfg.prune_inland_dead_ends = bool(prune_inland_dead_ends)

    if min_spur_length is not None:
        cfg.min_spur_length = float(min_spur_length)
        cfg.distributions.pop("min_spur_length", None)

    if simplify_pass_through_nodes is not None:
        cfg.simplify_pass_through_nodes = bool(simplify_pass_through_nodes)

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
