"""
Unified Facade Generator module for DeltaGraphModel (DGM).
Provides:
  - DeltaGenerator: single realization generator
  - EnsembleGenerator: stochastic cloud / ensemble sampling generator
  - generate(): convenient top-level function for single graphs
  - sample_ensemble(): convenient top-level function for ensembles
"""

import contextlib
import copy
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .config import DGMConfig
from .core.backbone import generate_polar_tree_backbone
from .core.channel_geometry import (
    apply_channel_proximity_constraints,
    generate_channel_centerlines,
)
from .core.domain import PolarDomain
from .core.hydraulics import apply_emergent_hydraulics
from .core.loops import apply_loop_mechanism
from .core.pruning import prune_and_simplify_network
from .metrics import MetricsBattery

_UNSET = object()


def _build_or_update_config(
    config: str | Path | DGMConfig | None = None,
    target_nodes: int | None = None,
    confinement: float | None = None,
    lateral_valley_width: float | None = None,
    coastal_node_decay: float | None = None,
    coastal_sinuosity_decay: float | None = None,
    enable_curved_centerlines: bool | None = None,
    min_channel_clearance: float | None = None,
    enable_pruning: bool | None = None,
    prune_inland_dead_ends: bool | None = None,
    min_spur_length: float | None = None,
    simplify_pass_through_nodes: bool | None = None,
    seed: Any = _UNSET,
    base_attrs: dict[str, Any] | None = None,
    **overrides: Any,
) -> DGMConfig:
    """
    Initializes or deep-copies a DGMConfig instance, applies baseline attributes and
    explicit keyword overrides (removing overridden keys from stochastic distributions),
    and validates the resulting configuration.
    """
    if isinstance(config, (str, Path)):
        cfg = DGMConfig.from_yaml(config)
    elif isinstance(config, DGMConfig):
        cfg = copy.deepcopy(config)
    elif config is None:
        cfg = DGMConfig.load_default()
    else:
        raise TypeError(f"Unsupported config type: {type(config).__name__}")

    if base_attrs:
        for k, v in base_attrs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    for k, v in overrides.items():
        if k == "target_nodes":
            if v is not None:
                cfg.N = int(v)
                cfg.distributions.pop("N", None)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
            cfg.distributions.pop(k, None)
        else:
            cfg.extensions[k] = v
            cfg.distributions.pop(k, None)

    if target_nodes is not None:
        cfg.N = int(target_nodes)
        cfg.distributions.pop("N", None)

    float_opts = {
        "confinement": confinement,
        "lateral_valley_width": lateral_valley_width,
        "coastal_node_decay": coastal_node_decay,
        "coastal_sinuosity_decay": coastal_sinuosity_decay,
        "min_channel_clearance": min_channel_clearance,
        "min_spur_length": min_spur_length,
    }
    for k, v in float_opts.items():
        if v is not None:
            setattr(cfg, k, float(v))
            cfg.distributions.pop(k, None)

    bool_opts = {
        "enable_curved_centerlines": enable_curved_centerlines,
        "enable_pruning": enable_pruning,
        "prune_inland_dead_ends": prune_inland_dead_ends,
        "simplify_pass_through_nodes": simplify_pass_through_nodes,
    }
    for k, v in bool_opts.items():
        if v is not None:
            setattr(cfg, k, bool(v))
            cfg.distributions.pop(k, None)

    if seed is not _UNSET:
        cfg.seed = seed

    cfg.validate()
    return cfg


class DeltaGenerator:
    """
    Primary generator for single synthetic delta graph realizations.
    """

    def __init__(
        self,
        config: DGMConfig | None = None,
        target_nodes: int | None = None,
        **overrides
    ):
        self.config = _build_or_update_config(
            config=config,
            target_nodes=target_nodes,
            **overrides
        )
        self.battery = MetricsBattery()

    def generate(
        self,
        seed_override: Any = _UNSET,
        return_stages: bool = False
    ) -> nx.DiGraph | tuple[nx.DiGraph, dict[str, nx.DiGraph]]:
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
            with contextlib.suppress(ValueError, TypeError):
                seed = int(seed)
        rng = np.random.default_rng(seed)
        sub_seed = int(rng.integers(0, 2**31 - 1)) if seed is None else int(seed)

        # Get parameter set (sampled from distributions or deterministic defaults)
        params = self.config.get_active_parameters(rng)
        active_N = int(params.get('N', self.config.N))
        active_R_rt = float(params.get('R_rt', self.config.R_rt))
        active_mech = str(params.get('loop_mechanism', self.config.loop_mechanism))
        params['N'] = active_N
        params['N_target'] = active_N
        params['R_rt'] = active_R_rt
        params['loop_mechanism'] = active_mech

        # 1. Initialize Polar Domain
        try:
            domain = PolarDomain(
                r_shore=params['r_shore'],
                theta_base=params['theta_base'],
                theta_delta=params['theta_delta'],
                R_rt=active_R_rt,
                confinement=params.get('confinement', self.config.confinement),
                lateral_valley_width=params.get('lateral_valley_width', self.config.lateral_valley_width)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PolarDomain: {e}") from e

        # 2. Generate Polar Growth Tree Backbone
        try:
            G_tree = generate_polar_tree_backbone(
                N_target=active_N,
                R_rt=active_R_rt,
                domain=domain,
                params=params,
                rng=rng
            )
        except Exception as e:
            raise RuntimeError(
                f"DGM polar tree backbone generation failed for N={active_N}, R_rt={active_R_rt}: {e}"
            ) from e
        G_tree_snapshot = G_tree.copy()

        # 3. Apply Emergent Loop Mechanism (Mechanism A or Mechanism B)
        try:
            G_looped = apply_loop_mechanism(
                G=G_tree,
                domain=domain,
                mechanism=active_mech,
                params=params,
                rng=rng,
                seed=sub_seed
            )
        except Exception as e:
            raise RuntimeError(
                f"DGM loop mechanism '{active_mech}' failed: {e}"
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
            if return_stages:
                # Synthesize centerlines for the tree backbone snapshot when stages are requested
                with contextlib.suppress(Exception):
                    generate_channel_centerlines(
                        G=G_tree_snapshot,
                        clearance_fraction=clr_frac,
                        num_points=pts_num,
                        num_bends=bends_num,
                        seed=sub_seed
                    )
        else:
            G_final = apply_channel_proximity_constraints(
                G=G_final,
                clearance_fraction=clr_frac
            )

        # Attach run metadata to graph
        G_final.graph['N_input'] = active_N
        G_final.graph['R_rt'] = active_R_rt
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
        G_final.graph['loop_mechanism'] = active_mech
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

    def evaluate_metrics(self, G: nx.DiGraph) -> dict[str, Any]:
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
        config: DGMConfig | None = None,
        target_nodes: int | None = None,
        **overrides
    ):
        if num_realizations < 1:
            raise ValueError(f"num_realizations must be >= 1 (received {num_realizations}).")

        eff_target = overrides.get("target_nodes", target_nodes)
        self.N = int(eff_target) if eff_target is not None else N
        self.R_rt = R_rt
        self.num_realizations = num_realizations
        self.loop_mechanism = loop_mechanism
        self.sample_parameters = sample_parameters
        self.base_seed = int(base_seed) if base_seed is not None else None

        self.config = _build_or_update_config(
            config=config,
            target_nodes=target_nodes,
            base_attrs={
                "N": self.N,
                "R_rt": self.R_rt,
                "loop_mechanism": self.loop_mechanism,
                "sample_parameters": self.sample_parameters,
            },
            **overrides
        )
        self.battery = MetricsBattery()

    def generate_ensemble(self) -> list[nx.DiGraph]:
        """Generates an array of M graph realizations."""
        if self.base_seed is None:
            raise ValueError('base_seed must be an integer, got None')

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

    def evaluate_ensemble(self) -> dict[str, Any]:
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
    confinement: float | None = None,
    lateral_valley_width: float | None = None,
    coastal_node_decay: float | None = None,
    coastal_sinuosity_decay: float | None = None,
    enable_curved_centerlines: bool | None = None,
    min_channel_clearance: float | None = None,
    enable_pruning: bool | None = None,
    prune_inland_dead_ends: bool | None = None,
    min_spur_length: float | None = None,
    simplify_pass_through_nodes: bool | None = None,
    return_stages: bool = False,
    config: str | Path | DGMConfig | None = None,
    seed: Any = _UNSET,
    target_nodes: int | None = None,
    **overrides
) -> nx.DiGraph | tuple[nx.DiGraph, dict[str, nx.DiGraph]]:
    """
    Convenient top-level function to generate a single synthetic delta graph.
    """
    if target_nodes is not None:
        N = int(target_nodes)

    cfg = _build_or_update_config(
        config=config,
        target_nodes=target_nodes,
        confinement=confinement,
        lateral_valley_width=lateral_valley_width,
        coastal_node_decay=coastal_node_decay,
        coastal_sinuosity_decay=coastal_sinuosity_decay,
        enable_curved_centerlines=enable_curved_centerlines,
        min_channel_clearance=min_channel_clearance,
        enable_pruning=enable_pruning,
        prune_inland_dead_ends=prune_inland_dead_ends,
        min_spur_length=min_spur_length,
        simplify_pass_through_nodes=simplify_pass_through_nodes,
        seed=seed,
        base_attrs={"N": N, "R_rt": R_rt},
        **overrides
    )

    gen = DeltaGenerator(cfg)
    return gen.generate(return_stages=return_stages)


def sample_ensemble(
    N: int = 50,
    R_rt: float = 0.5,
    confinement: float | None = None,
    lateral_valley_width: float | None = None,
    coastal_node_decay: float | None = None,
    coastal_sinuosity_decay: float | None = None,
    enable_curved_centerlines: bool | None = None,
    min_channel_clearance: float | None = None,
    enable_pruning: bool | None = None,
    prune_inland_dead_ends: bool | None = None,
    min_spur_length: float | None = None,
    simplify_pass_through_nodes: bool | None = None,
    num_realizations: int = 50,
    config: str | Path | DGMConfig | None = None,
    base_seed: int = 42,
    target_nodes: int | None = None,
    **overrides
) -> list[nx.DiGraph]:
    """
    Convenient top-level function to generate an ensemble of synthetic delta graphs.
    """
    if target_nodes is not None:
        N = int(target_nodes)

    cfg = _build_or_update_config(
        config=config,
        target_nodes=target_nodes,
        confinement=confinement,
        lateral_valley_width=lateral_valley_width,
        coastal_node_decay=coastal_node_decay,
        coastal_sinuosity_decay=coastal_sinuosity_decay,
        enable_curved_centerlines=enable_curved_centerlines,
        min_channel_clearance=min_channel_clearance,
        enable_pruning=enable_pruning,
        prune_inland_dead_ends=prune_inland_dead_ends,
        min_spur_length=min_spur_length,
        simplify_pass_through_nodes=simplify_pass_through_nodes,
        base_attrs={"N": N, "R_rt": R_rt},
        **overrides
    )

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
