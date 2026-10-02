"""
Configuration and distribution sampling engine for DeltaGraphModel (DGM).
Provides YAML-driven configuration, stochastic parameter distributions,
and extensibility hooks.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import copy
import yaml
import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default_config.yaml"


def sample_from_spec(spec: Dict[str, Any], rng: np.random.Generator) -> float:
    """Samples a scalar value from a distribution specification dict."""
    dist_type = spec.get("type", "uniform").lower()

    if dist_type == "uniform":
        low = float(spec.get("low", 0.0))
        high = float(spec.get("high", 1.0))
        return float(rng.uniform(low, high))

    elif dist_type in ("normal", "gaussian"):
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        val = rng.normal(mean, std)
        if "min" in spec:
            val = max(float(spec["min"]), val)
        if "max" in spec:
            val = min(float(spec["max"]), val)
        return float(val)

    elif dist_type == "lognormal":
        mean = float(spec.get("mean", 0.0))
        sigma = float(spec.get("sigma", 1.0))
        val = rng.lognormal(mean, sigma)
        if "min" in spec:
            val = max(float(spec["min"]), val)
        if "max" in spec:
            val = min(float(spec["max"]), val)
        return float(val)

    elif dist_type in ("choice", "categorical"):
        values = spec.get("values", [1.0])
        weights = spec.get("weights", None)
        if weights is not None:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
        return float(rng.choice(values, p=weights))

    elif dist_type in ("fixed", "constant"):
        return float(spec.get("value", 0.0))

    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


@dataclass
class ParameterBounds:
    """
    Physical and hydraulic parameter bounds for uniform/stochastic sampling.
    Maintained for backwards-compatibility with synthetic_graphs.v2.
    """
    murray_p_range: Tuple[float, float] = (1.2, 2.2)
    width_beta_range: Tuple[float, float] = (0.50, 0.99)
    manning_eta_range: Tuple[float, float] = (1.50, 1.80)
    length_gamma_range: Tuple[float, float] = (0.40, 0.60)
    apex_width_range: Tuple[float, float] = (50.0, 500.0)
    r_shore_range: Tuple[float, float] = (5000.0, 15000.0)
    theta_base_range: Tuple[float, float] = (30.0, 60.0)
    theta_delta_range: Tuple[float, float] = (45.0, 90.0)
    width_noise_range: Tuple[float, float] = (0.15, 0.50)
    candidate_dist_scale_range: Tuple[float, float] = (0.15, 0.35)
    tidal_flare_k_range: Tuple[float, float] = (0.8, 2.0)
    confinement_range: Tuple[float, float] = (0.0, 0.60)
    coastal_node_decay_range: Tuple[float, float] = (0.0, 0.80)
    coastal_sinuosity_decay_range: Tuple[float, float] = (0.0, 0.90)
    min_channel_clearance_range: Tuple[float, float] = (0.05, 0.80)
    min_spur_length_range: Tuple[float, float] = (0.0, 500.0)

    def sample(self, rng: np.random.Generator) -> Dict[str, float]:
        """Samples a single parameter set from uniform distributions within bounds."""
        return {
            'murray_p': float(rng.uniform(*self.murray_p_range)),
            'width_beta': float(rng.uniform(*self.width_beta_range)),
            'manning_eta': float(rng.uniform(*self.manning_eta_range)),
            'length_gamma': float(rng.uniform(*self.length_gamma_range)),
            'apex_width': float(rng.uniform(*self.apex_width_range)),
            'r_shore': float(rng.uniform(*self.r_shore_range)),
            'theta_base': float(rng.uniform(*self.theta_base_range)),
            'theta_delta': float(rng.uniform(*self.theta_delta_range)),
            'width_noise': float(rng.uniform(*self.width_noise_range)),
            'candidate_dist_scale': float(rng.uniform(*self.candidate_dist_scale_range)),
            'tidal_flare_k': float(rng.uniform(*self.tidal_flare_k_range)),
            'confinement': float(rng.uniform(*self.confinement_range)),
            'coastal_node_decay': float(rng.uniform(*self.coastal_node_decay_range)),
            'coastal_sinuosity_decay': float(rng.uniform(*self.coastal_sinuosity_decay_range)),
            'min_channel_clearance': float(rng.uniform(*self.min_channel_clearance_range)),
            'min_spur_length': float(rng.uniform(*self.min_spur_length_range)),
        }

    def default_values(self) -> Dict[str, float]:
        """Returns deterministic mid-point values for standard baseline runs."""
        return {
            'murray_p': float(np.mean(self.murray_p_range)),
            'width_beta': float(np.mean(self.width_beta_range)),
            'manning_eta': float(np.mean(self.manning_eta_range)),
            'length_gamma': float(np.mean(self.length_gamma_range)),
            'apex_width': float(np.mean(self.apex_width_range)),
            'r_shore': float(np.mean(self.r_shore_range)),
            'theta_base': float(np.mean(self.theta_base_range)),
            'theta_delta': float(np.mean(self.theta_delta_range)),
            'width_noise': float(np.mean(self.width_noise_range)),
            'candidate_dist_scale': float(np.mean(self.candidate_dist_scale_range)),
            'tidal_flare_k': float(np.mean(self.tidal_flare_k_range)),
            'confinement': float(np.mean(self.confinement_range)),
            'coastal_node_decay': float(np.mean(self.coastal_node_decay_range)),
            'coastal_sinuosity_decay': float(np.mean(self.coastal_sinuosity_decay_range)),
            'min_channel_clearance': float(np.mean(self.min_channel_clearance_range)),
            'min_spur_length': float(np.mean(self.min_spur_length_range)),
        }


@dataclass
class DGMConfig:
    """
    Primary configuration container for DeltaGraphModel (DGM).
    Supports loading from/saving to YAML, accessing baseline parameters,
    and sampling parameter clouds for ensemble generation.
    """
    # Model section
    N: int = 50
    R_rt: float = 0.5
    loop_mechanism: str = "potential_field"
    sample_parameters: bool = False
    seed: Optional[int] = 42

    # Domain section
    r_shore: float = 10000.0
    theta_base: float = 45.0
    theta_delta: float = 60.0
    confinement: float = 0.0
    lateral_valley_width: Optional[float] = None

    # Backbone section
    branching_weights: Tuple[float, float, float] = (0.50, 0.45, 0.05)
    radial_step_scale_min: float = 0.65
    radial_step_scale_max: float = 1.35
    coastal_node_decay: float = 0.40
    angular_spread_decay: float = 0.70
    single_branch_spread_factor: float = 0.35
    bifurcation_spread_min: float = 0.60
    bifurcation_spread_max: float = 1.40
    base_sinuosity_mean: float = 0.25
    base_sinuosity_std: float = 0.15
    coastal_sinuosity_decay: float = 0.30

    # Loops section
    candidate_dist_scale: float = 0.25
    reconnection_prob_base: float = 0.55
    anastomosis_prob_base: float = 0.0
    max_in_degree: int = 2
    max_out_degree: int = 3
    target_loop_fraction_slope: float = 0.25
    target_loop_fraction_intercept: float = 0.08
    target_max_loops_slope: float = 0.22
    target_max_loops_intercept: float = 0.05
    sa_initial_temp: float = 1.0
    sa_cooling_rate: float = 0.975
    sa_max_steps: int = 250

    # Pruning & topological simplification section
    enable_pruning: bool = False
    prune_inland_dead_ends: bool = True
    min_spur_length: float = 0.0
    simplify_pass_through_nodes: bool = False
    max_prune_iterations: int = 5

    # Hydraulics section
    routing_mode: str = "width"
    murray_p: float = 1.70
    width_beta: float = 0.745
    manning_eta: float = 1.65
    length_gamma: float = 0.50
    apex_width: float = 250.0
    width_noise: float = 0.30
    tidal_flare_k: float = 1.40
    min_channel_width: float = 0.1
    relaxation_max_iter: int = 15
    relaxation_tol: float = 1e-4

    # Channel geometry & proximity section
    enable_curved_centerlines: bool = False
    min_channel_clearance: float = 0.40
    num_polyline_points: int = 21
    meander_bends: int = 1

    # Distributions & Extensions
    distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validates configuration parameters and invariants."""
        if self.N < 4:
            raise ValueError(f"Node budget N must be >= 4 (received {self.N}).")
        if not (0.0 <= self.R_rt <= 1.0):
            raise ValueError(f"River-to-Tidal forcing ratio R_rt must be in [0.0, 1.0] (received {self.R_rt}).")
        if self.loop_mechanism not in ("potential_field", "ner_optimality"):
            raise ValueError(
                f"loop_mechanism must be 'potential_field' or 'ner_optimality' (received '{self.loop_mechanism}')."
            )
        if not (0.0 < self.sa_cooling_rate < 1.0):
            raise ValueError(f"sa_cooling_rate must be in (0, 1) (received {self.sa_cooling_rate}).")
        if self.sa_max_steps <= 0:
            raise ValueError("sa_max_steps must be > 0.")
        if self.sa_initial_temp <= 0:
            raise ValueError("sa_initial_temp must be > 0.")
        if len(self.branching_weights) != 3 or any(w < 0 for w in self.branching_weights):
            raise ValueError("branching_weights must be a 3-tuple of non-negative probabilities.")
        if abs(sum(self.branching_weights) - 1.0) > 0.01:
            raise ValueError(
                f"branching_weights must sum to approximately 1.0 within 0.01 (received {sum(self.branching_weights)})."
            )
        if self.routing_mode not in ("width", "conductance"):
            raise ValueError(
                f"routing_mode must be 'width' or 'conductance' (received '{self.routing_mode}')."
            )
        if self.r_shore <= 0:
            raise ValueError("r_shore must be > 0.")
        if self.murray_p <= 0:
            raise ValueError("murray_p must be > 0.")
        if not (0.0 <= self.confinement <= 0.95):
            raise ValueError(f"confinement parameter must be in [0.0, 0.95] (received {self.confinement}).")
        if self.lateral_valley_width is not None and self.lateral_valley_width <= 0:
            raise ValueError(f"lateral_valley_width must be > 0 (received {self.lateral_valley_width}).")
        if not (0.0 <= self.coastal_node_decay <= 0.80):
            raise ValueError(f"coastal_node_decay must be in [0.0, 0.80] (received {self.coastal_node_decay}).")
        if not (0.0 <= self.coastal_sinuosity_decay <= 0.90):
            raise ValueError(f"coastal_sinuosity_decay must be in [0.0, 0.90] (received {self.coastal_sinuosity_decay}).")
        if self.base_sinuosity_mean < 0.0:
            raise ValueError(f"base_sinuosity_mean must be >= 0.0 (received {self.base_sinuosity_mean}).")
        if self.base_sinuosity_std <= 0.0:
            raise ValueError(f"base_sinuosity_std must be > 0.0 (received {self.base_sinuosity_std}).")
        if not (0.05 <= self.min_channel_clearance <= 1.0):
            raise ValueError(f"min_channel_clearance must be in [0.05, 1.0] (received {self.min_channel_clearance}).")
        if self.num_polyline_points < 3:
            raise ValueError(f"num_polyline_points must be >= 3 (received {self.num_polyline_points}).")
        if self.meander_bends < 1:
            raise ValueError(f"meander_bends must be >= 1 (received {self.meander_bends}).")
        if self.min_spur_length < 0.0:
            raise ValueError(f"min_spur_length must be >= 0.0 (received {self.min_spur_length}).")
        if self.max_prune_iterations < 1:
            raise ValueError(f"max_prune_iterations must be >= 1 (received {self.max_prune_iterations}).")
        if "coastal_node_decay" in self.distributions:
            d = self.distributions["coastal_node_decay"]
            if isinstance(d, dict) and d.get("type") == "uniform":
                if d.get("low", 0.0) < 0.0 or d.get("high", 0.80) > 0.80:
                    raise ValueError("coastal_node_decay distribution must be within [0.0, 0.80].")
        if "coastal_sinuosity_decay" in self.distributions:
            d = self.distributions["coastal_sinuosity_decay"]
            if isinstance(d, dict) and d.get("type") == "uniform":
                if d.get("low", 0.0) < 0.0 or d.get("high", 0.90) > 0.90:
                    raise ValueError("coastal_sinuosity_decay distribution must be within [0.0, 0.90].")
        if "min_channel_clearance" in self.distributions:
            d = self.distributions["min_channel_clearance"]
            if isinstance(d, dict) and d.get("type") == "uniform":
                if d.get("low", 0.05) < 0.05 or d.get("high", 1.0) > 1.0:
                    raise ValueError("min_channel_clearance distribution must be within [0.05, 1.0].")
        if "min_spur_length" in self.distributions:
            d = self.distributions["min_spur_length"]
            if isinstance(d, dict) and d.get("type") == "uniform":
                if d.get("low", 0.0) < 0.0:
                    raise ValueError("min_spur_length distribution low bound must be >= 0.0.")

    @classmethod
    def load_default(cls, **overrides) -> "DGMConfig":
        """Loads configuration from the packaged default_config.yaml with optional overrides."""
        if DEFAULT_CONFIG_PATH.exists():
            return cls.from_yaml(DEFAULT_CONFIG_PATH, **overrides)
        return cls(**overrides)

    @classmethod
    def from_yaml(cls, path: Union[str, Path], **overrides) -> "DGMConfig":
        """
        Loads configuration from a YAML file with optional keyword parameter overrides.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Flatten sections into DGMConfig fields
        model_sec = raw.get("model", {})
        domain_sec = raw.get("domain", {})
        backbone_sec = raw.get("backbone", {})
        loops_sec = raw.get("loops", {})
        pruning_sec = raw.get("pruning", {})
        hydraulics_sec = raw.get("hydraulics", {})
        channel_geom_sec = raw.get("channel_geometry", {})
        dists = raw.get("distributions", {})
        exts = raw.get("extensions", {})

        kwargs: Dict[str, Any] = {}
        kwargs.update(model_sec)
        kwargs.update(domain_sec)
        kwargs.update(backbone_sec)
        kwargs.update(loops_sec)
        kwargs.update(pruning_sec)
        kwargs.update(hydraulics_sec)
        kwargs.update(channel_geom_sec)
        kwargs["distributions"] = dists
        kwargs["extensions"] = exts

        # Convert branching_weights to tuple if list
        if "branching_weights" in kwargs and isinstance(kwargs["branching_weights"], list):
            kwargs["branching_weights"] = tuple(kwargs["branching_weights"])

        # Apply direct user overrides
        kwargs.update(overrides)

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_kwargs: Dict[str, Any] = {}
        extensions = dict(kwargs.get("extensions", {}))
        for k, v in kwargs.items():
            if k in valid_fields:
                filtered_kwargs[k] = v
            else:
                extensions[k] = v
        filtered_kwargs["extensions"] = extensions

        return cls(**filtered_kwargs)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serializes current configuration back to a structured YAML file."""
        data = {
            "model": {
                "N": self.N,
                "R_rt": self.R_rt,
                "loop_mechanism": self.loop_mechanism,
                "sample_parameters": self.sample_parameters,
                "seed": self.seed,
            },
            "domain": {
                "r_shore": self.r_shore,
                "theta_base": self.theta_base,
                "theta_delta": self.theta_delta,
                "confinement": self.confinement,
                "lateral_valley_width": self.lateral_valley_width,
            },
            "backbone": {
                "branching_weights": list(self.branching_weights),
                "radial_step_scale_min": self.radial_step_scale_min,
                "radial_step_scale_max": self.radial_step_scale_max,
                "coastal_node_decay": self.coastal_node_decay,
                "angular_spread_decay": self.angular_spread_decay,
                "single_branch_spread_factor": self.single_branch_spread_factor,
                "bifurcation_spread_min": self.bifurcation_spread_min,
                "bifurcation_spread_max": self.bifurcation_spread_max,
                "base_sinuosity_mean": self.base_sinuosity_mean,
                "base_sinuosity_std": self.base_sinuosity_std,
                "coastal_sinuosity_decay": self.coastal_sinuosity_decay,
            },
            "loops": {
                "candidate_dist_scale": self.candidate_dist_scale,
                "reconnection_prob_base": self.reconnection_prob_base,
                "anastomosis_prob_base": self.anastomosis_prob_base,
                "max_in_degree": self.max_in_degree,
                "max_out_degree": self.max_out_degree,
                "target_loop_fraction_slope": self.target_loop_fraction_slope,
                "target_loop_fraction_intercept": self.target_loop_fraction_intercept,
                "target_max_loops_slope": self.target_max_loops_slope,
                "target_max_loops_intercept": self.target_max_loops_intercept,
                "sa_initial_temp": self.sa_initial_temp,
                "sa_cooling_rate": self.sa_cooling_rate,
                "sa_max_steps": self.sa_max_steps,
            },
            "pruning": {
                "enable_pruning": self.enable_pruning,
                "prune_inland_dead_ends": self.prune_inland_dead_ends,
                "min_spur_length": self.min_spur_length,
                "simplify_pass_through_nodes": self.simplify_pass_through_nodes,
                "max_prune_iterations": self.max_prune_iterations,
            },
            "hydraulics": {
                "routing_mode": self.routing_mode,
                "murray_p": self.murray_p,
                "width_beta": self.width_beta,
                "manning_eta": self.manning_eta,
                "length_gamma": self.length_gamma,
                "apex_width": self.apex_width,
                "width_noise": self.width_noise,
                "tidal_flare_k": self.tidal_flare_k,
                "min_channel_width": self.min_channel_width,
                "relaxation_max_iter": self.relaxation_max_iter,
                "relaxation_tol": self.relaxation_tol,
            },
            "channel_geometry": {
                "enable_curved_centerlines": self.enable_curved_centerlines,
                "min_channel_clearance": self.min_channel_clearance,
                "num_polyline_points": self.num_polyline_points,
                "meander_bends": self.meander_bends,
            },
            "distributions": copy.deepcopy(self.distributions),
            "extensions": copy.deepcopy(self.extensions),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @property
    def bounds(self) -> ParameterBounds:
        """Returns a ParameterBounds instance for backwards-compatibility."""
        return ParameterBounds()

    def sample_distribution_parameters(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Samples active parameters from configured distributions.
        For parameters not present in distributions, falls back to the instance baseline.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        params = self.get_active_parameters(sample=False)

        for param_name, dist_spec in self.distributions.items():
            if isinstance(dist_spec, dict) and "type" in dist_spec:
                params[param_name] = sample_from_spec(dist_spec, rng)

        return params

    def get_active_parameters(
        self,
        rng: Optional[np.random.Generator] = None,
        sample: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Returns the dictionary of active parameters for the generative pipeline.
        If sample=True (or sample is None and self.sample_parameters is True),
        stochastically samples from distributions.
        """
        do_sample = self.sample_parameters if sample is None else sample
        if do_sample:
            return self.sample_distribution_parameters(rng)

        # Baseline deterministic parameters
        params = {
            "N": self.N,
            "R_rt": self.R_rt,
            "loop_mechanism": self.loop_mechanism,
            "r_shore": self.r_shore,
            "theta_base": self.theta_base,
            "theta_delta": self.theta_delta,
            "confinement": self.confinement,
            "lateral_valley_width": self.lateral_valley_width,
            "branching_weights": self.branching_weights,
            "radial_step_scale_min": self.radial_step_scale_min,
            "radial_step_scale_max": self.radial_step_scale_max,
            "coastal_node_decay": self.coastal_node_decay,
            "angular_spread_decay": self.angular_spread_decay,
            "single_branch_spread_factor": self.single_branch_spread_factor,
            "bifurcation_spread_min": self.bifurcation_spread_min,
            "bifurcation_spread_max": self.bifurcation_spread_max,
            "base_sinuosity_mean": self.base_sinuosity_mean,
            "base_sinuosity_std": self.base_sinuosity_std,
            "coastal_sinuosity_decay": self.coastal_sinuosity_decay,
            "candidate_dist_scale": self.candidate_dist_scale,
            "reconnection_prob_base": self.reconnection_prob_base,
            "anastomosis_prob_base": self.anastomosis_prob_base,
            "max_in_degree": self.max_in_degree,
            "max_out_degree": self.max_out_degree,
            "target_loop_fraction_slope": self.target_loop_fraction_slope,
            "target_loop_fraction_intercept": self.target_loop_fraction_intercept,
            "target_max_loops_slope": self.target_max_loops_slope,
            "target_max_loops_intercept": self.target_max_loops_intercept,
            "sa_initial_temp": self.sa_initial_temp,
            "sa_cooling_rate": self.sa_cooling_rate,
            "sa_max_steps": self.sa_max_steps,
            "enable_pruning": self.enable_pruning,
            "prune_inland_dead_ends": self.prune_inland_dead_ends,
            "min_spur_length": self.min_spur_length,
            "simplify_pass_through_nodes": self.simplify_pass_through_nodes,
            "max_prune_iterations": self.max_prune_iterations,
            "routing_mode": self.routing_mode,
            "murray_p": self.murray_p,
            "width_beta": self.width_beta,
            "manning_eta": self.manning_eta,
            "length_gamma": self.length_gamma,
            "apex_width": self.apex_width,
            "width_noise": self.width_noise,
            "tidal_flare_k": self.tidal_flare_k,
            "min_channel_width": self.min_channel_width,
            "relaxation_max_iter": self.relaxation_max_iter,
            "relaxation_tol": self.relaxation_tol,
            "enable_curved_centerlines": self.enable_curved_centerlines,
            "min_channel_clearance": self.min_channel_clearance,
            "num_polyline_points": self.num_polyline_points,
            "meander_bends": self.meander_bends,
            "extensions": copy.deepcopy(self.extensions),
        }
        return params
