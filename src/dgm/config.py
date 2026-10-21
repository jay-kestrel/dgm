"""
Configuration and distribution sampling engine for DeltaGraphModel (DGM).
Provides YAML-driven configuration, stochastic parameter distributions,
and extensibility hooks.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PKG_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"
PKG_SUBDIR_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default_config.yaml"
REPO_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default_config.yaml"


def _resolve_default_config_path() -> Path:
    """Resolves the default YAML configuration path across package and repository layouts."""
    for candidate in (PKG_CONFIG_PATH, PKG_SUBDIR_CONFIG_PATH, REPO_CONFIG_PATH):
        if candidate.exists():
            return candidate
    return REPO_CONFIG_PATH


DEFAULT_CONFIG_PATH = _resolve_default_config_path()


def sample_from_spec(spec: dict[str, Any], rng: np.random.Generator) -> float:
    """Samples a scalar value from a distribution specification dict."""
    if not isinstance(spec, dict):
        raise TypeError(f"Distribution specification must be a dictionary, got {type(spec).__name__}.")

    dist_type = str(spec.get("type", "uniform")).lower()

    if dist_type == "uniform":
        low = float(spec.get("low", 0.0))
        high = float(spec.get("high", 1.0))
        if low > high:
            raise ValueError(f"Uniform distribution low bound ({low}) cannot exceed high bound ({high}).")
        val = float(rng.uniform(low, high))
        if "min" in spec:
            val = max(float(spec["min"]), val)
        if "max" in spec:
            val = min(float(spec["max"]), val)
        return float(val)

    elif dist_type in ("normal", "gaussian"):
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        if std <= 0 or not np.isfinite(std):
            raise ValueError(f"Normal distribution std must be > 0 (received {std}).")
        if "min" in spec and "max" in spec and float(spec["min"]) > float(spec["max"]):
            raise ValueError(f"Normal distribution min bound ({spec['min']}) cannot exceed max bound ({spec['max']}).")
        val = float(rng.normal(mean, std))
        if "min" in spec:
            val = max(float(spec["min"]), val)
        if "max" in spec:
            val = min(float(spec["max"]), val)
        if not np.isfinite(val):
            val = mean
        return float(val)

    elif dist_type == "lognormal":
        mean = float(spec.get("mean", 0.0))
        sigma = float(spec.get("sigma", 1.0))
        if sigma <= 0 or not np.isfinite(sigma):
            raise ValueError(f"Lognormal distribution sigma must be > 0 (received {sigma}).")
        if "min" in spec and "max" in spec and float(spec["min"]) > float(spec["max"]):
            raise ValueError(f"Lognormal distribution min bound ({spec['min']}) cannot exceed max bound ({spec['max']}).")
        val = float(rng.lognormal(mean, sigma))
        if "min" in spec:
            val = max(float(spec["min"]), val)
        if "max" in spec:
            val = min(float(spec["max"]), val)
        if not np.isfinite(val):
            val = float(np.exp(mean))
        return float(val)

    elif dist_type in ("choice", "categorical"):
        values = spec.get("values", [1.0])
        if not values:
            raise ValueError("Choice distribution values list cannot be empty.")
        weights = spec.get("weights")
        if weights is not None:
            weights = np.array(weights, dtype=float)
            if len(weights) != len(values):
                raise ValueError(
                    f"Distribution weights count ({len(weights)}) does not match values count ({len(values)})."
                )
            if weights.sum() <= 0 or (weights < 0).any() or not np.all(np.isfinite(weights)):
                raise ValueError("Distribution weights must be non-negative and sum to > 0.")
            weights = weights / weights.sum()
        chosen = rng.choice(values, p=weights)
        try:
            return float(chosen)
        except (ValueError, TypeError):
            return chosen

    elif dist_type in ("fixed", "constant"):
        val = spec.get("value", 0.0)
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


@dataclass
class ParameterBounds:
    """
    Physical and hydraulic parameter bounds for uniform/stochastic sampling.
    Maintained for backwards-compatibility with legacy parameter schema.
    """
    murray_p_range: tuple[float, float] = (1.2, 2.2)
    width_beta_range: tuple[float, float] = (0.50, 0.99)
    manning_eta_range: tuple[float, float] = (1.50, 1.80)
    length_gamma_range: tuple[float, float] = (0.40, 0.60)
    apex_width_range: tuple[float, float] = (50.0, 500.0)
    r_shore_range: tuple[float, float] = (5000.0, 15000.0)
    theta_base_range: tuple[float, float] = (50.0, 80.0)
    theta_delta_range: tuple[float, float] = (45.0, 90.0)
    width_noise_range: tuple[float, float] = (0.15, 0.50)
    candidate_dist_scale_range: tuple[float, float] = (0.15, 0.35)
    tidal_flare_k_range: tuple[float, float] = (0.8, 2.0)
    confinement_range: tuple[float, float] = (0.0, 0.50)
    coastal_node_decay_range: tuple[float, float] = (0.0, 0.80)
    coastal_sinuosity_decay_range: tuple[float, float] = (0.0, 0.90)
    min_channel_clearance_range: tuple[float, float] = (0.05, 0.80)
    min_spur_length_range: tuple[float, float] = (0.0, 500.0)
    tidal_flare_onset_range: tuple[float, float] = (0.10, 0.50)
    loop_flow_penalty_range: tuple[float, float] = (0.20, 0.70)
    outlet_tolerance_range: tuple[float, float] = (0.02, 0.10)

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
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
            'tidal_flare_onset': float(rng.uniform(*self.tidal_flare_onset_range)),
            'loop_flow_penalty': float(rng.uniform(*self.loop_flow_penalty_range)),
            'confinement': float(rng.uniform(*self.confinement_range)),
            'coastal_node_decay': float(rng.uniform(*self.coastal_node_decay_range)),
            'coastal_sinuosity_decay': float(rng.uniform(*self.coastal_sinuosity_decay_range)),
            'min_channel_clearance': float(rng.uniform(*self.min_channel_clearance_range)),
            'outlet_tolerance': float(rng.uniform(*self.outlet_tolerance_range)),
            'min_spur_length': float(rng.uniform(*self.min_spur_length_range)),
        }

    def default_values(self) -> dict[str, float]:
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
            'tidal_flare_onset': float(np.mean(self.tidal_flare_onset_range)),
            'loop_flow_penalty': float(np.mean(self.loop_flow_penalty_range)),
            'confinement': float(np.mean(self.confinement_range)),
            'coastal_node_decay': float(np.mean(self.coastal_node_decay_range)),
            'coastal_sinuosity_decay': float(np.mean(self.coastal_sinuosity_decay_range)),
            'min_channel_clearance': float(np.mean(self.min_channel_clearance_range)),
            'outlet_tolerance': float(np.mean(self.outlet_tolerance_range)),
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
    seed: int | None = 42

    # Domain section
    r_shore: float = 10000.0
    theta_base: float = 65.0
    theta_delta: float = 60.0
    confinement: float = 0.0
    lateral_valley_width: float | None = None

    # Backbone section
    branching_weights: tuple[float, float, float] = (0.50, 0.45, 0.05)
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
    target_max_loops_slope: float = 0.22
    target_max_loops_intercept: float = 0.05
    sa_initial_temp: float = 1.0
    sa_cooling_rate: float = 0.975
    sa_max_steps: int = 250

    # Pruning & topological simplification section
    enable_pruning: bool = False
    prune_inland_dead_ends: bool = True
    outlet_tolerance: float = 0.05
    min_spur_length: float = 0.0
    simplify_pass_through_nodes: bool = False
    recalculate_reach_sinuosity: bool = True
    max_prune_iterations: int = 5

    # Hydraulics section
    routing_mode: str = "width"
    murray_p: float = 1.85
    width_beta: float = 0.745
    manning_eta: float = 1.65
    length_gamma: float = 0.50
    apex_width: float = 250.0
    width_noise: float = 0.30
    tidal_flare_k: float = 1.40
    tidal_flare_onset: float = 0.30
    loop_flow_penalty: float = 0.40
    min_channel_width: float = 0.1
    relaxation_max_iter: int = 15
    relaxation_tol: float = 1e-4

    # Channel geometry & proximity section
    enable_curved_centerlines: bool = False
    min_channel_clearance: float = 0.40
    num_polyline_points: int = 21
    meander_bends: int = 1

    # Distributions & Extensions
    distributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validates configuration parameters and invariants."""
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0):
            raise ValueError(f"seed must be an integer >= 0 or None (received {self.seed}).")
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
        if not (0.001 <= self.outlet_tolerance <= 0.50):
            raise ValueError(f"outlet_tolerance must be in [0.001, 0.50] (received {self.outlet_tolerance}).")
        if self.max_prune_iterations < 1:
            raise ValueError(f"max_prune_iterations must be >= 1 (received {self.max_prune_iterations}).")
        if not (0.0 <= self.tidal_flare_onset < 1.0):
            raise ValueError(f"tidal_flare_onset must be in [0.0, 1.0) (received {self.tidal_flare_onset}).")
        if not (0.0 < self.loop_flow_penalty <= 1.0):
            raise ValueError(f"loop_flow_penalty must be in (0.0, 1.0] (received {self.loop_flow_penalty}).")
        if self.apex_width <= 0:
            raise ValueError(f"apex_width must be > 0 (received {self.apex_width}).")
        if self.theta_base <= 0 or self.theta_base > 360.0:
            raise ValueError(f"theta_base must be in (0, 360] degrees (received {self.theta_base}).")
        if self.theta_delta < 0 or self.theta_delta > 360.0:
            raise ValueError(f"theta_delta must be in [0, 360] degrees (received {self.theta_delta}).")
        if not (0.0 < self.width_beta < 1.0):
            raise ValueError(f"width_beta must be in (0.0, 1.0) (received {self.width_beta}).")
        if self.manning_eta <= 0:
            raise ValueError(f"manning_eta must be > 0 (received {self.manning_eta}).")
        if self.length_gamma <= 0:
            raise ValueError(f"length_gamma must be > 0 (received {self.length_gamma}).")
        if self.min_channel_width <= 0:
            raise ValueError(f"min_channel_width must be > 0 (received {self.min_channel_width}).")
        if self.relaxation_tol <= 0:
            raise ValueError(f"relaxation_tol must be > 0 (received {self.relaxation_tol}).")
        if self.relaxation_max_iter < 1:
            raise ValueError(f"relaxation_max_iter must be >= 1 (received {self.relaxation_max_iter}).")
        if self.radial_step_scale_min <= 0 or self.radial_step_scale_min > self.radial_step_scale_max:
            raise ValueError("radial_step_scale_min must be positive and <= radial_step_scale_max.")
        if self.bifurcation_spread_min <= 0 or self.bifurcation_spread_min > self.bifurcation_spread_max:
            raise ValueError("bifurcation_spread_min must be positive and <= bifurcation_spread_max.")
        if self.single_branch_spread_factor <= 0:
            raise ValueError(f"single_branch_spread_factor must be > 0 (received {self.single_branch_spread_factor}).")
        if self.angular_spread_decay <= 0:
            raise ValueError(f"angular_spread_decay must be > 0 (received {self.angular_spread_decay}).")
        if self.candidate_dist_scale <= 0:
            raise ValueError(f"candidate_dist_scale must be > 0 (received {self.candidate_dist_scale}).")
        if not (0.0 <= self.reconnection_prob_base <= 1.0):
            raise ValueError(f"reconnection_prob_base must be in [0.0, 1.0] (received {self.reconnection_prob_base}).")
        if not (0.0 <= self.anastomosis_prob_base <= 1.0):
            raise ValueError(f"anastomosis_prob_base must be in [0.0, 1.0] (received {self.anastomosis_prob_base}).")
        if self.max_in_degree < 1:
            raise ValueError(f"max_in_degree must be >= 1 (received {self.max_in_degree}).")
        if self.max_out_degree < 1:
            raise ValueError(f"max_out_degree must be >= 1 (received {self.max_out_degree}).")
        if self.width_noise < 0.0:
            raise ValueError(f"width_noise must be >= 0.0 (received {self.width_noise}).")
        if self.tidal_flare_k < 0.0:
            raise ValueError(f"tidal_flare_k must be >= 0.0 (received {self.tidal_flare_k}).")

        # Distribution specification validations
        for param_name, dist_spec in self.distributions.items():
            if not isinstance(dist_spec, dict):
                continue
            dtype = str(dist_spec.get("type", "uniform")).lower()
            if dtype == "uniform":
                low = float(dist_spec.get("low", 0.0))
                high = float(dist_spec.get("high", 1.0))
                if low > high:
                    raise ValueError(f"{param_name} distribution low bound ({low}) cannot exceed high bound ({high}).")
            elif dtype in ("normal", "gaussian"):
                std = float(dist_spec.get("std", 1.0))
                if std <= 0:
                    raise ValueError(f"{param_name} normal distribution std must be > 0.")
            elif dtype == "lognormal":
                sigma = float(dist_spec.get("sigma", 1.0))
                if sigma <= 0:
                    raise ValueError(f"{param_name} lognormal distribution sigma must be > 0.")

        if "murray_p" in self.distributions:
            d = self.distributions["murray_p"]
            if isinstance(d, dict) and d.get("type") == "uniform" and d.get("low", 1.0) <= 0.0:
                raise ValueError("murray_p distribution low bound must be > 0.")
        if "confinement" in self.distributions:
            d = self.distributions["confinement"]
            if isinstance(d, dict) and d.get("type") == "uniform" and (d.get("low", 0.0) < 0.0 or d.get("high", 0.95) > 0.95):
                raise ValueError("confinement distribution must be within [0.0, 0.95].")
        if "width_beta" in self.distributions:
            d = self.distributions["width_beta"]
            if isinstance(d, dict) and d.get("type") == "uniform" and (d.get("low", 0.5) <= 0.0 or d.get("high", 0.99) >= 1.0):
                raise ValueError("width_beta distribution must be within (0.0, 1.0).")
        if "coastal_node_decay" in self.distributions:
            d = self.distributions["coastal_node_decay"]
            if isinstance(d, dict) and d.get("type") == "uniform" and (d.get("low", 0.0) < 0.0 or d.get("high", 0.80) > 0.80):
                raise ValueError("coastal_node_decay distribution must be within [0.0, 0.80].")
        if "coastal_sinuosity_decay" in self.distributions:
            d = self.distributions["coastal_sinuosity_decay"]
            if isinstance(d, dict) and d.get("type") == "uniform" and (d.get("low", 0.0) < 0.0 or d.get("high", 0.90) > 0.90):
                raise ValueError("coastal_sinuosity_decay distribution must be within [0.0, 0.90].")
        if "min_channel_clearance" in self.distributions:
            d = self.distributions["min_channel_clearance"]
            if isinstance(d, dict) and d.get("type") == "uniform" and (d.get("low", 0.05) < 0.05 or d.get("high", 1.0) > 1.0):
                raise ValueError("min_channel_clearance distribution must be within [0.05, 1.0].")
        if "min_spur_length" in self.distributions:
            d = self.distributions["min_spur_length"]
            if isinstance(d, dict) and d.get("type") == "uniform" and d.get("low", 0.0) < 0.0:
                raise ValueError("min_spur_length distribution low bound must be >= 0.0.")

    @classmethod
    def load_default(cls, **overrides) -> "DGMConfig":
        """Loads configuration from the packaged default_config.yaml with optional overrides."""
        if DEFAULT_CONFIG_PATH.exists():
            return cls.from_yaml(DEFAULT_CONFIG_PATH, **overrides)
        return cls(**overrides)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "DGMConfig":
        """
        Loads configuration from a YAML file with optional keyword parameter overrides.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            try:
                raw = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Malformed YAML configuration in {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(f"Configuration file {path} must contain a YAML mapping (dict), got {type(raw).__name__}.")

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

        kwargs: dict[str, Any] = {}
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

        # Check section parameters for inline distribution specifications
        dists = dict(kwargs.get("distributions", {}))
        for sec_name in (model_sec, domain_sec, backbone_sec, loops_sec, pruning_sec, hydraulics_sec, channel_geom_sec):
            for k, v in sec_name.items():
                if isinstance(v, dict) and "type" in v:
                    dists[k] = copy.deepcopy(v)
                    if v["type"] == "uniform":
                        kwargs[k] = (float(v.get("low", 0.0)) + float(v.get("high", 1.0))) / 2.0
                    elif v["type"] in ("normal", "gaussian", "lognormal"):
                        kwargs[k] = float(v.get("mean", 0.0))
                    elif v["type"] in ("fixed", "constant"):
                        kwargs[k] = float(v.get("value", 0.0))
                elif isinstance(v, (tuple, list)) and len(v) == 2 and k != "branching_weights" and all(isinstance(x, (int, float)) for x in v):
                    low, high = float(v[0]), float(v[1])
                    dists[k] = {"type": "uniform", "low": low, "high": high}
                    kwargs[k] = (low + high) / 2.0

        # Apply direct user overrides
        for k, v in overrides.items():
            if k == "distributions" and isinstance(v, dict):
                dists.update(v)
            elif isinstance(v, dict) and "type" in v:
                dists[k] = copy.deepcopy(v)
                if v["type"] == "uniform":
                    kwargs[k] = (float(v.get("low", 0.0)) + float(v.get("high", 1.0))) / 2.0
                elif v["type"] in ("normal", "gaussian", "lognormal"):
                    kwargs[k] = float(v.get("mean", 0.0))
                elif v["type"] in ("fixed", "constant"):
                    kwargs[k] = float(v.get("value", 0.0))
            elif isinstance(v, (tuple, list)) and len(v) == 2 and k != "branching_weights" and all(isinstance(x, (int, float)) for x in v):
                low, high = float(v[0]), float(v[1])
                dists[k] = {"type": "uniform", "low": low, "high": high}
                kwargs[k] = (low + high) / 2.0
            else:
                kwargs[k] = v
                dists.pop(k, None)

        kwargs["distributions"] = dists

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_kwargs: dict[str, Any] = {}
        extensions = dict(kwargs.get("extensions", {}))
        for k, v in kwargs.items():
            if k in valid_fields:
                filtered_kwargs[k] = v
            else:
                extensions[k] = v
        filtered_kwargs["extensions"] = extensions

        return cls(**filtered_kwargs)

    def to_yaml(self, path: str | Path) -> None:
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
                "target_max_loops_slope": self.target_max_loops_slope,
                "target_max_loops_intercept": self.target_max_loops_intercept,
                "sa_initial_temp": self.sa_initial_temp,
                "sa_cooling_rate": self.sa_cooling_rate,
                "sa_max_steps": self.sa_max_steps,
            },
            "pruning": {
                "enable_pruning": self.enable_pruning,
                "prune_inland_dead_ends": self.prune_inland_dead_ends,
                "outlet_tolerance": self.outlet_tolerance,
                "min_spur_length": self.min_spur_length,
                "simplify_pass_through_nodes": self.simplify_pass_through_nodes,
                "recalculate_reach_sinuosity": self.recalculate_reach_sinuosity,
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
                "tidal_flare_onset": self.tidal_flare_onset,
                "loop_flow_penalty": self.loop_flow_penalty,
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
        import uuid
        tmp_path = path.with_name(f".{path.name}.tmp_{uuid.uuid4().hex}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            tmp_path.replace(path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @property
    def bounds(self) -> ParameterBounds:
        """Returns a ParameterBounds instance for backwards-compatibility."""
        return ParameterBounds()

    def sample_distribution_parameters(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """
        Samples active parameters from configured distributions.
        For parameters not present in distributions, falls back to the instance baseline.
        Applies physical bounds clamping as robust fallbacks.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        params = self.get_active_parameters(sample=False)
        int_params = {
            "N", "max_in_degree", "max_out_degree", "num_polyline_points",
            "meander_bends", "max_prune_iterations", "relaxation_max_iter", "sa_max_steps"
        }

        physical_param_bounds: dict[str, tuple[float | None, float | None]] = {
            "N": (4.0, None),
            "R_rt": (0.0, 1.0),
            "confinement": (0.0, 0.95),
            "coastal_node_decay": (0.0, 0.80),
            "coastal_sinuosity_decay": (0.0, 0.90),
            "min_channel_clearance": (0.05, 1.0),
            "min_spur_length": (0.0, None),
            "outlet_tolerance": (0.001, 0.50),
            "tidal_flare_onset": (0.0, 0.999),
            "loop_flow_penalty": (0.01, 1.0),
            "apex_width": (1.0, None),
            "r_shore": (100.0, None),
            "theta_base": (5.0, 360.0),
            "theta_delta": (0.0, 360.0),
            "width_beta": (0.01, 0.999),
            "murray_p": (0.1, None),
            "manning_eta": (0.1, None),
            "length_gamma": (0.05, None),
            "min_channel_width": (0.001, None),
            "reconnection_prob_base": (0.0, 1.0),
            "anastomosis_prob_base": (0.0, 1.0),
        }

        for param_name, dist_spec in self.distributions.items():
            if isinstance(dist_spec, dict) and "type" in dist_spec:
                val = sample_from_spec(dist_spec, rng)
                # Bounded fallback clamping for physical integrity
                if param_name in physical_param_bounds:
                    b_min, b_max = physical_param_bounds[param_name]
                    if b_min is not None and isinstance(val, (int, float)):
                        val = max(b_min, val)
                    if b_max is not None and isinstance(val, (int, float)):
                        val = min(b_max, val)

                if param_name in int_params:
                    params[param_name] = round(val)
                else:
                    params[param_name] = val

        return params

    def get_active_parameters(
        self,
        rng: np.random.Generator | None = None,
        sample: bool | None = None
    ) -> dict[str, Any]:
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
            "target_max_loops_slope": self.target_max_loops_slope,
            "target_max_loops_intercept": self.target_max_loops_intercept,
            "sa_initial_temp": self.sa_initial_temp,
            "sa_cooling_rate": self.sa_cooling_rate,
            "sa_max_steps": self.sa_max_steps,
            "enable_pruning": self.enable_pruning,
            "prune_inland_dead_ends": self.prune_inland_dead_ends,
            "outlet_tolerance": self.outlet_tolerance,
            "min_spur_length": self.min_spur_length,
            "simplify_pass_through_nodes": self.simplify_pass_through_nodes,
            "recalculate_reach_sinuosity": self.recalculate_reach_sinuosity,
            "max_prune_iterations": self.max_prune_iterations,
            "routing_mode": self.routing_mode,
            "murray_p": self.murray_p,
            "width_beta": self.width_beta,
            "manning_eta": self.manning_eta,
            "length_gamma": self.length_gamma,
            "apex_width": self.apex_width,
            "width_noise": self.width_noise,
            "tidal_flare_k": self.tidal_flare_k,
            "tidal_flare_onset": self.tidal_flare_onset,
            "loop_flow_penalty": self.loop_flow_penalty,
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

    def set_parameter(
        self,
        name: str,
        value_or_spec: float | str | bool | tuple[float, float] | list[float] | dict[str, Any],
    ) -> None:
        """
        Sets a parameter as either a single fixed value, a uniform distribution (low, high),
        or a full distribution specification dictionary.

        Examples:
            >>> cfg.set_parameter('tidal_flare_onset', 0.55)  # Fixed value, removes from stochastic sampling
            >>> cfg.set_parameter('tidal_flare_onset', (0.35, 0.65))  # Uniform distribution [0.35, 0.65]
            >>> cfg.set_parameter('tidal_flare_onset', {'type': 'normal', 'mean': 0.55, 'std': 0.05})
        """
        if isinstance(value_or_spec, dict) and "type" in value_or_spec:
            self.distributions[name] = copy.deepcopy(value_or_spec)
            if hasattr(self, name):
                if value_or_spec["type"] == "uniform":
                    setattr(self, name, (float(value_or_spec.get("low", 0.0)) + float(value_or_spec.get("high", 1.0))) / 2.0)
                elif value_or_spec["type"] in ("normal", "gaussian", "lognormal"):
                    setattr(self, name, float(value_or_spec.get("mean", 0.0)))
                elif value_or_spec["type"] in ("fixed", "constant"):
                    setattr(self, name, float(value_or_spec.get("value", 0.0)))
                elif value_or_spec["type"] in ("choice", "categorical") and "values" in value_or_spec and len(value_or_spec["values"]) > 0:
                    setattr(self, name, value_or_spec["values"][0])
        elif isinstance(value_or_spec, (tuple, list)) and len(value_or_spec) == 2 and name != "branching_weights" and all(isinstance(x, (int, float)) for x in value_or_spec):
            low, high = float(value_or_spec[0]), float(value_or_spec[1])
            self.distributions[name] = {"type": "uniform", "low": low, "high": high}
            if hasattr(self, name):
                setattr(self, name, (low + high) / 2.0)
        else:
            if hasattr(self, name):
                setattr(self, name, value_or_spec)
            else:
                self.extensions[name] = value_or_spec
            self.distributions.pop(name, None)

        self.validate()

