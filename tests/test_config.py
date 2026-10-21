"""
Tests for DGM configuration loading, validation, and distribution sampling.
"""

from pathlib import Path

import numpy as np
import pytest

from dgm.config import DGMConfig, sample_from_spec


def test_default_config_loading():
    """Verify default configuration values and baseline parameter distributions."""
    cfg = DGMConfig.load_default()
    assert cfg.N == 50
    assert cfg.R_rt == 0.5
    assert cfg.loop_mechanism == "potential_field"
    assert "murray_p" in cfg.distributions
    assert cfg.distributions["murray_p"]["type"] == "uniform"


@pytest.mark.parametrize(
    "config_kwargs,match_pattern",
    [
        ({"N": 2}, "Node budget N must be >= 4"),
        ({"R_rt": 1.5}, "River-to-Tidal forcing ratio R_rt must be in"),
        ({"loop_mechanism": "invalid_mech"}, "loop_mechanism must be"),
        ({"outlet_tolerance": 0.9}, "outlet_tolerance must be in"),
        ({"tidal_flare_onset": 1.5}, "tidal_flare_onset must be in"),
        ({"loop_flow_penalty": 0.0}, "loop_flow_penalty must be in"),
    ],
)
def test_config_validation(config_kwargs, match_pattern):
    """Verify out-of-bounds or invalid configuration parameters raise descriptive ValueError."""
    with pytest.raises(ValueError, match=match_pattern):
        DGMConfig(**config_kwargs)


def test_config_yaml_roundtrip(tmp_path):
    """Verify complete configuration roundtrip serialization through YAML using tmp_path."""
    cfg = DGMConfig(N=64, R_rt=0.75, loop_mechanism="ner_optimality", seed=999)
    yaml_path = tmp_path / "test_config.yaml"
    cfg.to_yaml(yaml_path)

    loaded_cfg = DGMConfig.from_yaml(yaml_path)
    assert loaded_cfg.N == 64
    assert loaded_cfg.R_rt == 0.75
    assert loaded_cfg.loop_mechanism == "ner_optimality"
    assert loaded_cfg.seed == 999


def test_sample_from_spec_uniform():
    """Verify uniform distribution spec sampling produces values strictly within bounds."""
    rng = np.random.default_rng(42)
    spec = {"type": "uniform", "low": 10.0, "high": 20.0}
    samples = np.array([sample_from_spec(spec, rng) for _ in range(50)])
    assert np.all(samples >= 10.0)
    assert np.all(samples <= 20.0)
    assert np.any(samples > 10.0) and np.any(samples < 20.0)


def test_sample_from_spec_normal():
    """Verify clamped normal distribution spec sampling respects truncated bounds."""
    rng = np.random.default_rng(42)
    spec = {"type": "normal", "mean": 100.0, "std": 10.0, "min": 80.0, "max": 120.0}
    samples = np.array([sample_from_spec(spec, rng) for _ in range(50)])
    assert np.all(samples >= 80.0)
    assert np.all(samples <= 120.0)


def test_sample_parameters_from_config():
    """Verify sampling from default config produces all physical distribution parameters in valid bounds."""
    cfg = DGMConfig.load_default()
    rng = np.random.default_rng(123)
    sampled = cfg.sample_distribution_parameters(rng)

    assert "murray_p" in sampled
    assert 1.20 <= sampled["murray_p"] <= 2.20
    assert 0.50 <= sampled["width_beta"] <= 0.99
    assert 5000.0 <= sampled["r_shore"] <= 15000.0
    assert 0.10 <= sampled["tidal_flare_onset"] <= 0.50
    assert 0.20 <= sampled["loop_flow_penalty"] <= 0.70
    assert 0.02 <= sampled["outlet_tolerance"] <= 0.10


def test_set_parameter_scalar_and_distributions():
    """Verify set_parameter correctly modifies scalars, uniform tuples, and spec dictionaries."""
    cfg = DGMConfig.load_default()

    # 1. Setting fixed scalar removes from distributions so it stays fixed during sampling
    cfg.set_parameter("tidal_flare_onset", 0.55)
    assert cfg.tidal_flare_onset == 0.55
    assert "tidal_flare_onset" not in cfg.distributions
    sampled = cfg.sample_distribution_parameters()
    assert sampled["tidal_flare_onset"] == 0.55

    # 2. Setting uniform tuple adds to distributions and sets baseline to midpoint
    cfg.set_parameter("tidal_flare_onset", (0.40, 0.60))
    assert cfg.tidal_flare_onset == 0.50
    assert "tidal_flare_onset" in cfg.distributions
    assert cfg.distributions["tidal_flare_onset"] == {"type": "uniform", "low": 0.40, "high": 0.60}

    # 3. Setting distribution spec dict
    spec = {"type": "normal", "mean": 0.55, "std": 0.05, "min": 0.30, "max": 0.80}
    cfg.set_parameter("tidal_flare_onset", spec)
    assert cfg.tidal_flare_onset == 0.55
    assert cfg.distributions["tidal_flare_onset"] == spec


def test_load_default_with_tuple_and_scalar_overrides():
    """Verify load_default accepts tuple range and scalar overrides."""
    # Test tuple range override
    cfg_tuple = DGMConfig.load_default(tidal_flare_onset=(0.35, 0.65))
    assert cfg_tuple.tidal_flare_onset == 0.50
    assert cfg_tuple.distributions["tidal_flare_onset"]["low"] == 0.35
    assert cfg_tuple.distributions["tidal_flare_onset"]["high"] == 0.65

    # Test fixed scalar override
    cfg_scalar = DGMConfig.load_default(tidal_flare_onset=0.55)
    assert cfg_scalar.tidal_flare_onset == 0.55
    assert "tidal_flare_onset" not in cfg_scalar.distributions


def test_config_files_identical():
    """Verify config/default_config.yaml and src/dgm/default_config.yaml remain synchronized."""
    repo_root = Path(__file__).resolve().parent.parent
    repo_cfg_path = repo_root / "config" / "default_config.yaml"
    pkg_cfg_path = repo_root / "src" / "dgm" / "default_config.yaml"
    if not repo_cfg_path.exists():
        pytest.skip("Root config/default_config.yaml not present in distribution environment")
    cfg1 = repo_cfg_path.read_text(encoding="utf-8")
    cfg2 = pkg_cfg_path.read_text(encoding="utf-8")
    assert cfg1 == cfg2, "config/default_config.yaml and src/dgm/default_config.yaml have drifted!"

