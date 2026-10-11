"""
Tests for DGM configuration loading, validation, and distribution sampling.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from dgm.config import DGMConfig, sample_from_spec


def test_default_config_loading():
    cfg = DGMConfig.load_default()
    assert cfg.N == 50
    assert cfg.R_rt == 0.5
    assert cfg.loop_mechanism == "potential_field"
    assert "murray_p" in cfg.distributions
    assert cfg.distributions["murray_p"]["type"] == "uniform"


def test_config_validation():
    with pytest.raises(ValueError, match="Node budget N must be >= 4"):
        DGMConfig(N=2)

    with pytest.raises(ValueError, match="River-to-Tidal forcing ratio R_rt must be in"):
        DGMConfig(R_rt=1.5)

    with pytest.raises(ValueError, match="loop_mechanism must be"):
        DGMConfig(loop_mechanism="invalid_mech")

    with pytest.raises(ValueError, match="outlet_tolerance must be in"):
        DGMConfig(outlet_tolerance=0.9)

    with pytest.raises(ValueError, match="tidal_flare_onset must be in"):
        DGMConfig(tidal_flare_onset=1.5)

    with pytest.raises(ValueError, match="loop_flow_penalty must be in"):
        DGMConfig(loop_flow_penalty=0.0)


def test_config_yaml_roundtrip():
    cfg = DGMConfig(N=64, R_rt=0.75, loop_mechanism="ner_optimality", seed=999)
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "test_config.yaml"
        cfg.to_yaml(yaml_path)

        loaded_cfg = DGMConfig.from_yaml(yaml_path)
        assert loaded_cfg.N == 64
        assert loaded_cfg.R_rt == 0.75
        assert loaded_cfg.loop_mechanism == "ner_optimality"
        assert loaded_cfg.seed == 999


def test_sample_from_spec_uniform():
    rng = np.random.default_rng(42)
    spec = {"type": "uniform", "low": 10.0, "high": 20.0}
    for _ in range(50):
        val = sample_from_spec(spec, rng)
        assert 10.0 <= val <= 20.0


def test_sample_from_spec_normal():
    rng = np.random.default_rng(42)
    spec = {"type": "normal", "mean": 100.0, "std": 10.0, "min": 80.0, "max": 120.0}
    for _ in range(50):
        val = sample_from_spec(spec, rng)
        assert 80.0 <= val <= 120.0


def test_sample_parameters_from_config():
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
    # Test tuple range override
    cfg_tuple = DGMConfig.load_default(tidal_flare_onset=(0.35, 0.65))
    assert cfg_tuple.tidal_flare_onset == 0.50
    assert cfg_tuple.distributions["tidal_flare_onset"]["low"] == 0.35
    assert cfg_tuple.distributions["tidal_flare_onset"]["high"] == 0.65

    # Test fixed scalar override
    cfg_scalar = DGMConfig.load_default(tidal_flare_onset=0.55)
    assert cfg_scalar.tidal_flare_onset == 0.55
    assert "tidal_flare_onset" not in cfg_scalar.distributions

