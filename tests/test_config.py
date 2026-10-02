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
