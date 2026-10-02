"""
Tests for Shoreward Spatial Scaling: node spacing decay and sinuosity reduction near the coast.
"""

import math
import tempfile
from pathlib import Path
import numpy as np
import networkx as nx
import pytest

import dgm
from dgm.config import DGMConfig, ParameterBounds
from dgm.core.geometry_utils import segments_intersect


def test_shoreward_scaling_config_validation():
    """Verify bounds checking on shoreward scaling parameters."""
    # Valid defaults
    cfg = DGMConfig()
    assert cfg.coastal_node_decay == 0.40
    assert cfg.coastal_sinuosity_decay == 0.30
    assert cfg.base_sinuosity_mean == 0.25
    assert cfg.base_sinuosity_std == 0.15

    # Valid boundaries
    cfg_zero = DGMConfig(coastal_node_decay=0.0, coastal_sinuosity_decay=0.0)
    assert cfg_zero.coastal_node_decay == 0.0
    assert cfg_zero.coastal_sinuosity_decay == 0.0

    cfg_max = DGMConfig(coastal_node_decay=0.80, coastal_sinuosity_decay=0.90)
    assert cfg_max.coastal_node_decay == 0.80
    assert cfg_max.coastal_sinuosity_decay == 0.90

    # Invalid coastal_node_decay
    with pytest.raises(ValueError, match="coastal_node_decay must be in"):
        DGMConfig(coastal_node_decay=-0.1)
    with pytest.raises(ValueError, match="coastal_node_decay must be in"):
        DGMConfig(coastal_node_decay=0.85)

    # Invalid coastal_sinuosity_decay
    with pytest.raises(ValueError, match="coastal_sinuosity_decay must be in"):
        DGMConfig(coastal_sinuosity_decay=-0.05)
    with pytest.raises(ValueError, match="coastal_sinuosity_decay must be in"):
        DGMConfig(coastal_sinuosity_decay=0.95)

    # Invalid base_sinuosity
    with pytest.raises(ValueError, match="base_sinuosity_mean must be >= 0.0"):
        DGMConfig(base_sinuosity_mean=-0.1)
    with pytest.raises(ValueError, match="base_sinuosity_std must be > 0.0"):
        DGMConfig(base_sinuosity_std=0.0)


def test_parameter_bounds_and_sampling():
    """Verify ParameterBounds includes shoreward scaling ranges and samples correctly."""
    bounds = ParameterBounds()
    assert bounds.coastal_node_decay_range == (0.0, 0.80)
    assert bounds.coastal_sinuosity_decay_range == (0.0, 0.90)

    rng = np.random.default_rng(42)
    s = bounds.sample(rng)
    assert "coastal_node_decay" in s
    assert 0.0 <= s["coastal_node_decay"] <= 0.80
    assert "coastal_sinuosity_decay" in s
    assert 0.0 <= s["coastal_sinuosity_decay"] <= 0.90

    defs = bounds.default_values()
    assert defs["coastal_node_decay"] == 0.40
    assert defs["coastal_sinuosity_decay"] == 0.45


def test_config_yaml_serialization_roundtrip():
    """Verify shoreward scaling parameters roundtrip cleanly via YAML."""
    cfg = DGMConfig(
        coastal_node_decay=0.55,
        coastal_sinuosity_decay=0.45,
        base_sinuosity_mean=0.20,
        base_sinuosity_std=0.10,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test_scaling.yaml"
        cfg.to_yaml(p)

        loaded = DGMConfig.from_yaml(p)
        assert math.isclose(loaded.coastal_node_decay, 0.55)
        assert math.isclose(loaded.coastal_sinuosity_decay, 0.45)
        assert math.isclose(loaded.base_sinuosity_mean, 0.20)
        assert math.isclose(loaded.base_sinuosity_std, 0.10)


def test_node_spacing_decreases_shoreward():
    """Verify radial node spacing delta_r shrinks as radial distance r increases."""
    # Under strong coastal node decay (0.75), tree edges near shore should be significantly shorter than near apex
    G = dgm.generate(N=60, R_rt=0.4, coastal_node_decay=0.75, seed=42)
    r_shore = G.graph['params'].get('r_shore', 10000.0)

    proximal_steps = []
    distal_steps = []

    for u, v, d in G.edges(data=True):
        if d.get('edge_type') == 'tree' and G.nodes[v].get('node_type') != 'outlet':
            r_u = G.nodes[u]['r']
            r_v = G.nodes[v]['r']
            delta_r = r_v - r_u
            if delta_r > 0:
                if r_u < 0.35 * r_shore:
                    proximal_steps.append(delta_r)
                elif r_u > 0.50 * r_shore:
                    distal_steps.append(delta_r)

    assert len(proximal_steps) >= 3
    assert len(distal_steps) >= 3
    mean_proximal = np.mean(proximal_steps)
    mean_distal = np.mean(distal_steps)

    # Distal step must be significantly shorter than proximal step
    assert mean_distal < mean_proximal
    assert mean_distal <= mean_proximal * 0.75


def test_sinuosity_decreases_shoreward_and_exceeds_unity():
    """Verify channel sinuosity decays shoreward and is strictly >= 1.0 everywhere."""
    for decay in [0.0, 0.3, 0.7]:
        G = dgm.generate(N=40, R_rt=0.5, coastal_sinuosity_decay=decay, seed=123)
        r_shore = G.graph['params'].get('r_shore', 10000.0)

        proximal_sins = []
        distal_sins = []

        for u, v, d in G.edges(data=True):
            sin = d.get('sinuosity', 1.0)
            assert sin >= 1.0, f"Physical violation: sinuosity {sin} < 1.0"

            r_mid = 0.5 * (G.nodes[u]['r'] + G.nodes[v]['r'])
            if r_mid < 0.35 * r_shore:
                proximal_sins.append(sin)
            elif r_mid > 0.65 * r_shore:
                distal_sins.append(sin)

        if decay > 0.4:
            assert len(proximal_sins) >= 2
            assert len(distal_sins) >= 2
            # Sinuosity near shore is strictly lower than upstream
            assert np.mean(distal_sins) < np.mean(proximal_sins)


def test_planarity_and_connectivity_with_shoreward_scaling():
    """Verify that shoreward scaling does not cause planarity crossings or disconnect the network."""
    for seed in [10, 42, 99]:
        G = dgm.generate(N=35, R_rt=0.5, coastal_node_decay=0.7, coastal_sinuosity_decay=0.6, seed=seed)
        assert isinstance(G, nx.DiGraph)
        assert nx.is_weakly_connected(G)
        assert G.graph['coastal_node_decay'] == 0.7
        assert G.graph['coastal_sinuosity_decay'] == 0.6

        # Check edge planarity
        edges = [
            ((G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y']))
            for u, v in G.edges()
        ]
        crossings = 0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                    crossings += 1
        assert crossings == 0


def test_ensemble_with_explicit_shoreward_scaling():
    """Verify explicit shoreward scaling override is preserved across ensemble realizations."""
    graphs = dgm.sample_ensemble(
        N=25,
        R_rt=0.4,
        coastal_node_decay=0.65,
        coastal_sinuosity_decay=0.55,
        num_realizations=4,
        base_seed=77
    )
    assert len(graphs) == 4
    for g in graphs:
        assert math.isclose(g.graph['coastal_node_decay'], 0.65)
        assert math.isclose(g.graph['coastal_sinuosity_decay'], 0.55)


def test_cli_shoreward_scaling(tmp_path):
    """Verify CLI accepts --coastal-node-decay and --coastal-sinuosity-decay."""
    from dgm.cli import main
    out_file = tmp_path / "scaling_delta.graphml"

    main([
        "generate",
        "-N", "25",
        "-R", "0.4",
        "--coastal-node-decay", "0.5",
        "--coastal-sinuosity-decay", "0.4",
        "-s", "42",
        "-o", str(out_file)
    ])
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_shoreward_scaling_extremes_and_boundaries():
    """Verify graph generation under zero and maximum allowed boundary decay parameters."""
    # Zero decay: no node contraction, no sinuosity reduction
    G_zero = dgm.generate(N=30, R_rt=0.3, coastal_node_decay=0.0, coastal_sinuosity_decay=0.0, seed=11)
    assert nx.is_weakly_connected(G_zero)
    for u, v, d in G_zero.edges(data=True):
        assert d.get('sinuosity', 1.0) >= 1.0
        assert d['length'] >= d['straight_distance'] * 0.999

    # Maximum decay boundary: 0.80 node decay, 0.90 sinuosity decay
    G_max = dgm.generate(N=30, R_rt=0.7, coastal_node_decay=0.80, coastal_sinuosity_decay=0.90, seed=22)
    assert nx.is_weakly_connected(G_max)
    for u, v, d in G_max.edges(data=True):
        assert d.get('sinuosity', 1.0) >= 1.0
        assert d['length'] >= d['straight_distance'] * 0.999

    # Check planarity at maximum decay
    edges = [
        ((G_max.nodes[u]['x'], G_max.nodes[u]['y']), (G_max.nodes[v]['x'], G_max.nodes[v]['y']))
        for u, v in G_max.edges()
    ]
    crossings = sum(
        1 for i in range(len(edges))
        for j in range(i + 1, len(edges))
        if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1])
    )
    assert crossings == 0


def test_shoreward_scaling_extreme_network_sizes():
    """Verify robust generation on boundary network sizes (N=4 minimal, N=100 large)."""
    # Minimal graph N=4
    G_min = dgm.generate(N=4, R_rt=0.5, coastal_node_decay=0.75, coastal_sinuosity_decay=0.85, seed=33)
    assert G_min.number_of_nodes() >= 4
    assert nx.is_weakly_connected(G_min)

    # Larger graph N=100
    G_large = dgm.generate(N=100, R_rt=0.4, coastal_node_decay=0.60, coastal_sinuosity_decay=0.50, seed=44)
    assert G_large.number_of_nodes() >= 80
    assert nx.is_weakly_connected(G_large)

    edges = [
        ((G_large.nodes[u]['x'], G_large.nodes[u]['y']), (G_large.nodes[v]['x'], G_large.nodes[v]['y']))
        for u, v in G_large.edges()
    ]
    crossings = sum(
        1 for i in range(len(edges))
        for j in range(i + 1, len(edges))
        if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1])
    )
    assert crossings == 0


def test_direct_override_validation_error():
    """Verify that passing out-of-bounds decay parameters to generate() and sample_ensemble() raises ValueError."""
    # coastal_node_decay out of bounds
    with pytest.raises(ValueError, match="coastal_node_decay must be in"):
        dgm.generate(N=20, coastal_node_decay=-0.1)

    with pytest.raises(ValueError, match="coastal_node_decay must be in"):
        dgm.generate(N=20, coastal_node_decay=0.85)

    with pytest.raises(ValueError, match="coastal_node_decay must be in"):
        dgm.sample_ensemble(N=20, coastal_node_decay=1.5, num_realizations=2)

    # coastal_sinuosity_decay out of bounds
    with pytest.raises(ValueError, match="coastal_sinuosity_decay must be in"):
        dgm.generate(N=20, coastal_sinuosity_decay=-0.05)

    with pytest.raises(ValueError, match="coastal_sinuosity_decay must be in"):
        dgm.generate(N=20, coastal_sinuosity_decay=0.95)

    with pytest.raises(ValueError, match="coastal_sinuosity_decay must be in"):
        dgm.sample_ensemble(N=20, coastal_sinuosity_decay=-0.5, num_realizations=2)


def test_invalid_distribution_range_rejection():
    """Verify that defining out-of-bounds distributions raises ValueError during validation."""
    with pytest.raises(ValueError, match="coastal_node_decay distribution must be within"):
        DGMConfig(distributions={"coastal_node_decay": {"type": "uniform", "low": -0.1, "high": 0.5}})

    with pytest.raises(ValueError, match="coastal_node_decay distribution must be within"):
        DGMConfig(distributions={"coastal_node_decay": {"type": "uniform", "low": 0.1, "high": 0.95}})

    with pytest.raises(ValueError, match="coastal_sinuosity_decay distribution must be within"):
        DGMConfig(distributions={"coastal_sinuosity_decay": {"type": "uniform", "low": -0.2, "high": 0.5}})

    with pytest.raises(ValueError, match="coastal_sinuosity_decay distribution must be within"):
        DGMConfig(distributions={"coastal_sinuosity_decay": {"type": "uniform", "low": 0.1, "high": 1.05}})

