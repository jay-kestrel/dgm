"""
Unit tests for edge cases and mathematical robustness in DGM:
- Empty graphs
- Single-node graphs
- Disconnected graphs
- Zero-length channels
- Singular Markov transition matrices
- Stochastic distribution types and validation guards
- Parameter overrides (target_nodes)
"""

import networkx as nx
import numpy as np
import pytest

from dgm.config import DGMConfig, sample_from_spec
from dgm.core.hydraulics import apply_emergent_hydraulics, solve_markov_flux
from dgm.generator import generate, sample_ensemble
from dgm.metrics import MetricsBattery


def test_empty_graph():
    """Verify Markov flux solver and metrics battery handle empty graphs gracefully."""
    G_empty = nx.DiGraph()
    C, W, phi = solve_markov_flux(G_empty)
    assert C.shape == (0, 0)
    assert W.shape == (0, 0)
    assert phi.shape == (0,)

    mb = MetricsBattery()
    metrics = mb.evaluate_all(G_empty)
    assert metrics["alpha_index"] == 0.0
    assert metrics["nER_star"] == 0.0


def test_single_node_graph():
    """Verify Markov flux solver and metrics battery handle single-node isolated apex."""
    G_single = nx.DiGraph()
    G_single.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    _C, _W, phi = solve_markov_flux(G_single)
    assert len(phi) == 1
    assert pytest.approx(phi[0], abs=1e-5) == 1.0

    mb = MetricsBattery()
    metrics = mb.evaluate_all(G_single)
    assert metrics["alpha_index"] == 0.0
    assert metrics["nER_star"] == 0.0


def test_disconnected_graph():
    """Verify Markov flux solver allocates 0 flux to disconnected components."""
    G_disc = nx.DiGraph()
    G_disc.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    G_disc.add_node(1, x=100.0, y=0.0, r=100.0, theta=0.0, node_type="outlet")
    G_disc.add_node(2, x=0.0, y=200.0, r=200.0, theta=0.0, node_type="outlet")
    G_disc.add_edge(0, 1, width=10.0, length=100.0)
    # Node 2 is isolated

    _C, _W, phi = solve_markov_flux(G_disc)
    assert len(phi) == 3
    assert pytest.approx(phi[0], abs=1e-5) == 1.0
    assert pytest.approx(phi[1], abs=1e-5) == 1.0
    assert pytest.approx(phi[2], abs=1e-5) == 0.0

    mb = MetricsBattery()
    metrics = mb.evaluate_all(G_disc)
    assert "alpha_index" in metrics
    assert "nER_star" in metrics


def test_zero_length_channel():
    """Verify hydraulics solver clamps zero-length channels to minimum physical length."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    G.add_node(1, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="outlet")
    G.add_edge(0, 1, width=10.0, length=0.0, straight_distance=0.0)

    apply_emergent_hydraulics(G)
    assert G.edges[0, 1]["length"] >= 1.0
    assert G.edges[0, 1]["conductance"] > 0.0


def test_singular_markov_matrix_cyclic_trap():
    """Verify flux solver remains stable and finite even when encountering absorbing cyclic subgraphs."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    G.add_node(1, x=10.0, y=0.0, r=10.0, theta=0.0, node_type="junction")
    G.add_node(2, x=20.0, y=0.0, r=20.0, theta=0.0, node_type="junction")
    G.add_edge(0, 1, width=10.0, length=10.0)
    G.add_edge(1, 2, width=10.0, length=10.0)
    G.add_edge(2, 1, width=10.0, length=10.0)

    _C, _W, phi = solve_markov_flux(G)
    assert len(phi) == 3
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0.0)


def test_distribution_sampling_types():
    """Verify stochastic parameter distribution types (lognormal, choice, fixed)."""
    rng = np.random.default_rng(42)

    # Lognormal
    val_lognorm = sample_from_spec({"type": "lognormal", "mean": 1.0, "sigma": 0.5, "min": 0.1, "max": 10.0}, rng)
    assert 0.1 <= val_lognorm <= 10.0

    # Choice / categorical with weights
    val_choice = sample_from_spec({"type": "choice", "values": [10.0, 20.0, 30.0], "weights": [0.1, 0.8, 0.1]}, rng)
    assert val_choice in [10.0, 20.0, 30.0]

    # Fixed / constant
    val_fixed = sample_from_spec({"type": "fixed", "value": 42.5}, rng)
    assert val_fixed == 42.5


@pytest.mark.parametrize(
    "spec,match_pattern",
    [
        ({"type": "uniform", "low": 10.0, "high": 5.0}, "Uniform distribution low bound"),
        ({"type": "normal", "mean": 0.0, "std": -1.0}, "Normal distribution std must be > 0"),
        ({"type": "lognormal", "mean": 0.0, "sigma": 0.0}, "Lognormal distribution sigma must be > 0"),
        ({"type": "choice", "values": []}, "Choice distribution values list cannot be empty"),
        ({"type": "choice", "values": [1.0, 2.0], "weights": [-1.0, 1.0]}, "Distribution weights must be non-negative"),
    ],
)
def test_distribution_sampling_validation_errors(spec, match_pattern):
    """Verify invalid distribution specifications raise ValueError."""
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match=match_pattern):
        sample_from_spec(spec, rng)


@pytest.mark.parametrize(
    "config_kwargs,match_pattern",
    [
        ({"apex_width": -5.0}, "apex_width must be > 0"),
        ({"theta_base": 0.0}, "theta_base must be in"),
        ({"theta_delta": -1.0}, "theta_delta must be in"),
        ({"width_beta": 1.5}, "width_beta must be in"),
        ({"manning_eta": -1.0}, "manning_eta must be > 0"),
        ({"min_channel_width": 0.0}, "min_channel_width must be > 0"),
        ({"distributions": {"murray_p": {"type": "uniform", "low": -1.0, "high": 2.0}}}, "murray_p distribution low bound"),
    ],

)
def test_config_physical_validation_guards(config_kwargs, match_pattern):
    """Verify physical bound guards in DGMConfig raise descriptive ValueError."""
    with pytest.raises(ValueError, match=match_pattern):
        DGMConfig(**config_kwargs)


def test_target_nodes_overrides_in_generator_facades():
    """Verify target_nodes keyword correctly propagates across generate and sample_ensemble."""
    G = generate(target_nodes=25, seed=42)
    assert G.graph["N_input"] == 25

    ens = sample_ensemble(target_nodes=25, num_realizations=2, base_seed=42)
    assert len(ens) == 2
    assert ens[0].graph["N_input"] == 25


def test_metrics_zero_outlets_and_angular_dispersion():
    """Verify 0-outlet graph reports num_outlets=0, and angular_dispersion calculates branching deflections."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex")
    G.add_node(1, x=10.0, y=0.0, r=10.0, theta=0.0, node_type="junction")
    G.add_node(2, x=20.0, y=0.0, r=20.0, theta=0.0, node_type="junction")
    G.add_edge(0, 1, width=10.0, length=10.0)
    G.add_edge(1, 2, width=10.0, length=10.0)
    G.add_edge(2, 1, width=10.0, length=10.0)

    mb = MetricsBattery()
    m = mb.evaluate_all(G)
    assert m["num_outlets"] == 0
    assert m["log_num_outlets"] == 0.0

    # Test angular dispersion on a network with bifurcation
    G_bif = nx.DiGraph()
    G_bif.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex")
    G_bif.add_node(1, x=100.0, y=0.0, r=100.0, theta=0.0, node_type="bifurcation")
    G_bif.add_node(2, x=200.0, y=50.0, r=206.1, theta=0.24, node_type="outlet")
    G_bif.add_node(3, x=200.0, y=-50.0, r=206.1, theta=-0.24, node_type="outlet")
    G_bif.add_edge(0, 1, width=20.0, length=100.0, edge_type="tree")
    G_bif.add_edge(1, 2, width=14.0, length=111.8, edge_type="tree")
    G_bif.add_edge(1, 3, width=14.0, length=111.8, edge_type="tree")

    m_bif = mb.evaluate_all(G_bif)
    assert m_bif["num_outlets"] == 2
    assert m_bif["angular_dispersion"] > 0.0

