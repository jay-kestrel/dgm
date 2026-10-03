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
from dgm.generator import DeltaGenerator, EnsembleGenerator, generate, sample_ensemble
from dgm.core.hydraulics import apply_emergent_hydraulics, solve_markov_flux
from dgm.metrics import MetricsBattery


def test_empty_graph():
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
    G_single = nx.DiGraph()
    G_single.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    C, W, phi = solve_markov_flux(G_single)
    assert len(phi) == 1
    assert phi[0] == 1.0

    mb = MetricsBattery()
    metrics = mb.evaluate_all(G_single)
    assert metrics["alpha_index"] == 0.0
    assert metrics["nER_star"] == 0.0


def test_disconnected_graph():
    G_disc = nx.DiGraph()
    G_disc.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    G_disc.add_node(1, x=100.0, y=0.0, r=100.0, theta=0.0, node_type="outlet")
    G_disc.add_node(2, x=0.0, y=200.0, r=200.0, theta=0.0, node_type="outlet")
    G_disc.add_edge(0, 1, width=10.0, length=100.0)
    # Node 2 is isolated

    C, W, phi = solve_markov_flux(G_disc)
    assert len(phi) == 3
    assert phi[0] == 1.0
    assert phi[1] == 1.0
    assert phi[2] == 0.0

    mb = MetricsBattery()
    metrics = mb.evaluate_all(G_disc)
    assert "alpha_index" in metrics
    assert "nER_star" in metrics


def test_zero_length_channel():
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    G.add_node(1, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="outlet")
    G.add_edge(0, 1, width=10.0, length=0.0, straight_distance=0.0)

    apply_emergent_hydraulics(G)
    assert G.edges[0, 1]["length"] >= 1.0
    assert G.edges[0, 1]["conductance"] > 0.0


def test_singular_markov_matrix_cyclic_trap():
    # Construct a directed graph with a cycle that cannot reach absorbing outlets
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type="apex", is_apex=True)
    G.add_node(1, x=10.0, y=0.0, r=10.0, theta=0.0, node_type="junction")
    G.add_node(2, x=20.0, y=0.0, r=20.0, theta=0.0, node_type="junction")
    G.add_edge(0, 1, width=10.0, length=10.0)
    G.add_edge(1, 2, width=10.0, length=10.0)
    G.add_edge(2, 1, width=10.0, length=10.0)

    C, W, phi = solve_markov_flux(G)
    assert len(phi) == 3
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0.0)


def test_distribution_sampling_types():
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

    # Validation guards for distributions
    with pytest.raises(ValueError, match="Uniform distribution low bound"):
        sample_from_spec({"type": "uniform", "low": 10.0, "high": 5.0}, rng)

    with pytest.raises(ValueError, match="Normal distribution std must be > 0"):
        sample_from_spec({"type": "normal", "mean": 0.0, "std": -1.0}, rng)

    with pytest.raises(ValueError, match="Lognormal distribution sigma must be > 0"):
        sample_from_spec({"type": "lognormal", "mean": 0.0, "sigma": 0.0}, rng)

    with pytest.raises(ValueError, match="Choice distribution values list cannot be empty"):
        sample_from_spec({"type": "choice", "values": []}, rng)

    with pytest.raises(ValueError, match="Distribution weights must be non-negative"):
        sample_from_spec({"type": "choice", "values": [1.0, 2.0], "weights": [-1.0, 1.0]}, rng)


def test_config_physical_validation_guards():
    with pytest.raises(ValueError, match="apex_width must be > 0"):
        DGMConfig(apex_width=-5.0)

    with pytest.raises(ValueError, match="theta_base must be > 0"):
        DGMConfig(theta_base=0.0)

    with pytest.raises(ValueError, match="theta_delta must be >= 0"):
        DGMConfig(theta_delta=-1.0)

    with pytest.raises(ValueError, match="width_beta must be in"):
        DGMConfig(width_beta=1.5)

    with pytest.raises(ValueError, match="manning_eta must be > 0"):
        DGMConfig(manning_eta=-1.0)

    with pytest.raises(ValueError, match="min_channel_width must be > 0"):
        DGMConfig(min_channel_width=0.0)

    with pytest.raises(ValueError, match="murray_p distribution low bound"):
        DGMConfig(distributions={"murray_p": {"type": "uniform", "low": -1.0, "high": 2.0}})


def test_target_nodes_overrides_in_generator_facades():
    # Test generate top-level with target_nodes
    G = generate(target_nodes=25, seed=42)
    assert G.graph["N_input"] == 25

    # Test sample_ensemble top-level with target_nodes
    ens = sample_ensemble(target_nodes=25, num_realizations=2, base_seed=42)
    assert len(ens) == 2
    assert ens[0].graph["N_input"] == 25
