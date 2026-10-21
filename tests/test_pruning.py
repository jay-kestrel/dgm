"""
Tests for pre-hydraulic network pruning and topological simplification in DGM.
Verifies dead-end spur removal, shoreline outlet protection, pass-through chain contraction,
coordinate merging, and seamless integration with hydraulics and ensemble generation.
"""

import math

import networkx as nx
import numpy as np
import pytest

import dgm
from dgm.cli import main as cli_main
from dgm.config import DGMConfig
from dgm.core.hydraulics import solve_markov_flux
from dgm.core.pruning import (
    prune_and_simplify_network,
    prune_dead_end_spurs,
    refresh_node_types,
    simplify_degree_two_chains,
)


def test_refresh_node_types():
    """Verify dynamic classification of node roles based on degrees and radial distance."""
    G = nx.DiGraph()
    # Apex
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    # Bifurcation
    G.add_node(1, x=2000.0, y=0.0, r=2000.0, theta=0.0)
    # Confluence
    G.add_node(2, x=4000.0, y=500.0, r=4031.0, theta=0.12)
    # Inland dead end
    G.add_node(3, x=5000.0, y=-500.0, r=5024.0, theta=-0.10)
    # Shoreline outlet (r >= 10000 * 0.95 = 9500)
    G.add_node(4, x=9800.0, y=500.0, r=9812.0, theta=0.05)
    # Internal pass-through
    G.add_node(5, x=6000.0, y=500.0, r=6020.0, theta=0.08)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 5)
    G.add_edge(5, 4)

    refresh_node_types(G, r_shore=10000.0, outlet_tolerance=0.05)

    assert G.nodes[0]['node_type'] == 'apex'
    assert G.nodes[1]['node_type'] == 'bifurcation'
    assert G.nodes[2]['node_type'] == 'internal'  # in=1, out=1
    assert G.nodes[3]['node_type'] == 'dead_end'   # in=1, out=0, r < 9500
    assert G.nodes[4]['node_type'] == 'outlet'     # in=1, out=0, r >= 9500


def test_prune_dead_end_spurs_inland_only():
    """Verify inland blind dead ends are pruned while shoreline outlets and apex are strictly protected."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    G.add_node(1, x=3000.0, y=0.0, r=3000.0, theta=0.0)
    # Inland dead end
    G.add_node(2, x=5000.0, y=-500.0, r=5024.0, theta=-0.10)
    # Shoreline outlet
    G.add_node(3, x=9900.0, y=200.0, r=9902.0, theta=0.02)

    G.add_edge(0, 1, length=3000.0)
    G.add_edge(1, 2, length=2061.0)
    G.add_edge(1, 3, length=6902.0)

    pruned_G, count = prune_dead_end_spurs(G, r_shore=10000.0, prune_inland_dead_ends=True)
    assert count == 1
    assert 2 not in pruned_G
    assert 0 in pruned_G
    assert 1 in pruned_G
    assert 3 in pruned_G
    assert pruned_G.has_edge(0, 1)
    assert pruned_G.has_edge(1, 3)
    assert not pruned_G.has_edge(1, 2)


def test_prune_cascading_dead_ends():
    """Verify cascading inland dead-ends are pruned up to max_iterations."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    G.add_node(1, x=2000.0, y=0.0, r=2000.0, theta=0.0)
    # Node 2 bifurcates into 3 and 4, both dead-ends
    G.add_node(2, x=4000.0, y=-500.0, r=4031.0, theta=-0.12)
    G.add_node(3, x=5000.0, y=-300.0, r=5008.0, theta=-0.06)
    G.add_node(4, x=5000.0, y=-700.0, r=5048.0, theta=-0.14)
    # Valid shoreline outlet
    G.add_node(5, x=9800.0, y=300.0, r=9804.0, theta=0.03)

    G.add_edge(0, 1, length=2000.0)
    G.add_edge(1, 2, length=2061.0)
    G.add_edge(2, 3, length=1019.0)
    G.add_edge(2, 4, length=1019.0)
    G.add_edge(1, 5, length=7805.0)

    # Iteration 1 prunes 3 and 4 -> node 2 becomes dead-end
    # Iteration 2 prunes 2 -> node 1 now has only 1 child (node 5)
    pruned_G, count = prune_dead_end_spurs(G, r_shore=10000.0, max_iterations=5, prune_inland_dead_ends=True)
    assert count == 3
    assert 3 not in pruned_G
    assert 4 not in pruned_G
    assert 2 not in pruned_G
    assert 0 in pruned_G
    assert 1 in pruned_G
    assert 5 in pruned_G
    assert list(pruned_G.successors(1)) == [5]


def test_shoreline_outlets_strictly_protected():
    """Verify that even with min_spur_length set high, valid shoreline outlets are NEVER pruned."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    G.add_node(1, x=9000.0, y=0.0, r=9000.0, theta=0.0)
    # Outlet very close to node 1 (short edge, only 600m), but reaches shoreline r=9600 >= 9500
    G.add_node(2, x=9600.0, y=0.0, r=9600.0, theta=0.0)

    G.add_edge(0, 1, length=9000.0)
    G.add_edge(1, 2, length=600.0)

    # min_spur_length=2000 > 600m, but node 2 is an outlet (r >= 9500)
    pruned_G, count = prune_dead_end_spurs(
        G, r_shore=10000.0, min_spur_length=2000.0, prune_inland_dead_ends=True
    )
    assert count == 0
    assert 2 in pruned_G
    assert pruned_G.has_edge(1, 2)


def test_min_spur_length_filtering():
    """Verify that short inland spurs below min_spur_length are pruned."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    G.add_node(1, x=3000.0, y=0.0, r=3000.0, theta=0.0)
    # Short inland spur (length 150m < 500m)
    G.add_node(2, x=3150.0, y=0.0, r=3150.0, theta=0.0)
    # Long inland spur (length 1500m >= 500m)
    G.add_node(3, x=4500.0, y=200.0, r=4504.0, theta=0.04)
    # Shoreline outlet
    G.add_node(4, x=9800.0, y=0.0, r=9800.0, theta=0.0)

    G.add_edge(0, 1, length=3000.0)
    G.add_edge(1, 2, length=150.0)
    G.add_edge(1, 3, length=1513.0)
    G.add_edge(1, 4, length=6800.0)

    # prune_inland_dead_ends=False, but min_spur_length=500.0
    pruned_G, count = prune_dead_end_spurs(
        G, r_shore=10000.0, min_spur_length=500.0, prune_inland_dead_ends=False
    )
    assert count == 1
    assert 2 not in pruned_G
    assert 3 in pruned_G
    assert 4 in pruned_G


def test_simplify_degree_two_chains_and_coordinates():
    """Verify pass-through degree-2 nodes are contracted and polyline coordinates merged."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    # Pass-through node 1
    G.add_node(1, x=2000.0, y=100.0, r=2002.0, theta=0.05)
    # Pass-through node 2
    G.add_node(2, x=4000.0, y=200.0, r=4004.0, theta=0.05)
    # Bifurcation node 3
    G.add_node(3, x=6000.0, y=0.0, r=6000.0, theta=0.0)
    # Outlets 4 and 5
    G.add_node(4, x=9800.0, y=500.0, r=9812.0, theta=0.05)
    G.add_node(5, x=9800.0, y=-500.0, r=9812.0, theta=-0.05)

    coords_01 = [[0.0, 0.0], [1000.0, 50.0], [2000.0, 100.0]]
    coords_12 = [[2000.0, 100.0], [3000.0, 150.0], [4000.0, 200.0]]
    coords_23 = [[4000.0, 200.0], [5000.0, 100.0], [6000.0, 0.0]]

    G.add_edge(0, 1, length=2002.5, straight_distance=2002.5, sinuosity=1.0, edge_type='tree', coordinates=coords_01)
    G.add_edge(1, 2, length=2002.5, straight_distance=2002.5, sinuosity=1.0, edge_type='tree', coordinates=coords_12)
    G.add_edge(2, 3, length=2009.9, straight_distance=2009.9, sinuosity=1.0, edge_type='tree', coordinates=coords_23)
    G.add_edge(3, 4, length=3832.0, straight_distance=3832.0, sinuosity=1.0, edge_type='tree')
    G.add_edge(3, 5, length=3832.0, straight_distance=3832.0, sinuosity=1.0, edge_type='tree')

    simplified_G, count = simplify_degree_two_chains(G, r_shore=10000.0)
    assert count == 2
    assert 1 not in simplified_G
    assert 2 not in simplified_G
    assert 0 in simplified_G
    assert 3 in simplified_G
    assert simplified_G.has_edge(0, 3)

    edge_data = simplified_G.edges[0, 3]
    assert math.isclose(edge_data['length'], 2002.5 + 2002.5 + 2009.9, rel_tol=1e-5)
    assert 'coordinates' in edge_data
    # Merged coordinates should have no duplicates at joints
    coords = edge_data['coordinates']
    assert coords[0] == [0.0, 0.0]
    assert coords[-1] == [6000.0, 0.0]
    assert len(coords) == 7  # 3 + 2 + 2 = 7 points total


def test_simplify_chains_avoids_multiedge_and_cycle():
    """Verify that chain contraction does not create multi-edges or self-loops."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    G.add_node(1, x=2000.0, y=100.0, r=2002.0, theta=0.05)
    G.add_node(2, x=4000.0, y=0.0, r=4000.0, theta=0.0)

    # Edge 0 -> 2 already exists! Node 1 is an alternative path 0 -> 1 -> 2
    G.add_edge(0, 2, length=4000.0, straight_distance=4000.0, sinuosity=1.0)
    G.add_edge(0, 1, length=2002.0, straight_distance=2002.0, sinuosity=1.0)
    G.add_edge(1, 2, length=2002.0, straight_distance=2002.0, sinuosity=1.0)

    # Attempting to simplify node 1 should be skipped because 0 -> 2 already exists
    simplified_G, count = simplify_degree_two_chains(G, r_shore=10000.0)
    assert count == 0
    assert 1 in simplified_G


def test_pruning_disabled_by_default():
    """Verify prune_and_simplify_network returns input graph unchanged if enable_pruning=False."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0)
    G.add_node(1, x=1000.0, y=0.0, r=1000.0)
    G.add_edge(0, 1, length=1000.0)

    result = prune_and_simplify_network(G, params={'enable_pruning': False})
    assert result is G


@pytest.mark.parametrize("config_kwargs,match_pattern", [
    ({"min_spur_length": -10.0}, "min_spur_length"),
    ({"max_prune_iterations": 0}, "max_prune_iterations"),
])
def test_pruning_config_validation_errors(config_kwargs, match_pattern):
    """Verify invalid pruning parameters raise ValueError."""
    with pytest.raises(ValueError, match=match_pattern):
        DGMConfig(**config_kwargs)


def test_pruning_config_yaml_roundtrip(tmp_path):
    """Verify DGMConfig pruning parameters roundtrip cleanly via YAML using tmp_path."""
    cfg = DGMConfig(
        enable_pruning=True,
        prune_inland_dead_ends=True,
        min_spur_length=150.0,
        simplify_pass_through_nodes=True,
        max_prune_iterations=8
    )

    active = cfg.get_active_parameters()
    assert active["enable_pruning"] is True
    assert active["prune_inland_dead_ends"] is True
    assert active["min_spur_length"] == 150.0
    assert active["simplify_pass_through_nodes"] is True
    assert active["max_prune_iterations"] == 8

    yaml_path = tmp_path / "test_pruning_cfg.yaml"
    cfg.to_yaml(yaml_path)

    loaded_cfg = DGMConfig.from_yaml(yaml_path)
    assert loaded_cfg.enable_pruning is True
    assert loaded_cfg.prune_inland_dead_ends is True
    assert loaded_cfg.min_spur_length == 150.0
    assert loaded_cfg.simplify_pass_through_nodes is True
    assert loaded_cfg.max_prune_iterations == 8



def test_generator_end_to_end_pruning():
    """Verify full end-to-end generation with pruning enabled succeeds and eliminates inland dead-ends."""
    G = dgm.generate(
        N=50,
        R_rt=0.4,
        enable_pruning=True,
        prune_inland_dead_ends=True,
        simplify_pass_through_nodes=True,
        seed=101
    )

    assert G.graph['enable_pruning'] is True
    assert 'pruning_stats' in G.graph
    stats = G.graph['pruning_stats']
    assert stats['final_nodes'] <= stats['initial_nodes']

    # Invariants verification
    # 1. Apex must exist and have in_degree 0
    assert 0 in G
    assert G.in_degree(0) == 0

    # 2. Every node must be reachable from apex
    reachable = nx.descendants(G, 0) | {0}
    assert reachable == set(G.nodes())

    # 3. All leaf nodes (out_degree 0) must be legitimate shoreline outlets
    r_shore = G.graph['params']['r_shore']
    shore_threshold = r_shore * 0.95
    for n in G.nodes():
        if G.out_degree(n) == 0:
            assert G.nodes[n]['r'] >= shore_threshold, f"Node {n} is inland leaf at r={G.nodes[n]['r']} < {shore_threshold}"

    # 4. Hydraulics must have executed successfully
    for _, _, data in G.edges(data=True):
        assert 'width' in data and data['width'] > 0
        assert 'conductance' in data and data['conductance'] > 0
        assert 'length' in data and data['length'] > 0

    # 5. Markov flux throughput solver executes successfully and conserves mass
    _C, _W, phi = solve_markov_flux(G, G.graph.get('params'))
    assert len(phi) == G.number_of_nodes()
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0)
    # Unit flux injected at apex (node 0)
    assert math.isclose(phi[0], 1.0, rel_tol=1e-3)


def test_sample_ensemble_with_pruning():
    """Verify ensemble generation functions properly with pruning options enabled."""
    graphs = dgm.sample_ensemble(
        N=40,
        R_rt=0.5,
        num_realizations=4,
        enable_pruning=True,
        simplify_pass_through_nodes=True,
        base_seed=42
    )

    assert len(graphs) == 4
    for G in graphs:
        assert G.graph['enable_pruning'] is True
        assert 'pruning_stats' in G.graph
        # Every leaf must be at shoreline
        r_shore = G.graph['params']['r_shore']
        for n in G.nodes():
            if G.out_degree(n) == 0:
                assert G.nodes[n]['r'] >= r_shore * 0.95


def test_cli_pruning_flags(capsys):
    """Verify CLI accepts pruning arguments."""
    cli_main([
        "generate",
        "-N", "35",
        "-R", "0.5",
        "--enable-pruning",
        "--simplify-chains",
        "--min-spur-length", "100.0",
        "-s", "42"
    ])
    captured = capsys.readouterr()
    assert "=== Generated DGM Realization ===" in captured.out


def test_simplify_hybrid_polyline_coordinates():
    """Verify chain simplification correctly merges one curved edge with one straight edge."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    G.add_node(1, x=2000.0, y=100.0, r=2002.0, theta=0.05)
    G.add_node(2, x=4000.0, y=0.0, r=4000.0, theta=0.0)

    # Edge 0 -> 1 is curved with 3 points
    coords_01 = [[0.0, 0.0], [1000.0, 50.0], [2000.0, 100.0]]
    G.add_edge(0, 1, length=2002.0, straight_distance=2002.0, sinuosity=1.0, edge_type='tree', coordinates=coords_01)
    # Edge 1 -> 2 is straight (no 'coordinates' key)
    G.add_edge(1, 2, length=2002.0, straight_distance=2002.0, sinuosity=1.0, edge_type='tree')

    simplified_G, count = simplify_degree_two_chains(G, r_shore=10000.0)
    assert count == 1
    assert simplified_G.has_edge(0, 2)
    edge_data = simplified_G.edges[0, 2]
    assert 'coordinates' in edge_data
    # Should include coords_01 + endpoint of node 2
    assert edge_data['coordinates'] == [[0.0, 0.0], [1000.0, 50.0], [2000.0, 100.0], [4000.0, 0.0]]


def test_simplify_contracts_to_terminal_outlet():
    """Verify chain simplification contracts pass-through nodes into a direct link to terminal outlet."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    # Near shoreline intermediate pass-through node
    G.add_node(1, x=9600.0, y=0.0, r=9600.0, theta=0.0)
    # Terminal outlet (out=0)
    G.add_node(2, x=10000.0, y=0.0, r=10000.0, theta=0.0, node_type='outlet')

    G.add_edge(0, 1, length=9600.0, straight_distance=9600.0, sinuosity=1.0)
    G.add_edge(1, 2, length=400.0, straight_distance=400.0, sinuosity=1.0)

    simplified_G, count = simplify_degree_two_chains(G, r_shore=10000.0, outlet_tolerance=0.05)
    # Node 1 is contracted, apex 0 and terminal outlet 2 are preserved!
    assert count == 1
    assert 1 not in simplified_G
    assert 0 in simplified_G
    assert 2 in simplified_G
    assert simplified_G.has_edge(0, 2)
    assert simplified_G.edges[0, 2]['length'] == pytest.approx(10000.0)
    assert simplified_G.edges[0, 2]['straight_distance'] == pytest.approx(10000.0)
    assert simplified_G.edges[0, 2]['sinuosity'] == pytest.approx(1.0)


def test_simplify_chains_planarity_guard():
    """Verify simplify_degree_two_chains refuses to contract a bent chain if the chord crosses another edge."""
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0, node_type='apex')
    # Pass-through node 1 bends around obstacle edge (3 -> 4)
    G.add_node(1, x=2000.0, y=2000.0, r=2828.0, theta=0.785)
    G.add_node(2, x=4000.0, y=0.0, r=4000.0, theta=0.0, node_type='outlet')

    # Obstacle edge (3 -> 4) crosses the straight chord (0,0)->(4000,0) at (2000,0)
    G.add_node(3, x=2000.0, y=-500.0, r=2061.0, theta=-0.24)
    G.add_node(4, x=2000.0, y=500.0, r=2061.0, theta=0.24)

    G.add_edge(0, 1, length=2828.0, straight_distance=2828.0, sinuosity=1.0, edge_type='tree')
    G.add_edge(1, 2, length=2828.0, straight_distance=2828.0, sinuosity=1.0, edge_type='tree')
    G.add_edge(3, 4, length=1000.0, straight_distance=1000.0, sinuosity=1.0, edge_type='tree')

    simplified_G, count = simplify_degree_two_chains(G, r_shore=4000.0)
    assert count == 0
    assert 1 in simplified_G
    assert not simplified_G.has_edge(0, 2)



