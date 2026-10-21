"""
Comprehensive tests for empirical validation, benchmark evaluation,
and physical/mathematical invariants of DGM.
"""

import math

import networkx as nx
import numpy as np
import pytest

import dgm
from dgm.core.geometry_utils import segments_intersect
from dgm.core.hydraulics import apply_emergent_hydraulics, solve_markov_flux
from dgm.validation import (
    EMPIRICAL_5D_KEYS,
    EMPIRICAL_5D_LABELS,
    EXPANDED_METRIC_KEYS,
    EXPANDED_METRIC_LABELS,
    FULL_20D_METRIC_KEYS,
    FULL_20D_METRIC_LABELS,
    EmpiricalBenchmark,
    compute_goodness_of_fit,
    compute_ks_test,
    compute_mahalanobis_validation,
    compute_mmd,
    compute_permanova,
    get_empirical_5d_vector,
    get_empirical_vector,
)


def test_validation_keys_and_labels():
    """Verify dimension and naming of 20D and 5D metric key suites."""
    assert len(FULL_20D_METRIC_KEYS) == 20
    assert len(FULL_20D_METRIC_LABELS) == 20
    assert len(EXPANDED_METRIC_KEYS) == 20
    assert len(EXPANDED_METRIC_LABELS) == 20
    assert len(EMPIRICAL_5D_KEYS) == 5
    assert len(EMPIRICAL_5D_LABELS) == 5


def test_canonical_empirical_deltas_data():
    """Verify loading and data integrity of canonical empirical deltas."""
    emp_list = dgm.get_empirical_deltas()
    emp_dict = dgm.get_empirical_deltas_dict()

    assert len(emp_list) == 28
    assert len(emp_dict) == 28

    # Verify Mississippi, Wax Lake, and Yukon values match expected benchmark
    expected = {
        'Mississippi': (142, 149, 8),
        'Wax Lake': (105, 124, 20),
        'Yukon': (500, 680, 181),
    }
    for name, (nodes, edges, loops) in expected.items():
        assert name in emp_dict
        d = emp_dict[name]
        assert d['nodes'] == nodes
        assert d['edges'] == edges
        assert d['loops'] == loops
        assert d['alpha'] > 0.0

    # Verify feature vector extraction
    d_first = emp_list[0]
    vec_20d = get_empirical_vector(d_first)
    assert len(vec_20d) == 20
    vec_5d = get_empirical_5d_vector(d_first)
    assert len(vec_5d) == 5


def test_goodness_of_fit():
    """Verify quantitative goodness-of-fit error calculations."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.05, 1.95, 3.02, 3.98, 5.01])

    gof = compute_goodness_of_fit(y_true, y_pred)
    assert 'mae' in gof
    assert 'rmse' in gof
    assert 'nrmse' in gof
    assert 'r2' in gof
    assert gof['r2'] > 0.99
    assert gof['mae'] < 0.06


def test_ks_test():
    """Verify Kolmogorov-Smirnov 2-sample statistical test."""
    rng = np.random.default_rng(42)
    s1 = rng.normal(0.0, 1.0, size=50)
    s2 = rng.normal(0.0, 1.0, size=50)
    res = compute_ks_test(s1, s2)
    assert 'ks_stat' in res
    assert 'p_value' in res
    assert 0.0 <= res['ks_stat'] <= 1.0
    assert 0.0 <= res['p_value'] <= 1.0


def test_mahalanobis_validation():
    """Verify Mahalanobis hyper-ellipsoid confidence validation."""
    rng = np.random.default_rng(42)
    synth = rng.multivariate_normal(mean=[0, 0, 0], cov=np.eye(3), size=100)
    emp = rng.multivariate_normal(mean=[0, 0, 0], cov=np.eye(3), size=10)

    res = compute_mahalanobis_validation(synth, emp, df=3)
    assert 'coverage_ratio' in res
    assert 'inside_95_ci' in res
    assert res['coverage_ratio'] >= 0.50
    assert len(res['distances']) == 10


def test_mmd_and_permanova():
    """Verify Maximum Mean Discrepancy and PERMANOVA test implementations."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(20, 3))
    Y = rng.normal(0, 1, size=(10, 3))

    mmd_val = compute_mmd(X, Y)
    assert mmd_val >= 0.0

    perm_res = compute_permanova(X, Y, num_permutations=20)
    assert 'pseudo_f' in perm_res
    assert 'p_value' in perm_res
    assert perm_res['num_permutations'] == 20


@pytest.mark.parametrize("r_rt", [0.0, 0.5, 1.0])
def test_planarity_zero_crossings(r_rt):
    """Verify geometric planarity filter guarantees 0 edge crossings across forcing regimes."""
    G = dgm.generate(N=25, R_rt=r_rt, seed=42)
    edges = [
        ((G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y']))
        for u, v in G.edges()
    ]
    crossings = sum(
        1 for i in range(len(edges))
        for j in range(i + 1, len(edges))
        if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1])
    )
    assert crossings == 0, f"Found {crossings} crossings for R_rt={r_rt}"


@pytest.mark.parametrize("mech", ["potential_field", "ner_optimality"])
def test_flux_mass_conservation(mech):
    """Verify Markov steady-state flux conservation: apex=1.0 and sum(outlets)=1.0."""
    G = dgm.generate(N=25, R_rt=0.4, loop_mechanism=mech, seed=123)
    _C, _W, phi = solve_markov_flux(G)
    nodes = list(G.nodes())

    apex_nodes = [n for n in nodes if G.nodes[n].get('node_type') == 'apex']
    apex_idx = nodes.index(apex_nodes[0]) if apex_nodes else 0
    assert math.isclose(phi[apex_idx], 1.0, abs_tol=1e-5)

    outlet_indices = [idx for idx, n in enumerate(nodes) if G.out_degree(n) == 0]
    outlet_flux_sum = sum(phi[idx] for idx in outlet_indices)
    assert math.isclose(outlet_flux_sum, 1.0, abs_tol=1e-5)



def test_murrays_law_cyclic_relaxation():
    """Verify iterative relaxation preserves Murray's Law width conservation on cyclic networks."""
    G = nx.DiGraph()
    G.add_node(0, r=0.0, theta=0.0, x=0.0, y=0.0, node_type='apex')
    G.add_node(1, r=1000.0, theta=0.0, x=1000.0, y=0.0, node_type='bifurcation')
    G.add_node(2, r=2000.0, theta=0.1, x=2000.0, y=200.0, node_type='internal')
    G.add_node(3, r=3000.0, theta=0.2, x=3000.0, y=400.0, node_type='confluence')
    G.add_node(4, r=4000.0, theta=0.0, x=4000.0, y=0.0, node_type='outlet')
    G.add_node(5, r=2000.0, theta=-0.1, x=2000.0, y=-200.0, node_type='outlet')

    G.add_edge(0, 1, edge_type='tree')
    G.add_edge(1, 2, edge_type='tree')
    G.add_edge(2, 3, edge_type='tree')
    G.add_edge(3, 4, edge_type='tree')
    G.add_edge(1, 5, edge_type='tree')
    # Cyclic feedback loop
    G.add_edge(3, 1, edge_type='loop')

    assert not nx.is_directed_acyclic_graph(G)

    params = {
        'apex_width': 200.0,
        'murray_p': 1.85,
        'width_beta': 0.65,
        'R_rt': 0.3,
        'r_shore': 5000.0,
        'min_channel_width': 1.0,
        'relaxation_max_iter': 50,
        'relaxation_tol': 1e-5,
    }
    G_hyd = apply_emergent_hydraulics(G, params=params, seed=42)

    p_exp = params['murray_p']
    in_edges_1 = list(G_hyd.in_edges(1, data=True))
    w_in_sum_1 = sum(d['width_river'] ** p_exp for _, _, d in in_edges_1)
    w_in_equiv_1 = w_in_sum_1 ** (1.0 / p_exp)

    out_edges_1 = list(G_hyd.out_edges(1, data=True))
    w_out_sum_1 = sum(d['width_river'] ** p_exp for _, _, d in out_edges_1)
    w_out_equiv_1 = w_out_sum_1 ** (1.0 / p_exp)

    ratio = w_out_equiv_1 / w_in_equiv_1
    assert 0.70 <= ratio <= 1.30


@pytest.mark.parametrize("mode", ["width", "conductance"])
def test_routing_mode_conductance_vs_width(mode):
    """Verify both width and conductance hydraulic routing modes run cleanly."""
    cfg = dgm.DGMConfig(N=20, R_rt=0.3, routing_mode=mode, seed=42)
    gen = dgm.DeltaGenerator(cfg)
    G = gen.generate()
    m = gen.evaluate_metrics(G)
    assert m['nER_star'] >= 0.0
    assert m['eBI'] >= 1.0


def test_non_deterministic_seeding():
    """Verify seed=None produces distinct realizations across invocations."""
    g1 = dgm.generate(N=20, R_rt=0.3, seed=None)
    g2 = dgm.generate(N=20, R_rt=0.3, seed=None)
    # Check that either node count, edge positions, or widths differ
    w1 = [d['width'] for _, _, d in g1.edges(data=True)]
    w2 = [d['width'] for _, _, d in g2.edges(data=True)]
    assert (
        len(w1) != len(w2)
        or set(g1.edges()) != set(g2.edges())
        or not np.allclose(w1, w2, atol=1e-3)
    )


@pytest.mark.parametrize("config_kwargs,match_pattern", [
    ({"branching_weights": (0.5, 0.5, 0.5)}, "branching_weights must sum to"),
    ({"branching_weights": (0.5, 0.5)}, "branching_weights must be a 3-tuple"),
    ({"sa_cooling_rate": 0.0}, "sa_cooling_rate must be in"),
    ({"sa_cooling_rate": 1.0}, "sa_cooling_rate must be in"),
    ({"routing_mode": "invalid_mode"}, "routing_mode must be"),
])

def test_config_invariants_and_validation(config_kwargs, match_pattern):
    """Verify validation guards on branching weights, cooling rate, and routing mode raise ValueError."""
    with pytest.raises(ValueError, match=match_pattern):
        dgm.DGMConfig(**config_kwargs)



def test_parameter_bounds_sampling():
    """Verify ParameterBounds sampling respects bounds for backwards-compatibility."""
    bounds = dgm.ParameterBounds()
    rng = np.random.default_rng(42)
    sampled = bounds.sample(rng)

    assert 1.2 <= sampled['murray_p'] <= 2.2
    assert 0.50 <= sampled['width_beta'] <= 0.99
    assert 1.50 <= sampled['manning_eta'] <= 1.80
    assert 0.40 <= sampled['length_gamma'] <= 0.60
    assert 50.0 <= sampled['apex_width'] <= 500.0
    assert 5000.0 <= sampled['r_shore'] <= 15000.0
    assert bounds.theta_base_range[0] <= sampled['theta_base'] <= bounds.theta_base_range[1]
    assert bounds.theta_delta_range[0] <= sampled['theta_delta'] <= bounds.theta_delta_range[1]


def test_empirical_benchmark_run():
    """Verify EmpiricalBenchmark runs and returns complete statistical output."""
    bench = EmpiricalBenchmark(realizations_per_delta=1)
    bench.empirical_data = bench.empirical_data[:2]
    res = bench.run_benchmark(loop_mechanism="potential_field", sample_parameters=True)

    assert "goodness_of_fit" in res
    assert "overall_goodness_of_fit" in res
    assert "mahalanobis" in res
    assert "mmd" in res
    assert "permanova" in res
    assert len(res["detailed_results"]) == 2

