"""
Tests for top-level generator facades, reproducibility, metrics, and ensembles.
"""

from pathlib import Path
import numpy as np
import networkx as nx

import dgm


def test_single_generation_and_reproducibility():
    """Verify deterministic generator facade reproduces exact node/edge topology and hydraulic widths."""
    G1 = dgm.generate(N=30, R_rt=0.4, seed=42)
    G2 = dgm.generate(N=30, R_rt=0.4, seed=42)

    assert G1.number_of_nodes() == G2.number_of_nodes()
    assert G1.number_of_edges() == G2.number_of_edges()
    assert set(G1.nodes()) == set(G2.nodes())
    assert set(G1.edges()) == set(G2.edges())

    # Verify edge widths are identical
    for e in G1.edges():
        assert np.isclose(G1.edges[e]['width'], G2.edges[e]['width'])


def test_metrics_evaluation():
    """Verify MetricsBattery computes valid, bounded morphological and graph metrics."""
    G = dgm.generate(N=30, R_rt=0.5, seed=123)
    battery = dgm.MetricsBattery()
    metrics = battery.evaluate_all(G)

    assert 'alpha_index' in metrics
    assert 'nER_star' in metrics
    assert 'eBI' in metrics
    assert 'ratio_vc_vb' in metrics
    assert 'betti_1' in metrics

    # Value ranges
    assert 0.0 <= metrics['alpha_index'] <= 1.0
    assert 0.0 <= metrics['nER_star'] <= 1.0
    assert metrics['eBI'] >= 1.0


def test_ensemble_sampling():
    """Verify sample_ensemble returns connected realizations meeting minimum node budget."""
    graphs = dgm.sample_ensemble(N=25, R_rt=0.3, num_realizations=5, base_seed=10)
    assert len(graphs) == 5
    for g in graphs:
        assert g.number_of_nodes() >= 15
        assert nx.is_weakly_connected(g)
        assert 0 in g
        assert g.nodes[0]['node_type'] == 'apex'


def test_ensemble_evaluation_summary():
    """Verify EnsembleGenerator compiles metric distribution statistics across realizations."""
    ens = dgm.EnsembleGenerator(N=20, R_rt=0.4, num_realizations=4, base_seed=99)
    summary = ens.evaluate_ensemble()

    assert summary['num_realizations'] == 4
    assert 'metrics' in summary
    assert 'alpha_index' in summary['metrics']
    assert 'mean' in summary['metrics']['alpha_index']
    assert 'std' in summary['metrics']['alpha_index']


def test_top_level_io_exports(tmp_path):
    """Verify top-level package facade export functions write non-empty output files using tmp_path."""
    G = dgm.generate(N=25, R_rt=0.5, seed=42)
    battery = dgm.MetricsBattery()
    m = battery.evaluate_all(G)

    graphml_path = tmp_path / "delta.graphml"
    geojson_path = tmp_path / "delta.geojson"
    csv_path = tmp_path / "metrics.csv"

    dgm.to_graphml(G, graphml_path)
    assert graphml_path.exists() and graphml_path.stat().st_size > 0

    dgm.to_geojson(G, geojson_path)
    assert geojson_path.exists() and geojson_path.stat().st_size > 0

    dgm.metrics_to_csv(m, csv_path)
    assert csv_path.exists() and csv_path.stat().st_size > 0


def test_sampled_N_and_R_rt_respected_in_generator():
    """Verify sampled N and R_rt from config distributions are used by DeltaGenerator."""
    cfg = dgm.DGMConfig(
        N=20,
        R_rt=0.1,
        sample_parameters=True,
        seed=42,
        distributions={
            "N": {"type": "fixed", "value": 42},
            "R_rt": {"type": "fixed", "value": 0.85},
        },
    )
    gen = dgm.DeltaGenerator(cfg)
    G = gen.generate(seed_override=42)

    assert G.graph["N_input"] == 42
    assert np.isclose(G.graph["R_rt"], 0.85)
    assert G.graph["params"]["N"] == 42
    assert np.isclose(G.graph["params"]["R_rt"], 0.85)


def test_no_global_runtime_warning_filter():
    """Verify importing dgm does not inject a global ignore filter for RuntimeWarning."""
    import warnings
    for f in warnings.filters:
        action, _msg, category, module, _lineno = f
        if action == "ignore" and category is RuntimeWarning and not module:
            raise AssertionError(f"Found global RuntimeWarning ignore filter: {f}")

