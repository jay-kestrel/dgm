"""
Tests for top-level generator facades, reproducibility, metrics, and ensembles.
"""

import tempfile
from pathlib import Path
import numpy as np
import networkx as nx

import dgm


def test_single_generation_and_reproducibility():
    G1 = dgm.generate(N=30, R_rt=0.4, seed=42)
    G2 = dgm.generate(N=30, R_rt=0.4, seed=42)

    assert isinstance(G1, nx.DiGraph)
    assert G1.number_of_nodes() == G2.number_of_nodes()
    assert G1.number_of_edges() == G2.number_of_edges()

    # Verify edge widths are identical
    for e in G1.edges():
        assert np.isclose(G1.edges[e]['width'], G2.edges[e]['width'])


def test_metrics_evaluation():
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
    graphs = dgm.sample_ensemble(N=25, R_rt=0.3, num_realizations=5, base_seed=10)
    assert len(graphs) == 5
    for g in graphs:
        assert isinstance(g, nx.DiGraph)
        assert g.number_of_nodes() >= 15


def test_ensemble_evaluation_summary():
    ens = dgm.EnsembleGenerator(N=20, R_rt=0.4, num_realizations=4, base_seed=99)
    summary = ens.evaluate_ensemble()

    assert summary['num_realizations'] == 4
    assert 'metrics' in summary
    assert 'alpha_index' in summary['metrics']
    assert 'mean' in summary['metrics']['alpha_index']


def test_io_exports():
    G = dgm.generate(N=25, R_rt=0.5, seed=42)
    battery = dgm.MetricsBattery()
    m = battery.evaluate_all(G)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        graphml_path = tmp_path / "delta.graphml"
        geojson_path = tmp_path / "delta.geojson"
        csv_path = tmp_path / "metrics.csv"

        dgm.to_graphml(G, graphml_path)
        assert graphml_path.exists()
        assert graphml_path.stat().st_size > 0

        dgm.to_geojson(G, geojson_path)
        assert geojson_path.exists()
        assert geojson_path.stat().st_size > 0

        dgm.metrics_to_csv(m, csv_path)
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
