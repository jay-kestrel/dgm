"""
Unit tests for DGM Mega-Delta Scale Benchmark Suite.
"""

from pathlib import Path
import pytest
import numpy as np

from dgm.benchmarks.mega_delta_benchmark import (
    MegaDeltaScaleBenchmark,
    DEFAULT_MEGA_DELTAS,
)


def test_mega_delta_benchmark_instantiation():
    bm = MegaDeltaScaleBenchmark(
        delta_names=["Lena", "GBMD"],
        realizations_per_delta=1,
        base_seed=123,
    )
    assert len(bm.target_deltas) == 2
    names = [d.name for d in bm.target_deltas]
    assert "Lena" in names
    assert "GBMD" in names
    assert bm.realizations_per_delta == 1


def test_mega_delta_benchmark_execution_and_export(tmp_path):
    # Run a scaled test with small node count to keep CI ultra fast
    bm = MegaDeltaScaleBenchmark(
        delta_names=["Niger"],
        realizations_per_delta=1,
        base_seed=42,
    )
    # Temporarily set small node count on the target delta for rapid unit testing
    bm.target_deltas[0].nodes = 35

    res = bm.run(verbose=False)
    assert "benchmark_metadata" in res
    assert "per_delta_summary" in res
    assert len(res["per_delta_summary"]) == 1

    summary = res["per_delta_summary"][0]
    assert summary["delta_name"] == "Niger"
    assert "synthetic_alpha_mean" in summary
    assert "synthetic_nER_star_mean" in summary
    assert "synthetic_eBI_mean" in summary

    # Verify file exports
    csv_file = tmp_path / "test_mega.csv"
    json_file = tmp_path / "test_mega.json"
    plot_file = tmp_path / "test_mega.png"
    rpt_file = tmp_path / "test_mega.md"

    bm.export_csv(csv_file)
    bm.export_json(json_file)
    bm.export_plot(plot_file)
    bm.export_report(rpt_file)

    assert csv_file.exists()
    assert json_file.exists()
    assert plot_file.exists()
    assert rpt_file.exists()
    assert rpt_file.stat().st_size > 100


def test_mega_delta_benchmark_guards_and_errors(tmp_path):
    # Test invalid delta name raises ValueError
    with pytest.raises(ValueError, match="No valid mega-deltas found"):
        MegaDeltaScaleBenchmark(delta_names=["NonExistentDelta12345"])

    # Test export methods before run() raise RuntimeError
    bm = MegaDeltaScaleBenchmark(delta_names=["Niger"], realizations_per_delta=1)
    with pytest.raises(RuntimeError, match="Benchmark has not been run yet"):
        bm.export_csv(tmp_path / "out.csv")
    with pytest.raises(RuntimeError, match="Benchmark has not been run yet"):
        bm.export_json(tmp_path / "out.json")
    with pytest.raises(RuntimeError, match="Benchmark has not been run yet"):
        bm.export_plot(tmp_path / "out.png")
    with pytest.raises(RuntimeError, match="Benchmark has not been run yet"):
        bm.export_report(tmp_path / "out.md")


def test_target_nodes_alias_in_generators():
    from dgm.generator import DeltaGenerator, EnsembleGenerator

    gen = DeltaGenerator(target_nodes=120)
    assert gen.config.N == 120
    assert "N" not in gen.config.distributions

    ens = EnsembleGenerator(target_nodes=150, num_realizations=2)
    assert ens.N == 150
    assert ens.config.N == 150
    assert "N" not in ens.config.distributions


def test_geometry_acceleration_exactness():
    from dgm.core.geometry_utils import segments_intersect
    from dgm.core.channel_geometry import compute_edge_clearance
    import networkx as nx

    # 1. 2D AABB fast rejection edge cases
    # Segments with completely disjoint bounding boxes (AABB should fast-reject)
    assert segments_intersect((0, 0), (1, 1), (10, 10), (11, 11)) is False
    assert segments_intersect((10, 0), (11, 1), (0, 0), (1, 1)) is False
    assert segments_intersect((0, 10), (1, 11), (0, 0), (1, 1)) is False
    assert segments_intersect((0, 0), (1, 1), (0, 10), (1, 11)) is False

    # Crossing segments whose bounding boxes overlap
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)) is True

    # Shared endpoint (should return False)
    assert segments_intersect((0, 0), (1, 1), (1, 1), (2, 0)) is False

    # Collinear overlapping
    assert segments_intersect((0, 0), (2, 0), (1, 0), (3, 0)) is True

    # Collinear disjoint
    assert segments_intersect((0, 0), (1, 0), (2, 0), (3, 0)) is False

    # 2. Clearance bounding-box rejection exactness
    G = nx.DiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=10.0, y=0.0)
    G.add_edge(0, 1)

    # Edge far away (bbox distance > 100)
    G.add_node(2, x=0.0, y=100.0)
    G.add_node(3, x=10.0, y=100.0)
    G.add_edge(2, 3)

    # Edge closer (bbox distance = 20)
    G.add_node(4, x=0.0, y=20.0)
    G.add_node(5, x=10.0, y=20.0)
    G.add_edge(4, 5)

    c = compute_edge_clearance(G, 0, 1)
    assert pytest.approx(c, abs=1e-4) == 20.0
