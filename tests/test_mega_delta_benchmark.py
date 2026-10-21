"""
Unit tests for DGM Mega-Delta Scale Benchmark Suite.
"""

import networkx as nx
import pytest

from dgm.benchmarks.mega_delta_benchmark import (
    MegaDeltaScaleBenchmark,
)
from dgm.core.channel_geometry import compute_edge_clearance
from dgm.core.geometry_utils import segments_intersect
from dgm.generator import DeltaGenerator, EnsembleGenerator


def test_mega_delta_benchmark_instantiation():
    """Verify benchmark recognizes and filters mega-delta dataset targets."""
    bm = MegaDeltaScaleBenchmark(
        delta_names=["Lena", "GBMD"],
        realizations_per_delta=1,
        base_seed=123,
    )
    assert len(bm.target_deltas) == 2
    names = [d.name for d in bm.target_deltas]
    assert "Lena" in names
    assert "GBMD" in names
    for d in bm.target_deltas:
        assert d.nodes >= 50
        assert d.edges > 0


def test_mega_delta_benchmark_execution_and_export(tmp_path):
    """Verify end-to-end execution of benchmark subset and file export formats."""
    # Run a scaled test with small node count to keep CI fast
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

    assert csv_file.exists() and csv_file.stat().st_size > 0
    assert json_file.exists() and json_file.stat().st_size > 0
    assert plot_file.exists() and plot_file.stat().st_size > 0
    assert rpt_file.exists() and rpt_file.stat().st_size > 100


def test_mega_delta_benchmark_invalid_delta_raises():
    """Verify requesting an unknown delta name raises ValueError."""
    with pytest.raises(ValueError, match="No valid mega-deltas found"):
        MegaDeltaScaleBenchmark(delta_names=["NonExistentDelta12345"])


@pytest.mark.parametrize("export_method,ext", [
    ("export_csv", "csv"),
    ("export_json", "json"),
    ("export_plot", "png"),
    ("export_report", "md"),
])
def test_mega_delta_benchmark_unrun_exports_raise(tmp_path, export_method, ext):
    """Verify calling export methods before run() raises RuntimeError."""
    bm = MegaDeltaScaleBenchmark(delta_names=["Niger"], realizations_per_delta=1)
    method = getattr(bm, export_method)
    with pytest.raises(RuntimeError, match="Benchmark has not been run yet"):
        method(tmp_path / f"out.{ext}")


def test_target_nodes_alias_in_generators():
    """Verify target_nodes keyword correctly maps to config node budget N."""
    gen = DeltaGenerator(target_nodes=120)
    assert gen.config.N == 120
    assert "N" not in gen.config.distributions

    ens = EnsembleGenerator(target_nodes=150, num_realizations=2)
    assert ens.N == 150
    assert ens.config.N == 150
    assert "N" not in ens.config.distributions


def test_geometry_acceleration_exactness():
    """Verify AABB bounding box fast rejection and edge clearance bounding box accuracy."""
    # 1. 2D AABB fast rejection edge cases
    assert segments_intersect((0, 0), (1, 1), (10, 10), (11, 11)) is False
    assert segments_intersect((10, 0), (11, 1), (0, 0), (1, 1)) is False
    assert segments_intersect((0, 10), (1, 11), (0, 0), (1, 1)) is False
    assert segments_intersect((0, 0), (1, 1), (0, 10), (1, 11)) is False

    # Crossing segments whose bounding boxes overlap
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)) is True

    # Shared endpoint
    assert segments_intersect((0, 0), (1, 1), (1, 1), (2, 0)) is False

    # Collinear overlapping vs disjoint
    assert segments_intersect((0, 0), (2, 0), (1, 0), (3, 0)) is True
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
