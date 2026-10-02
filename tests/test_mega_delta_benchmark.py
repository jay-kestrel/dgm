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
