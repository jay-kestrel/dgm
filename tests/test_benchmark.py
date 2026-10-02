"""
Tests for DGM Empirical Validation Benchmark Suite.
"""

from pathlib import Path
import pytest
from dgm.benchmarks.empirical_validation import (
    EmpiricalValidationBenchmark,
    compute_goodness_of_fit,
    PRIMARY_METRICS,
)
import numpy as np


def test_compute_goodness_of_fit():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    gof = compute_goodness_of_fit(y_true, y_pred)

    assert "mae" in gof
    assert "rmse" in gof
    assert "nrmse" in gof
    assert "r2" in gof
    assert gof["r2"] > 0.95
    assert gof["mae"] < 0.2


def test_benchmark_suite_instantiation():
    suite = EmpiricalValidationBenchmark(
        realizations_per_delta=2,
        max_nodes=30,
        base_seed=123,
    )
    assert len(suite.deltas) == 28
    assert suite.realizations_per_delta == 2
    assert suite.max_nodes == 30


def test_benchmark_suite_subset_run(tmp_path):
    # Test execution on a small 2-delta subset for quick test suite validation
    suite = EmpiricalValidationBenchmark(
        realizations_per_delta=2,
        max_nodes=25,
        base_seed=999,
    )
    # Temporarily slice deltas to 2 for speedy CI test
    suite.deltas = suite.deltas[:2]

    res = suite.run(verbose=False)
    assert "goodness_of_fit" in res
    assert "detailed_results" in res
    assert len(res["detailed_results"]) == 2

    # Test file exports
    csv_file = tmp_path / "test_results.csv"
    json_file = tmp_path / "test_summary.json"
    fig_file = tmp_path / "test_figure.png"
    rpt_file = tmp_path / "test_report.md"

    suite.save_csv(csv_file)
    suite.save_json(json_file)
    suite.plot_benchmark(fig_file)
    suite.generate_report(rpt_file)

    assert csv_file.exists()
    assert json_file.exists()
    assert fig_file.exists()
    assert rpt_file.exists()
    assert csv_file.stat().st_size > 0
    assert json_file.stat().st_size > 0
    assert fig_file.stat().st_size > 0
    assert rpt_file.stat().st_size > 0
