"""
Tests for DGM Empirical Validation Benchmark Suite.
"""

import numpy as np

from dgm.benchmarks.empirical_validation import (
    EmpiricalValidationBenchmark,
    compute_goodness_of_fit,
)


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
    # Verify each benchmark delta has a valid identifier and expected structure
    for d in suite.deltas:
        assert d.name, "Delta name must not be empty"
        assert d.nodes > 0, f"Delta {d.name} must have positive node count"
        assert d.edges > 0, f"Delta {d.name} must have positive edge count"


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

    assert csv_file.exists() and csv_file.stat().st_size > 0
    assert json_file.exists() and json_file.stat().st_size > 0
    assert fig_file.exists() and fig_file.stat().st_size > 0
    assert rpt_file.exists() and rpt_file.stat().st_size > 0

    # Verify JSON content is valid
    import json
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    assert "goodness_of_fit" in data or "benchmark_metadata" in data

    # Verify Markdown report header
    report_text = rpt_file.read_text(encoding="utf-8")
    assert report_text.startswith("#")


def test_broadened_exploration_mode():
    suite = EmpiricalValidationBenchmark(
        realizations_per_delta=3,
        max_nodes=25,
        exploration_mode="broad",
        r_rt_uncertainty=0.10,
        node_jitter=0.10,
        sample_pruning=True,
        base_seed=777,
    )
    suite.deltas = suite.deltas[:2]
    res = suite.run(verbose=False)
    assert "mahalanobis_analysis" in res
    assert "synth_r_rt" in res
    assert len(res["synth_r_rt"]) == 6  # 2 deltas * 3 realizations
    assert res["benchmark_metadata"]["exploration_mode"] == "broad"
    assert res["benchmark_metadata"]["r_rt_uncertainty"] == 0.10

