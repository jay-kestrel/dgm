"""
Unit tests for DGM IO and export utilities (GraphML, GeoJSON, CSV).
"""

import json

import networkx as nx
import numpy as np

from dgm.generator import DeltaGenerator
from dgm.io.export import (
    metrics_to_csv,
    metrics_to_dataframe,
    to_geojson,
    to_graphml,
)


def test_to_graphml(tmp_path):
    gen = DeltaGenerator(N=15, seed=42)
    G = gen.generate()
    out_file = tmp_path / "test.graphml"

    to_graphml(G, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Verify GraphML can be read back and preserves network structure
    G_loaded = nx.read_graphml(str(out_file))
    assert G_loaded.number_of_edges() == G.number_of_edges()
    assert G_loaded.number_of_nodes() >= G.number_of_nodes()



def test_to_geojson(tmp_path):
    gen = DeltaGenerator(N=15, seed=42)
    G = gen.generate()
    # Add numpy array and special set attribute to graph metadata to test serialization robustness
    G.graph["test_array"] = np.array([1.0, 2.0, 3.0])
    G.graph["test_set"] = {"a", "b"}

    out_file = tmp_path / "test.geojson"
    fc = to_geojson(G, out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == G.number_of_nodes() + G.number_of_edges()

    # Re-read JSON to verify valid format
    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == len(fc["features"])


def test_metrics_to_dataframe_and_csv(tmp_path):
    gen = DeltaGenerator(N=15, seed=42)
    G = gen.generate()
    m = gen.evaluate_metrics(G)

    # Test single dictionary conversion
    df1 = metrics_to_dataframe(m)
    assert len(df1) == 1
    assert "alpha_index" in df1.columns

    # Test list of dictionaries conversion
    df2 = metrics_to_dataframe([m, m])
    assert len(df2) == 2

    # Test CSV export
    csv_file = tmp_path / "metrics.csv"
    metrics_to_csv(m, csv_file)
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    import pandas as pd
    df_loaded = pd.read_csv(csv_file)
    assert len(df_loaded) == 1
    assert "alpha_index" in df_loaded.columns
    assert np.isclose(df_loaded["alpha_index"].iloc[0], m["alpha_index"])


def test_summary_to_dataframe_and_csv(tmp_path):
    from dgm.generator import EnsembleGenerator
    from dgm.io.export import summary_to_csv, summary_to_dataframe

    ens = EnsembleGenerator(N=15, num_realizations=3, seed=42)
    summary = ens.evaluate_ensemble()

    df = summary_to_dataframe(summary)
    assert "metric" in df.columns
    assert "mean" in df.columns
    assert "std" in df.columns
    assert len(df) > 0

    csv_file = tmp_path / "ensemble_summary.csv"
    summary_to_csv(summary, csv_file)
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0


def test_export_fallback_without_pandas(tmp_path, monkeypatch):
    """Verify metrics_to_csv and summary_to_csv succeed with standard csv when pandas is absent."""
    import builtins

    from dgm.generator import DeltaGenerator, EnsembleGenerator
    from dgm.io.export import metrics_to_csv, metrics_to_dataframe, summary_to_csv, summary_to_dataframe

    # Simulate missing pandas
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    gen = DeltaGenerator(N=15, seed=42)
    G = gen.generate()
    m = gen.evaluate_metrics(G)

    # DataFrame conversion should raise informative ImportError
    import pytest
    with pytest.raises(ImportError, match="pandas is required"):
        metrics_to_dataframe(m)

    # CSV export should gracefully fall back to csv module and write valid CSV
    csv_file = tmp_path / "metrics_no_pandas.csv"
    metrics_to_csv(m, csv_file)
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "alpha_index" in content

    # Summary DataFrame should also raise ImportError
    ens = EnsembleGenerator(N=15, num_realizations=2, seed=42)
    summary = ens.evaluate_ensemble()
    with pytest.raises(ImportError, match="pandas is required"):
        summary_to_dataframe(summary)

    # Summary CSV should fall back to csv module
    sum_csv = tmp_path / "summary_no_pandas.csv"
    summary_to_csv(summary, sum_csv)
    assert sum_csv.exists()
    sum_content = sum_csv.read_text(encoding="utf-8")
    assert "metric,mean,std" in sum_content



