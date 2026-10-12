"""
Unit tests for DGM IO and export utilities (GraphML, GeoJSON, CSV).
"""

import json
import networkx as nx
import numpy as np

from dgm.generator import DeltaGenerator
from dgm.io.export import (
    to_graphml,
    to_geojson,
    metrics_to_dataframe,
    metrics_to_csv,
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
    with open(out_file, "r", encoding="utf-8") as f:
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

