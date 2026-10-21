"""
Tests for empirical delta dataset structures and loaders.
"""

import json

import pytest

from dgm.io.datasets import (
    DELTA_FORCING_REGIMES,
    DIDNTLoader,
)


def test_delta_forcing_regimes_table():
    """Verify completeness and required attributes of empirical delta forcing table."""
    assert len(DELTA_FORCING_REGIMES) == 28
    assert "Mississippi" in DELTA_FORCING_REGIMES
    assert "Niger" in DELTA_FORCING_REGIMES
    assert DELTA_FORCING_REGIMES["Mississippi"]["forcing_type"] == "river_dominated"
    assert DELTA_FORCING_REGIMES["Niger"]["forcing_type"] == "tide_influenced"

    # Verify every entry has valid regime and numerical bounds
    for name, data in DELTA_FORCING_REGIMES.items():
        assert "forcing_type" in data, f"Missing forcing_type in {name}"
        assert "estimated_R_rt" in data, f"Missing estimated_R_rt in {name}"
        assert 0.0 <= data["estimated_R_rt"] <= 1.0, f"estimated_R_rt out of bounds for {name}: {data['estimated_R_rt']}"



def test_didnt_loader_unavailable_handling(tmp_path):
    """Verify DIDNTLoader gracefully reports unavailability and raises FileNotFoundError on load attempt."""
    empty_dir = tmp_path / "nonexistent_didnt"
    loader = DIDNTLoader(dataset_dir=empty_dir)

    assert loader.is_available() is False
    assert loader.list_available_deltas() == []

    with pytest.raises(FileNotFoundError, match="DIDNT dataset directory not found"):
        loader.load_graph("NonExistentDelta")


def test_didnt_loader_mock_dataset(tmp_path):
    """Verify DIDNTLoader successfully parses GeoJSON node/link networks into a DiGraph."""
    nnl_dir = tmp_path / "NetworkNodesAndLinks"
    nnl_dir.mkdir(parents=True)

    nodes_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": 0,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "properties": {"id": 0, "distance_from_apex": 0.0, "type": "inlet"},
            },
            {
                "id": 1,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [5000.0, 500.0]},
                "properties": {"id": 1, "distance_from_apex": 5024.0, "type": "junction"},
            },
            {
                "id": 2,
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10000.0, 1000.0]},
                "properties": {"id": 2, "distance_from_apex": 10050.0, "type": "outlet"},
            },
        ],
    }

    links_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [5000.0, 500.0]]},
                "properties": {"source": 0, "target": 1, "length": 5024.0, "width": 150.0},
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[5000.0, 500.0], [10000.0, 1000.0]]},
                "properties": {"source": 1, "target": 2, "length": 5026.0, "width": 100.0},
            },
        ],
    }

    (nnl_dir / "testdelta_nodes.json").write_text(json.dumps(nodes_data), encoding="utf-8")
    (nnl_dir / "testdelta_links.json").write_text(json.dumps(links_data), encoding="utf-8")

    loader = DIDNTLoader(dataset_dir=tmp_path)
    assert loader.is_available() is True
    available = loader.list_available_deltas()
    assert "testdelta" in available

    G = loader.load_graph("testdelta")
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.nodes[0]["node_type"] == "apex"
    assert G.nodes[2]["node_type"] == "outlet"
    assert G.has_edge(0, 1)
    assert G.has_edge(1, 2)


def test_lazy_canonical_empirical_deltas():
    """Verify CANONICAL_EMPIRICAL_DELTAS behaves as a full dictionary and loads lazily on demand."""
    from dgm.io.datasets import CANONICAL_EMPIRICAL_DELTAS, _LazyEmpiricalDeltasDict

    lazy_dict = _LazyEmpiricalDeltasDict()
    assert lazy_dict._loaded is False
    assert isinstance(lazy_dict, dict)
    assert len(lazy_dict) == 28
    assert lazy_dict._loaded is True
    assert "Mississippi" in CANONICAL_EMPIRICAL_DELTAS
    assert CANONICAL_EMPIRICAL_DELTAS["Mississippi"]["nodes"] > 0

