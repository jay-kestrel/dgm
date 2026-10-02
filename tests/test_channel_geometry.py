"""
Tests for curved channel centerline generation and proximity-constrained meander dynamics.
"""

import math
import tempfile
from pathlib import Path
import json
import numpy as np
import networkx as nx
import pytest

import dgm
from dgm.config import DGMConfig, ParameterBounds
from dgm.core.channel_geometry import (
    point_to_segment_distance,
    compute_edge_clearance,
    apply_channel_proximity_constraints,
    generate_channel_centerlines,
)


def test_point_to_segment_distance():
    """Verify orthogonal projection, clamp to endpoints, and zero-length segment handling."""
    a = (0.0, 0.0)
    b = (10.0, 0.0)

    # Point directly above interior of segment
    p1 = (5.0, 4.0)
    assert math.isclose(point_to_segment_distance(p1, a, b), 4.0, abs_tol=1e-6)

    # Point beyond endpoint B
    p2 = (13.0, 4.0)
    # Distance to (10, 0) is sqrt(3^2 + 4^2) = 5.0
    assert math.isclose(point_to_segment_distance(p2, a, b), 5.0, abs_tol=1e-6)

    # Point behind endpoint A
    p3 = (-3.0, 4.0)
    # Distance to (0, 0) is sqrt(3^2 + 4^2) = 5.0
    assert math.isclose(point_to_segment_distance(p3, a, b), 5.0, abs_tol=1e-6)

    # Point directly on segment
    p4 = (7.0, 0.0)
    assert math.isclose(point_to_segment_distance(p4, a, b), 0.0, abs_tol=1e-6)

    # Degenerate segment
    assert math.isclose(point_to_segment_distance((3.0, 4.0), a, a), 5.0, abs_tol=1e-6)


def test_compute_edge_clearance():
    """Verify clearance to neighboring channels correctly excludes incident edges."""
    G = nx.DiGraph()
    # Channel 1: (0, 0) -> (1000, 0)
    G.add_node(0, x=0.0, y=0.0, r=0.0, theta=0.0)
    G.add_node(1, x=1000.0, y=0.0, r=1000.0, theta=0.0)
    G.add_edge(0, 1, straight_distance=1000.0, sinuosity=1.3)

    # Incident branch sharing node 0: (0, 0) -> (1000, 500)
    G.add_node(2, x=1000.0, y=500.0, r=1118.0, theta=0.46)
    G.add_edge(0, 2, straight_distance=1118.0, sinuosity=1.2)

    # Clearance of 0->1 with only incident edge 0->2 should be infinity
    c_isolated = compute_edge_clearance(G, 0, 1)
    assert math.isinf(c_isolated)

    # Add parallel non-incident channel: (0, 200) -> (1000, 200)
    G.add_node(3, x=0.0, y=200.0, r=200.0, theta=math.pi / 2)
    G.add_node(4, x=1000.0, y=200.0, r=1020.0, theta=0.2)
    G.add_edge(3, 4, straight_distance=1000.0, sinuosity=1.2)

    # Midpoint of 0->1 is (500, 0). Distance to segment 3->4 is 200.0
    c_parallel = compute_edge_clearance(G, 0, 1)
    assert math.isclose(c_parallel, 200.0, abs_tol=1e-4)


def test_proximity_constrains_sinuosity():
    """Verify that when channels are close, meander amplitude and sinuosity are capped."""
    G = nx.DiGraph()
    # Two parallel channels only 50m apart with long length (1000m)
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=1000.0, y=0.0)
    G.add_edge(0, 1, straight_distance=1000.0, sinuosity=1.5)

    G.add_node(2, x=0.0, y=50.0)
    G.add_node(3, x=1000.0, y=50.0)
    G.add_edge(2, 3, straight_distance=1000.0, sinuosity=1.5)

    # Clearance is 50m. With clearance_fraction = 0.40, max amplitude is 20m.
    # S_max = 1 + (pi * 20 / 1000)^2 = 1 + 0.0039 = 1.0039 << 1.5!
    apply_channel_proximity_constraints(G, clearance_fraction=0.40)

    s1 = G.edges[0, 1]['sinuosity']
    s2 = G.edges[2, 3]['sinuosity']
    assert s1 < 1.05
    assert s2 < 1.05
    assert s1 >= 1.0
    assert s2 >= 1.0
    assert G.edges[0, 1]['clearance_distance'] == 50.0


def test_curved_centerline_geometry():
    """Verify polyline endpoint attachment, coordinate dimensions, and meander amplitude."""
    G = nx.DiGraph()
    G.add_node(0, x=100.0, y=200.0)
    G.add_node(1, x=1100.0, y=200.0)
    G.add_edge(0, 1, straight_distance=1000.0, sinuosity=1.25)

    generate_channel_centerlines(G, num_points=25, num_bends=1, seed=42)

    coords = G.edges[0, 1]['coordinates']
    assert len(coords) == 25

    # Endpoints must attach exactly to nodes
    assert math.isclose(coords[0][0], 100.0, abs_tol=1e-5)
    assert math.isclose(coords[0][1], 200.0, abs_tol=1e-5)
    assert math.isclose(coords[-1][0], 1100.0, abs_tol=1e-5)
    assert math.isclose(coords[-1][1], 200.0, abs_tol=1e-5)

    # Interior points meander off the baseline y=200
    y_vals = [pt[1] for pt in coords]
    max_dev = max(abs(y - 200.0) for y in y_vals)
    assert max_dev > 10.0

    # Polyline arc length must be strictly greater than Euclidean chord
    poly_len = sum(
        math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
        for i in range(1, len(coords))
    )
    assert poly_len > 1000.0


def test_config_channel_geometry_validation():
    """Verify validation bounds for channel geometry parameters."""
    cfg = DGMConfig()
    assert cfg.enable_curved_centerlines is False
    assert cfg.min_channel_clearance == 0.40
    assert cfg.num_polyline_points == 21
    assert cfg.meander_bends == 1

    # Invalid clearance fraction
    with pytest.raises(ValueError, match="min_channel_clearance must be in"):
        DGMConfig(min_channel_clearance=0.01)
    with pytest.raises(ValueError, match="min_channel_clearance must be in"):
        DGMConfig(min_channel_clearance=1.5)

    # Invalid polyline points
    with pytest.raises(ValueError, match="num_polyline_points must be >= 3"):
        DGMConfig(num_polyline_points=2)

    # Invalid meander bends
    with pytest.raises(ValueError, match="meander_bends must be >= 1"):
        DGMConfig(meander_bends=0)


def test_generator_with_curved_centerlines():
    """Verify single generation with enable_curved_centerlines=True attaches coordinates to all edges."""
    G = dgm.generate(N=30, R_rt=0.5, enable_curved_centerlines=True, seed=42)
    assert isinstance(G, nx.DiGraph)
    assert G.graph['enable_curved_centerlines'] is True

    for u, v, d in G.edges(data=True):
        assert 'coordinates' in d
        assert len(d['coordinates']) >= 3
        # First point equals node u, last point equals node v
        assert math.isclose(d['coordinates'][0][0], G.nodes[u]['x'], abs_tol=1e-4)
        assert math.isclose(d['coordinates'][0][1], G.nodes[u]['y'], abs_tol=1e-4)
        assert math.isclose(d['coordinates'][-1][0], G.nodes[v]['x'], abs_tol=1e-4)
        assert math.isclose(d['coordinates'][-1][1], G.nodes[v]['y'], abs_tol=1e-4)


def test_geojson_export_with_curved_centerlines(tmp_path):
    """Verify GeoJSON export includes multi-point LineString geometries when curved centerlines are enabled."""
    G = dgm.generate(N=25, R_rt=0.4, enable_curved_centerlines=True, seed=99)
    out_file = tmp_path / "delta_curved.geojson"
    dgm.to_geojson(G, out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    channel_features = [feat for feat in data["features"] if feat["properties"].get("feature_type") == "channel"]
    assert len(channel_features) == G.number_of_edges()

    # Verify that LineStrings have multiple coordinate vertices (> 2)
    for feat in channel_features:
        coords = feat["geometry"]["coordinates"]
        assert len(coords) >= 3
        assert "sinuosity" in feat["properties"]


def test_cli_curved_flag(tmp_path):
    """Verify CLI accepts --curved and produces valid output."""
    from dgm.cli import main
    out_file = tmp_path / "cli_curved.geojson"

    main([
        "generate",
        "-N", "25",
        "-R", "0.4",
        "--curved",
        "--min-channel-clearance", "0.35",
        "-s", "42",
        "-o", str(out_file)
    ])

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    channel_features = [feat for feat in data["features"] if feat["properties"].get("feature_type") == "channel"]
    assert len(channel_features) > 0
    assert len(channel_features[0]["geometry"]["coordinates"]) >= 3
