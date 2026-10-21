"""
Tests for Jordan Curve Theorem planarity filtering and edge intersections.
"""

import math

import networkx as nx
import pytest

from dgm.core.domain import PolarDomain
from dgm.core.geometry_utils import (
    edge_intersects_graph,
    extend_leaf_to_shoreline,
    segments_intersect,
)


@pytest.mark.parametrize(
    "p1,p2,q1,q2,expected,description",
    [
        ((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0), True, "Crossing X segments"),
        ((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0), False, "Parallel disjoint lines"),
        ((0.0, 0.0), (5.0, 5.0), (5.0, 5.0), (10.0, 0.0), False, "Shared endpoint junction"),
        ((0.0, 0.0), (1.0, 1.0), (10.0, 10.0), (11.0, 11.0), False, "Disjoint bounding boxes"),
        ((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (3.0, 0.0), True, "Collinear overlapping"),
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), False, "Collinear separated"),
    ],
)
def test_segments_intersect(p1, p2, q1, q2, expected, description):
    """Verify segment intersection detection across standard, degenerate, and boundary configurations."""
    assert segments_intersect(p1, p2, q1, q2) is expected, f"Failed on: {description}"


@pytest.mark.parametrize(
    "u,v,expected,description",
    [
        ((0.0, 10.0), (10.0, 0.0), True, "Crosses existing segment"),
        ((50.0, 50.0), (60.0, 60.0), False, "Completely disjoint edge"),
        ((10.0, 10.0), (10.0, 20.0), False, "Shares existing junction vertex without interior crossing"),
    ],
)
def test_edge_intersects_graph(u, v, expected, description):
    """Verify edge intersection against a collection of existing graph edges."""
    existing = [
        ((0.0, 0.0), (10.0, 10.0)),
        ((10.0, 10.0), (20.0, 10.0)),
    ]
    assert edge_intersects_graph(u, v, existing) is expected, f"Failed on: {description}"


def test_extend_leaf_to_shoreline_confluence():
    """Verify extend_leaf_to_shoreline updates all predecessor edges when extending a confluent leaf."""
    domain = PolarDomain(r_shore=1000.0, theta_base=90.0, theta_delta=60.0)
    G = nx.DiGraph()
    G.add_node(0, r=0.0, theta=0.0, x=0.0, y=0.0, node_type="apex")
    G.add_node(1, r=400.0, theta=-0.1, x=400.0 * math.cos(-0.1), y=400.0 * math.sin(-0.1), node_type="internal")
    G.add_node(2, r=400.0, theta=0.1, x=400.0 * math.cos(0.1), y=400.0 * math.sin(0.1), node_type="internal")
    # Leaf node 3 has two predecessors: 1 and 2
    G.add_node(3, r=600.0, theta=0.0, x=600.0, y=0.0, node_type="internal")

    G.add_edge(0, 1, straight_distance=400.0, length=400.0, sinuosity=1.0)
    G.add_edge(0, 2, straight_distance=400.0, length=400.0, sinuosity=1.0)
    G.add_edge(1, 3, straight_distance=200.0, length=200.0, sinuosity=1.0)
    G.add_edge(2, 3, straight_distance=200.0, length=200.0, sinuosity=1.0)

    extend_leaf_to_shoreline(G, domain)

    assert G.nodes[3]["r"] == 1000.0
    assert G.nodes[3]["node_type"] == "outlet"
    # Verify both incoming edges have updated straight_distance and length
    dist1 = math.hypot(G.nodes[3]["x"] - G.nodes[1]["x"], G.nodes[3]["y"] - G.nodes[1]["y"])
    dist2 = math.hypot(G.nodes[3]["x"] - G.nodes[2]["x"], G.nodes[3]["y"] - G.nodes[2]["y"])
    assert math.isclose(G.edges[1, 3]["straight_distance"], dist1)
    assert math.isclose(G.edges[2, 3]["straight_distance"], dist2)
    assert math.isclose(G.edges[1, 3]["length"], dist1)
    assert math.isclose(G.edges[2, 3]["length"], dist2)

