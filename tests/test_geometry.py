"""
Tests for Jordan Curve Theorem planarity filtering and edge intersections.
"""

from dgm.core.geometry_utils import segments_intersect, edge_intersects_graph


def test_intersecting_segments():
    # Crossing X
    p1 = (0.0, 0.0)
    p2 = (10.0, 10.0)
    q1 = (0.0, 10.0)
    q2 = (10.0, 0.0)
    assert segments_intersect(p1, p2, q1, q2) is True


def test_non_intersecting_segments():
    # Parallel lines
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.0)
    q1 = (0.0, 5.0)
    q2 = (10.0, 5.0)
    assert segments_intersect(p1, p2, q1, q2) is False


def test_shared_endpoint_allowed():
    # Segments that share a vertex (a legitimate junction) must NOT be marked as an intersection
    p1 = (0.0, 0.0)
    p2 = (5.0, 5.0)
    q1 = (5.0, 5.0)
    q2 = (10.0, 0.0)
    assert segments_intersect(p1, p2, q1, q2) is False


def test_edge_intersects_graph():
    existing = [
        ((0.0, 0.0), (10.0, 10.0)),
        ((10.0, 10.0), (20.0, 10.0))
    ]
    # Crossing the first edge
    assert edge_intersects_graph((0.0, 10.0), (10.0, 0.0), existing) is True

    # Completely disjoint
    assert edge_intersects_graph((50.0, 50.0), (60.0, 60.0), existing) is False
