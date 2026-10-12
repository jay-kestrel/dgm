"""
Tests for Jordan Curve Theorem planarity filtering and edge intersections.
"""

import pytest
from dgm.core.geometry_utils import segments_intersect, edge_intersects_graph


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
