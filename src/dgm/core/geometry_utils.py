"""
Jordan Curve Theorem planarity filtering and geometric graph utilities for DGM.
"""

from typing import Tuple, List, Optional
import math
import networkx as nx

from .domain import PolarDomain


def is_close_point(p1: Tuple[float, float], p2: Tuple[float, float], tol: float = 1e-6) -> bool:
    """Robust floating point equality check for 2D points."""
    return math.isclose(p1[0], p2[0], abs_tol=tol) and math.isclose(p1[1], p2[1], abs_tol=tol)


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float]
) -> bool:
    """
    Jordan Curve Theorem line segment intersection test.
    Returns True if segment (p1, p2) intersects segment (q1, q2).
    Shared endpoints evaluate to False (valid non-intersecting junctions).
    Handles collinear overlaps and floating-point tolerances robustly.
    """
    # Shared endpoints are valid topological junctions, not planar crossings
    if is_close_point(p1, q1) or is_close_point(p1, q2) or is_close_point(p2, q1) or is_close_point(p2, q2):
        return False

    def orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        coord_scale = max(1.0, max(abs(x) for pt in (p, q, r) for x in pt))
        tol = 1e-9 * max(1.0, coord_scale ** 2)
        if abs(val) < tol:
            return 0  # Collinear
        return 1 if val > 0 else 2  # 1 = Clockwise, 2 = Counter-clockwise

    def on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
        return (
            q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])
        )

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General Case: Straddling orientations
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases: Collinear overlaps
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False


def edge_intersects_graph(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    existing_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> bool:
    """
    Checks if a candidate line segment (p1, p2) intersects any existing edge in the graph.
    """
    for e_q1, e_q2 in existing_edges:
        if segments_intersect(p1, p2, e_q1, e_q2):
            return True
    return False


def extend_leaf_to_shoreline(
    G: nx.DiGraph,
    domain: Optional[PolarDomain],
    existing_edges: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
) -> None:
    """
    Extends all terminal leaf nodes (outlets) directly to shoreline r_shore,
    minimizing edge crossings (with fallback forced extension if no crossing-free angle is found).
    Updates node coordinates and edge lengths in place.
    """
    if domain is None:
        return

    if existing_edges is None:
        existing_edges = [
            ((G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y']))
            for u, v in G.edges()
        ]

    for n in list(G.nodes()):
        if G.out_degree(n) == 0 and n != 0:
            if G.nodes[n]['r'] < domain.r_shore:
                px, py = G.nodes[n]['x'], G.nodes[n]['y']
                pred = list(G.predecessors(n))
                parent_n = pred[0] if pred else None
                par_pos = (G.nodes[parent_n]['x'], G.nodes[parent_n]['y']) if parent_n is not None else None

                other_edges = [e for e in existing_edges if e != (par_pos, (px, py))]

                orig_theta = G.nodes[n]['theta']
                extended = False
                for t_offset in [0.0, 0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                    cand_theta = domain.clamp_theta(orig_theta + t_offset, domain.r_shore)
                    cx, cy = domain.polar_to_cartesian(domain.r_shore, cand_theta)
                    if par_pos is None or not edge_intersects_graph(par_pos, (cx, cy), other_edges):
                        G.nodes[n]['r'] = domain.r_shore
                        G.nodes[n]['theta'] = cand_theta
                        G.nodes[n]['x'] = cx
                        G.nodes[n]['y'] = cy
                        if parent_n is not None and G.has_edge(parent_n, n):
                            straight_dist = max(10.0, math.hypot(cx - par_pos[0], cy - par_pos[1]))
                            sinuosity = G.edges[parent_n, n].get('sinuosity', 1.0)
                            G.edges[parent_n, n]['straight_distance'] = straight_dist
                            G.edges[parent_n, n]['length'] = straight_dist * sinuosity
                            if (par_pos, (px, py)) in existing_edges:
                                idx = existing_edges.index((par_pos, (px, py)))
                                existing_edges[idx] = (par_pos, (cx, cy))
                        extended = True
                        break

                if not extended:
                    # Fallback: force extension along original theta
                    cand_theta = domain.clamp_theta(orig_theta, domain.r_shore)
                    cx, cy = domain.polar_to_cartesian(domain.r_shore, cand_theta)
                    G.nodes[n]['r'] = domain.r_shore
                    G.nodes[n]['theta'] = cand_theta
                    G.nodes[n]['x'] = cx
                    G.nodes[n]['y'] = cy
                    if parent_n is not None and G.has_edge(parent_n, n):
                        straight_dist = max(10.0, math.hypot(cx - par_pos[0], cy - par_pos[1]))
                        sinuosity = G.edges[parent_n, n].get('sinuosity', 1.0)
                        G.edges[parent_n, n]['straight_distance'] = straight_dist
                        G.edges[parent_n, n]['length'] = straight_dist * sinuosity
                        if (par_pos, (px, py)) in existing_edges:
                            idx = existing_edges.index((par_pos, (px, py)))
                            existing_edges[idx] = (par_pos, (cx, cy))

            G.nodes[n]['node_type'] = 'outlet'
