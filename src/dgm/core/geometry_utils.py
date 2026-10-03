"""
Jordan Curve Theorem planarity filtering and geometric graph utilities for DGM.
"""

from typing import Tuple, List, Optional, Any
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
    p1x, p1y = p1
    p2x, p2y = p2
    q1x, q1y = q1
    q2x, q2y = q2

    # Fast 2D Axis-Aligned Bounding Box (AABB) rejection
    if (p1x < q1x and p1x < q2x and p2x < q1x and p2x < q2x) or \
       (p1x > q1x and p1x > q2x and p2x > q1x and p2x > q2x) or \
       (p1y < q1y and p1y < q2y and p2y < q1y and p2y < q2y) or \
       (p1y > q1y and p1y > q2y and p2y > q1y and p2y > q2y):
        return False

    # Shared endpoints are valid topological junctions, not planar crossings
    if is_close_point(p1, q1) or is_close_point(p1, q2) or is_close_point(p2, q1) or is_close_point(p2, q2):
        return False

    def orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        coord_scale = max(1.0, abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1]), abs(r[0]), abs(r[1]))
        tol = 1e-9 * (coord_scale * coord_scale)
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
    existing_edges: List[Any]
) -> bool:
    """
    Checks if a candidate line segment (p1, p2) intersects any existing edge in the graph.
    Employs fast inlined 2D AABB bounding-box rejection before Jordan Curve evaluation.
    """
    p1x, p1y = p1
    p2x, p2y = p2
    min_px = p1x if p1x < p2x else p2x
    max_px = p2x if p1x < p2x else p1x
    min_py = p1y if p1y < p2y else p2y
    max_py = p2y if p1y < p2y else p1y

    for edge in existing_edges:
        e_q1, e_q2 = edge[0], edge[1]
        q1x, q1y = e_q1
        q2x, q2y = e_q2
        min_qx = q1x if q1x < q2x else q2x
        max_qx = q2x if q1x < q2x else q1x
        min_qy = q1y if q1y < q2y else q2y
        max_qy = q2y if q1y < q2y else q1y

        # Fast inlined AABB rejection: disjoint bounding boxes cannot intersect
        if max_px < min_qx or min_px > max_qx or max_py < min_qy or min_py > max_qy:
            continue

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
    minimizing edge crossings (retaining valid position if no crossing-free path to shoreline exists).
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

                # Compute effective angular span at shoreline
                span = domain.theta_open
                if domain.lateral_valley_width is not None and domain.r_shore > 1e-3:
                    max_sin = min(1.0, (domain.lateral_valley_width / 2.0) / domain.r_shore)
                    span = min(span, 2.0 * math.asin(max_sin))

                # Adaptive offsets proportional to span, plus static offsets
                candidate_offsets = [0.0]
                for frac in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.35, -0.35, 0.5, -0.5, 0.7, -0.7]:
                    candidate_offsets.append(frac * span)
                for t_offset in [0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                    if t_offset not in candidate_offsets:
                        candidate_offsets.append(t_offset)

                extended = False
                for t_offset in candidate_offsets:
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

                # If no crossing-free path to shoreline exists (e.g. downstream branches
                # densely block the corridor in narrow valleys), retain the node's existing
                # valid planar coordinates rather than creating planar edge crossings.

            G.nodes[n]['node_type'] = 'outlet'
