"""
Jordan Curve Theorem planarity filtering and geometric graph utilities for DGM.
"""

import math
from typing import Any

import networkx as nx

from .domain import PolarDomain


def is_close_point(p1: tuple[float, float], p2: tuple[float, float], tol: float = 1e-6) -> bool:
    """Robust floating point equality check for 2D points in meters."""
    return math.isclose(float(p1[0]), float(p2[0]), abs_tol=tol) and math.isclose(float(p1[1]), float(p2[1]), abs_tol=tol)


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float]
) -> bool:
    """
    Jordan Curve Theorem line segment intersection test.
    Returns True if segment (p1, p2) intersects segment (q1, q2).
    Shared endpoints evaluate to False for valid non-intersecting topological junctions,
    but evaluate to True if collinear overlapping edges share the same ray.
    Handles collinear overlaps, degenerate zero-length segments, and floating-point tolerances robustly.
    """
    p1x, p1y = float(p1[0]), float(p1[1])
    p2x, p2y = float(p2[0]), float(p2[1])
    q1x, q1y = float(q1[0]), float(q1[1])
    q2x, q2y = float(q2[0]), float(q2[1])

    # Check for degenerate zero-length segments
    p_degen = is_close_point(p1, p2)
    q_degen = is_close_point(q1, q2)
    if p_degen and q_degen:
        return False
    if p_degen or q_degen:
        # A degenerate point cannot cross a planar line segment
        return False

    # Fast 2D Axis-Aligned Bounding Box (AABB) rejection
    if (p1x < q1x and p1x < q2x and p2x < q1x and p2x < q2x) or \
       (p1x > q1x and p1x > q2x and p2x > q1x and p2x > q2x) or \
       (p1y < q1y and p1y < q2y and p2y < q1y and p2y < q2y) or \
       (p1y > q1y and p1y > q2y and p2y > q1y and p2y > q2y):
        return False

    # Identify shared endpoints
    shared_p1_q1 = is_close_point(p1, q1)
    shared_p1_q2 = is_close_point(p1, q2)
    shared_p2_q1 = is_close_point(p2, q1)
    shared_p2_q2 = is_close_point(p2, q2)

    # Identical segments (both endpoints match) -> collinear overlap
    if (shared_p1_q1 and shared_p2_q2) or (shared_p1_q2 and shared_p2_q1):
        return True

    # Shared endpoints: valid junctions unless collinear overlapping along the same ray
    if shared_p1_q1 or shared_p1_q2 or shared_p2_q1 or shared_p2_q2:
        if shared_p1_q1:
            S, A, B = p1, p2, q2
        elif shared_p1_q2:
            S, A, B = p1, p2, q1
        elif shared_p2_q1:
            S, A, B = p2, p1, q2
        else:
            S, A, B = p2, p1, q1

        vAx, vAy = A[0] - S[0], A[1] - S[1]
        vBx, vBy = B[0] - S[0], B[1] - S[1]
        cross = vAx * vBy - vAy * vBx
        coord_scale = max(1.0, abs(S[0]), abs(S[1]), abs(A[0]), abs(A[1]), abs(B[0]), abs(B[1]))
        tol = 1e-9 * (coord_scale * coord_scale)

        if abs(cross) < tol:
            dot = vAx * vBx + vAy * vBy
            if dot > 0.0:
                # Same ray from shared vertex -> invalid collinear overlap
                return True
        return False

    def orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> int:
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        coord_scale = max(1.0, abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1]), abs(r[0]), abs(r[1]))
        tol = 1e-9 * (coord_scale * coord_scale)
        if abs(val) < tol:
            return 0  # Collinear
        return 1 if val > 0 else 2  # 1 = Clockwise, 2 = Counter-clockwise

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        coord_scale = max(1.0, abs(p[0]), abs(p[1]), abs(q[0]), abs(q[1]), abs(r[0]), abs(r[1]))
        tol = 1e-9 * coord_scale
        return (
            q[0] <= max(p[0], r[0]) + tol and q[0] >= min(p[0], r[0]) - tol and
            q[1] <= max(p[1], r[1]) + tol and q[1] >= min(p[1], r[1]) - tol
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
    return bool(o4 == 0 and on_segment(q1, p2, q2))


def edge_intersects_graph(
    p1: tuple[float, float],
    p2: tuple[float, float],
    existing_edges: list[Any]
) -> bool:
    """
    Checks if a candidate line segment (p1, p2) intersects any existing edge in the graph.
    Employs fast inlined 2D AABB bounding-box rejection before Jordan Curve evaluation.
    """
    p1x, p1y = float(p1[0]), float(p1[1])
    p2x, p2y = float(p2[0]), float(p2[1])
    min_px = min(p2x, p1x)
    max_px = max(p1x, p2x)
    min_py = min(p2y, p1y)
    max_py = max(p1y, p2y)

    for edge in existing_edges:
        e_q1, e_q2 = edge[0], edge[1]
        q1x, q1y = float(e_q1[0]), float(e_q1[1])
        q2x, q2y = float(e_q2[0]), float(e_q2[1])
        min_qx = min(q2x, q1x)
        max_qx = max(q1x, q2x)
        min_qy = min(q2y, q1y)
        max_qy = max(q1y, q2y)

        # Fast inlined AABB rejection: disjoint bounding boxes cannot intersect
        if max_px < min_qx or min_px > max_qx or max_py < min_qy or min_py > max_qy:
            continue

        if segments_intersect(p1, p2, e_q1, e_q2):
            return True
    return False


def point_to_segment_distance(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float]
) -> float:
    """
    Computes the shortest Euclidean distance in meters from point p to line segment (a, b).
    Handles zero-length segments and clamping to segment endpoints robustly.
    """
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-12:
        return float(math.hypot(px - ax, py - ay))

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return float(math.hypot(px - proj_x, py - proj_y))


def incident_angle_between_segments(
    p_common: tuple[float, float],
    p_a: tuple[float, float],
    p_b: tuple[float, float]
) -> float:
    """
    Computes the interior angle in radians between vector (p_a - p_common) and (p_b - p_common).
    Returns value in [0, pi].
    """
    ux = float(p_a[0]) - float(p_common[0])
    uy = float(p_a[1]) - float(p_common[1])
    vx = float(p_b[0]) - float(p_common[0])
    vy = float(p_b[1]) - float(p_common[1])
    len_u = math.hypot(ux, uy)
    len_v = math.hypot(vx, vy)
    if len_u < 1e-6 or len_v < 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, (ux * vx + uy * vy) / (len_u * len_v)))
    return float(math.acos(dot))


def edge_too_close_to_graph(
    p1: tuple[float, float],
    p2: tuple[float, float],
    existing_edges: list[Any],
    min_clearance: float
) -> bool:
    """
    Checks if a candidate line segment (p1, p2) comes closer than min_clearance
    to any non-incident edge in existing_edges.
    """
    if min_clearance <= 0.0 or not existing_edges:
        return False

    p1x, p1y = float(p1[0]), float(p1[1])
    p2x, p2y = float(p2[0]), float(p2[1])

    min_px = min(p2x, p1x)
    max_px = max(p1x, p2x)
    min_py = min(p2y, p1y)
    max_py = max(p1y, p2y)

    for edge in existing_edges:
        e_q1, e_q2 = edge[0], edge[1]
        q1x, q1y = float(e_q1[0]), float(e_q1[1])
        q2x, q2y = float(e_q2[0]), float(e_q2[1])

        # Skip if they share an endpoint (incident edges)
        if (abs(p1x - q1x) < 1e-4 and abs(p1y - q1y) < 1e-4) or \
           (abs(p1x - q2x) < 1e-4 and abs(p1y - q2y) < 1e-4) or \
           (abs(p2x - q1x) < 1e-4 and abs(p2y - q1y) < 1e-4) or \
           (abs(p2x - q2x) < 1e-4 and abs(p2y - q2y) < 1e-4):
            continue

        min_qx = min(q2x, q1x)
        max_qx = max(q1x, q2x)
        min_qy = min(q2y, q1y)
        max_qy = max(q1y, q2y)

        # Bounding box rejection with clearance buffer
        if (min_qx - max_px > min_clearance) or (min_px - max_qx > min_clearance) or \
           (min_qy - max_py > min_clearance) or (min_py - max_qy > min_clearance):
            continue

        # For non-intersecting 2D line segments, the shortest distance occurs at one of the 4 endpoints
        if point_to_segment_distance(p1, e_q1, e_q2) < min_clearance or \
           point_to_segment_distance(p2, e_q1, e_q2) < min_clearance or \
           point_to_segment_distance(e_q1, p1, p2) < min_clearance or \
           point_to_segment_distance(e_q2, p1, p2) < min_clearance:
            return True

    return False



def extend_leaf_to_shoreline(
    G: nx.DiGraph,
    domain: PolarDomain | None,
    existing_edges: list[tuple[tuple[float, float], tuple[float, float]]] | None = None
) -> None:
    """
    Extends all terminal leaf nodes (outlets) directly to shoreline r_shore,
    minimizing edge crossings (retaining valid position if no crossing-free path to shoreline exists).
    Updates node coordinates and edge lengths in place.
    """
    if domain is None or domain.r_shore <= 0.0:
        return

    if existing_edges is None:
        existing_edges = [
            ((float(G.nodes[u]['x']), float(G.nodes[u]['y'])), (float(G.nodes[v]['x']), float(G.nodes[v]['y'])))
            for u, v in G.edges()
        ]

    for n in list(G.nodes()):
        if G.out_degree(n) == 0 and G.nodes[n].get('node_type') != 'apex' and n != 0:
            if float(G.nodes[n].get('r', 0.0)) < domain.r_shore:
                px, py = float(G.nodes[n].get('x', 0.0)), float(G.nodes[n].get('y', 0.0))
                pred = list(G.predecessors(n))
                parent_n = pred[0] if pred else None
                par_pos = (float(G.nodes[parent_n]['x']), float(G.nodes[parent_n]['y'])) if parent_n is not None else None

                other_edges = [e for e in existing_edges if e != (par_pos, (px, py))]

                orig_theta = float(G.nodes[n].get('theta', 0.0))

                # Compute effective angular span at shoreline
                span = max(1e-4, domain.theta_open)
                if domain.lateral_valley_width is not None and domain.r_shore > 1e-3:
                    max_sin = min(1.0, max(0.0, (domain.lateral_valley_width / 2.0) / domain.r_shore))
                    span = min(span, 2.0 * math.asin(max_sin))

                # Adaptive offsets proportional to span, plus static offsets
                candidate_offsets = [0.0]
                for frac in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.35, -0.35, 0.5, -0.5, 0.7, -0.7]:
                    candidate_offsets.append(frac * span)
                for t_offset in [0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                    if t_offset not in candidate_offsets:
                        candidate_offsets.append(t_offset)

                for t_offset in candidate_offsets:
                    cand_theta = domain.clamp_theta(orig_theta + t_offset, domain.r_shore)
                    cx, cy = domain.polar_to_cartesian(domain.r_shore, cand_theta)

                    # Ensure new shoreline outlet position does not collide with existing nodes
                    too_close_node = False
                    for other_n in G.nodes():
                        if other_n != n:
                            ox, oy = float(G.nodes[other_n]['x']), float(G.nodes[other_n]['y'])
                            if math.hypot(cx - ox, cy - oy) < 20.0:
                                too_close_node = True
                                break
                    if too_close_node:
                        continue

                    if par_pos is None or (
                        not edge_intersects_graph(par_pos, (cx, cy), other_edges)
                        and not edge_too_close_to_graph(par_pos, (cx, cy), other_edges, 15.0)
                    ):
                        G.nodes[n]['r'] = domain.r_shore
                        G.nodes[n]['theta'] = cand_theta
                        G.nodes[n]['x'] = cx
                        G.nodes[n]['y'] = cy
                        if parent_n is not None and G.has_edge(parent_n, n):
                            straight_dist = max(10.0, math.hypot(cx - par_pos[0], cy - par_pos[1]))
                            sinuosity = float(G.edges[parent_n, n].get('sinuosity', 1.0))
                            G.edges[parent_n, n]['straight_distance'] = straight_dist
                            G.edges[parent_n, n]['length'] = straight_dist * sinuosity
                            if (par_pos, (px, py)) in existing_edges:
                                idx = existing_edges.index((par_pos, (px, py)))
                                existing_edges[idx] = (par_pos, (cx, cy))
                        break

            # Set node role: if at or near shoreline (r >= r_shore * 0.95), designate outlet
            if float(G.nodes[n].get('r', 0.0)) >= domain.r_shore * 0.95:
                G.nodes[n]['node_type'] = 'outlet'
            elif G.nodes[n].get('node_type') != 'apex':
                G.nodes[n]['node_type'] = 'dead_end'

