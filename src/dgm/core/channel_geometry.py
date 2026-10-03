"""
Curved channel centerline generation and proximity-constrained meander dynamics for DGM.
Provides spatial clearance calculation, meander amplitude bounding, and 2D polyline generation.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import zlib
import numpy as np
import networkx as nx
from .geometry_utils import segments_intersect


def point_to_segment_distance(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float]
) -> float:
    """
    Computes the shortest Euclidean distance from point p to line segment (a, b).
    """
    px, py = p
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)

    # Project point onto segment line, clamped to [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def compute_edge_clearance(
    G: nx.DiGraph,
    u: Any,
    v: Any,
    edge_records: Optional[List[Any]] = None,
    node_coords: Optional[Dict[Any, Tuple[float, float]]] = None
) -> float:
    """
    Computes the minimum spatial clearance distance from edge (u, v) (sampled along
    meander crests and midpoint) to all other non-incident channel edges in the network.
    If any non-incident edges cross or touch, clearance is 0.0.
    """
    if node_coords is None:
        p_u = (float(G.nodes[u].get('x', 0.0)), float(G.nodes[u].get('y', 0.0)))
        p_v = (float(G.nodes[v].get('x', 0.0)), float(G.nodes[v].get('y', 0.0)))
    else:
        p_u = node_coords[u]
        p_v = node_coords[v]

    min_dist = float('inf')

    # Sample interior points along (u, v) at quarter points (crests) and midpoint
    sample_points = (
        (0.5 * (p_u[0] + p_v[0]), 0.5 * (p_u[1] + p_v[1])),
        (0.75 * p_u[0] + 0.25 * p_v[0], 0.75 * p_u[1] + 0.25 * p_v[1]),
        (0.25 * p_u[0] + 0.75 * p_v[0], 0.25 * p_u[1] + 0.75 * p_v[1]),
    )

    min_pu_pv_x = p_u[0] if p_u[0] < p_v[0] else p_v[0]
    max_pu_pv_x = p_v[0] if p_u[0] < p_v[0] else p_u[0]
    min_pu_pv_y = p_u[1] if p_u[1] < p_v[1] else p_v[1]
    max_pu_pv_y = p_v[1] if p_u[1] < p_v[1] else p_u[1]

    if edge_records is None:
        if node_coords is None:
            node_coords = {n: (float(d.get('x', 0.0)), float(d.get('y', 0.0))) for n, d in G.nodes(data=True)}
        edge_records = []
        for edge_u, edge_v in G.edges():
            a = node_coords[edge_u]
            b = node_coords[edge_v]
            edge_records.append((
                edge_u, edge_v, a, b,
                min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])
            ))

    for edge_u, edge_v, a, b, min_ab_x, max_ab_x, min_ab_y, max_ab_y in edge_records:
        if edge_u == u or edge_u == v or edge_v == u or edge_v == v:
            continue

        # Bounding box clearance rejection: if bbox distance exceeds min_dist, cannot be closer
        if min_dist < float('inf'):
            if (min_ab_x - max_pu_pv_x > min_dist) or (min_pu_pv_x - max_ab_x > min_dist) or \
               (min_ab_y - max_pu_pv_y > min_dist) or (min_pu_pv_y - max_ab_y > min_dist):
                continue

        # If non-incident edges cross, clearance is immediately zero
        if segments_intersect(p_u, p_v, a, b):
            return 0.0

        for pt in sample_points:
            dist = point_to_segment_distance(pt, a, b)
            if dist < min_dist:
                min_dist = dist

    return min_dist


def apply_channel_proximity_constraints(
    G: nx.DiGraph,
    clearance_fraction: float = 0.40,
    min_amplitude_floor: float = 0.0
) -> nx.DiGraph:
    """
    Constrains channel sinuosity based on proximity to neighboring non-incident channels.
    Prevents adjacent channels from meandering into each other.

    Updates edge attributes in place:
      - 'clearance_distance': distance to nearest adjacent non-incident channel
      - 'max_meander_amplitude': maximum allowable meander amplitude
      - 'sinuosity': constrained sinuosity S_effective <= S_max
      - 'length': updated physical length (straight_distance * sinuosity)
    """
    clearance_fraction = max(0.05, min(1.0, float(clearance_fraction)))
    min_amplitude_floor = max(0.0, float(min_amplitude_floor))

    # Pre-extract all node coordinates and edge bounding boxes once for O(E^2) acceleration
    node_coords = {n: (float(d.get('x', 0.0)), float(d.get('y', 0.0))) for n, d in G.nodes(data=True)}
    edge_records = []
    for edge_u, edge_v in G.edges():
        a = node_coords[edge_u]
        b = node_coords[edge_v]
        edge_records.append((
            edge_u, edge_v, a, b,
            min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])
        ))

    for u, v, d in G.edges(data=True):
        straight_dist = float(d.get('straight_distance', 0.0))
        if straight_dist <= 1e-6:
            p_u = node_coords[u]
            p_v = node_coords[v]
            straight_dist = max(1.0, math.hypot(p_v[0] - p_u[0], p_v[1] - p_u[1]))
            d['straight_distance'] = straight_dist

        target_sin = float(d.get('sinuosity', 1.0))

        # Compute clearance to non-incident channels with precomputed records
        clearance = compute_edge_clearance(G, u, v, edge_records=edge_records, node_coords=node_coords)
        d['clearance_distance'] = clearance

        if math.isinf(clearance):
            # No adjacent channels: full unconstrained sinuosity
            d['max_meander_amplitude'] = float('inf')
            effective_sin = max(1.0, target_sin)
        else:
            # Maximum allowed meander amplitude is bounded by clearance fraction
            # If clearance <= 0.0 (intersecting/touching), amplitude is capped to 0.0
            max_amp = max(min_amplitude_floor, max(0.0, clearance * clearance_fraction))
            d['max_meander_amplitude'] = max_amp

            # Analytical relationship: S_max = 1 + (pi * A_max / L)^2
            s_max = 1.0 + (math.pi * max_amp / straight_dist) ** 2
            effective_sin = max(1.0, min(target_sin, s_max))

        d['sinuosity'] = effective_sin
        d['length'] = max(1.0, straight_dist * effective_sin)

    return G


def generate_channel_centerlines(
    G: nx.DiGraph,
    clearance_fraction: float = 0.40,
    num_points: int = 21,
    num_bends: int = 1,
    seed: Optional[int] = 42
) -> nx.DiGraph:
    """
    Generates smooth 2D polyline centerlines for all edges in G.
    Respects channel proximity constraints, calculates meander amplitude,
    and directs bends toward the side with maximum spatial clearance.

    Stores polyline coordinates on each edge under 'coordinates': List[List[float, float]].
    """
    num_points = max(3, int(num_points))
    num_bends = max(1, int(num_bends))
    seed_int = 42 if seed is None else int(seed)

    # First apply proximity constraints to determine effective sinuosities
    apply_channel_proximity_constraints(G, clearance_fraction=clearance_fraction)

    s_values = np.linspace(0.0, 1.0, num_points)
    # Window function sin(pi * s) ensures endpoints attach exactly at nodes (h(0)=0, h(1)=0)
    window = np.sin(np.pi * s_values)
    wave = np.sin(2.0 * np.pi * num_bends * s_values)
    profile = wave * window

    # Meander crest fractions for direction selection across all wave lobes
    crest_s_values = [(2 * k - 1) / (4.0 * num_bends) for k in range(1, 2 * num_bends + 1)]

    # Pre-extract all node coordinates and edge bounding boxes once
    node_coords = {n: (float(d.get('x', 0.0)), float(d.get('y', 0.0))) for n, d in G.nodes(data=True)}
    edge_records = []
    for edge_u, edge_v in G.edges():
        a = node_coords[edge_u]
        b = node_coords[edge_v]
        edge_records.append((
            edge_u, edge_v, a, b,
            min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])
        ))

    for u, v, d in G.edges(data=True):
        x_u, y_u = node_coords[u]
        x_v, y_v = node_coords[v]
        chord = math.hypot(x_v - x_u, y_v - y_u)
        straight_dist = float(d.get('straight_distance', chord))

        if chord <= 1e-6 or straight_dist <= 1e-6:
            d['coordinates'] = [[x_u, y_u], [x_v, y_v]]
            d['meander_amplitude'] = 0.0
            continue

        # If edge already has an explicit multi-vertex polyline from reach simplification,
        # preserve its trajectory
        if 'coordinates' in d and len(d['coordinates']) > 2:
            continue

        sinuosity = float(d.get('sinuosity', 1.0))

        if sinuosity <= 1.0001:
            # Straight channel
            pts = [[float(x_u + s * (x_v - x_u)), float(y_u + s * (y_v - y_u))] for s in s_values]
            pts[0] = [x_u, y_u]
            pts[-1] = [x_v, y_v]
            d['coordinates'] = pts
            d['meander_amplitude'] = 0.0
            continue

        # Target amplitude: A = (L / pi) * sqrt(S - 1)
        amp = (straight_dist / math.pi) * math.sqrt(max(0.0, sinuosity - 1.0))
        max_amp = d.get('max_meander_amplitude', float('inf'))
        if not math.isinf(max_amp):
            amp = min(amp, max_amp)
        d['meander_amplitude'] = amp

        if amp <= 1e-6:
            pts = [[float(x_u + s * (x_v - x_u)), float(y_u + s * (y_v - y_u))] for s in s_values]
            pts[0] = [x_u, y_u]
            pts[-1] = [x_v, y_v]
            d['coordinates'] = pts
            continue

        # Normal unit vector perpendicular to channel segment
        tx = (x_v - x_u) / chord
        ty = (y_v - y_u) / chord
        nx_val = -ty
        ny_val = tx

        # Deterministic seed for meander direction tie-breaking
        if isinstance(u, (int, np.integer)) and isinstance(v, (int, np.integer)):
            edge_hash = (seed_int + int(u) * 31337 + int(v) * 7919) % (2**31)
        else:
            edge_hash = (seed_int + zlib.crc32(f"{u}_{v}".encode())) % (2**31)

        # Test both candidate signs (+1/-1) across all wave crests to pick the direction
        # that maximizes clearance to neighboring channels (deflects into open floodplain)
        incident_nodes = {u, v}
        clearance_pos = float('inf')
        clearance_neg = float('inf')

        for s_c in crest_s_values:
            mid_x = x_u + s_c * (x_v - x_u)
            mid_y = y_u + s_c * (y_v - y_u)
            h_c = math.sin(2.0 * math.pi * num_bends * s_c) * math.sin(math.pi * s_c)
            disp = amp * h_c

            p_pos = (mid_x + disp * nx_val, mid_y + disp * ny_val)
            p_neg = (mid_x - disp * nx_val, mid_y - disp * ny_val)

            for other_u, other_v, seg_a, seg_b, min_ab_x, max_ab_x, min_ab_y, max_ab_y in edge_records:
                if other_u in incident_nodes or other_v in incident_nodes:
                    continue
                d_pos = point_to_segment_distance(p_pos, seg_a, seg_b)
                d_neg = point_to_segment_distance(p_neg, seg_a, seg_b)
                if d_pos < clearance_pos:
                    clearance_pos = d_pos
                if d_neg < clearance_neg:
                    clearance_neg = d_neg

        # Direct meander towards open space (larger clearance); if equal/symmetric, use pseudo-random hash
        if clearance_pos > clearance_neg + 1e-3:
            sign = 1.0
        elif clearance_neg > clearance_pos + 1e-3:
            sign = -1.0
        else:
            sign = 1.0 if (edge_hash % 2 == 0) else -1.0

        # Construct polyline vertices
        polyline = []
        for s, h in zip(s_values, profile):
            x_s = x_u + s * (x_v - x_u) + sign * amp * h * nx_val
            y_s = y_u + s * (y_v - y_u) + sign * amp * h * ny_val
            polyline.append([float(x_s), float(y_s)])

        # Ensure exact node attachments at endpoints
        polyline[0] = [x_u, y_u]
        polyline[-1] = [x_v, y_v]

        d['coordinates'] = polyline

    return G
