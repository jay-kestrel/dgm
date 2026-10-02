"""
Curved channel centerline generation and proximity-constrained meander dynamics for DGM.
Provides spatial clearance calculation, meander amplitude bounding, and 2D polyline generation.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import networkx as nx


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
    v: Any
) -> float:
    """
    Computes the minimum spatial clearance distance from the midpoint of edge (u, v)
    to all other non-incident channel edges in the network.
    """
    p_u = (float(G.nodes[u]['x']), float(G.nodes[u]['y']))
    p_v = (float(G.nodes[v]['x']), float(G.nodes[v]['y']))
    midpoint = (0.5 * (p_u[0] + p_v[0]), 0.5 * (p_u[1] + p_v[1]))

    incident_nodes = {u, v}
    min_dist = float('inf')

    for edge_u, edge_v in G.edges():
        # Skip identical or incident edges sharing a node
        if edge_u in incident_nodes or edge_v in incident_nodes:
            continue

        a = (float(G.nodes[edge_u]['x']), float(G.nodes[edge_u]['y']))
        b = (float(G.nodes[edge_v]['x']), float(G.nodes[edge_v]['y']))

        dist = point_to_segment_distance(midpoint, a, b)
        if dist < min_dist:
            min_dist = dist

    return min_dist


def apply_channel_proximity_constraints(
    G: nx.DiGraph,
    clearance_fraction: float = 0.40,
    min_amplitude_floor: float = 10.0
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

    for u, v, d in G.edges(data=True):
        straight_dist = float(d.get('straight_distance', 0.0))
        if straight_dist <= 0:
            x_u, y_u = G.nodes[u]['x'], G.nodes[u]['y']
            x_v, y_v = G.nodes[v]['x'], G.nodes[v]['y']
            straight_dist = max(1.0, math.hypot(x_v - x_u, y_v - y_u))
            d['straight_distance'] = straight_dist

        target_sin = float(d.get('sinuosity', 1.0))

        # Compute clearance to non-incident channels
        clearance = compute_edge_clearance(G, u, v)
        d['clearance_distance'] = clearance

        if math.isinf(clearance):
            # No adjacent channels: full unconstrained sinuosity
            d['max_meander_amplitude'] = float('inf')
            effective_sin = max(1.0, target_sin)
        else:
            # Maximum allowed meander amplitude is bounded by clearance fraction
            max_amp = max(min_amplitude_floor, clearance * clearance_fraction)
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
    seed: int = 42
) -> nx.DiGraph:
    """
    Generates smooth 2D polyline centerlines for all edges in G.
    Respects channel proximity constraints, calculates meander amplitude,
    and directs bends toward the side with maximum spatial clearance.

    Stores polyline coordinates on each edge under 'coordinates': List[List[float, float]].
    """
    num_points = max(3, int(num_points))
    num_bends = max(1, int(num_bends))

    # First apply proximity constraints to determine effective sinuosities
    apply_channel_proximity_constraints(G, clearance_fraction=clearance_fraction)

    s_values = np.linspace(0.0, 1.0, num_points)
    # Window function sin(pi * s) ensures endpoints attach exactly at nodes (h(0)=0, h(1)=0)
    window = np.sin(np.pi * s_values)
    wave = np.sin(2.0 * np.pi * num_bends * s_values)
    profile = wave * window

    for u, v, d in G.edges(data=True):
        x_u, y_u = float(G.nodes[u]['x']), float(G.nodes[u]['y'])
        x_v, y_v = float(G.nodes[v]['x']), float(G.nodes[v]['y'])
        straight_dist = float(d.get('straight_distance', math.hypot(x_v - x_u, y_v - y_u)))

        if straight_dist <= 1e-6:
            d['coordinates'] = [[x_u, y_u], [x_v, y_v]]
            d['meander_amplitude'] = 0.0
            continue

        sinuosity = float(d.get('sinuosity', 1.0))

        if sinuosity <= 1.0001:
            # Straight channel
            pts = [[float(x_u + s * (x_v - x_u)), float(y_u + s * (y_v - y_u))] for s in s_values]
            d['coordinates'] = pts
            d['meander_amplitude'] = 0.0
            continue

        # Target amplitude: A = (L / pi) * sqrt(S - 1)
        amp = (straight_dist / math.pi) * math.sqrt(max(0.0, sinuosity - 1.0))
        max_amp = d.get('max_meander_amplitude', float('inf'))
        if not math.isinf(max_amp):
            amp = min(amp, max_amp)
        d['meander_amplitude'] = amp

        # Normal unit vector perpendicular to channel segment
        tx = (x_v - x_u) / straight_dist
        ty = (y_v - y_u) / straight_dist
        nx_val = -ty
        ny_val = tx

        # Deterministic seed for meander direction check
        if isinstance(u, (int, np.integer)) and isinstance(v, (int, np.integer)):
            edge_hash = (int(seed) + int(u) * 31337 + int(v) * 7919) % (2**31)
        else:
            edge_hash = (int(seed) + abs(hash((str(u), str(v))))) % (2**31)

        # Test both normal signs (+/-) at crest (s = 0.5 / num_bends) to pick the direction
        # with greater clearance to neighboring channels
        crest_s = 0.25 / num_bends
        mid_x = x_u + crest_s * (x_v - x_u)
        mid_y = y_u + crest_s * (y_v - y_u)
        crest_amp = amp * math.sin(np.pi * crest_s)

        p_pos = (mid_x + crest_amp * nx_val, mid_y + crest_amp * ny_val)
        p_neg = (mid_x - crest_amp * nx_val, mid_y - crest_amp * ny_val)

        incident_nodes = {u, v}
        clearance_pos = float('inf')
        clearance_neg = float('inf')

        for other_u, other_v in G.edges():
            if other_u in incident_nodes or other_v in incident_nodes:
                continue
            seg_a = (float(G.nodes[other_u]['x']), float(G.nodes[other_u]['y']))
            seg_b = (float(G.nodes[other_v]['x']), float(G.nodes[other_v]['y']))
            d_pos = point_to_segment_distance(p_pos, seg_a, seg_b)
            d_neg = point_to_segment_distance(p_neg, seg_a, seg_b)
            if d_pos < clearance_pos:
                clearance_pos = d_pos
            if d_neg < clearance_neg:
                clearance_neg = d_neg

        # Direct meander towards open space (larger clearance); if equal, use pseudo-random hash
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
