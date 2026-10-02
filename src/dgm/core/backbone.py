"""
Primary polar growth tree backbone generator for DGM.
Synthesizes a planar bifurcating tree skeleton across the polar domain.
"""

from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import math
import numpy as np
import networkx as nx

from .domain import PolarDomain
from .geometry_utils import edge_intersects_graph, extend_leaf_to_shoreline


def generate_polar_tree_backbone(
    N_target: int,
    R_rt: float,
    domain: PolarDomain,
    params: Dict[str, Any],
    rng: np.random.Generator,
    branching_weights: Optional[Tuple[float, float, float]] = None
) -> nx.DiGraph:
    """
    Generates a polar tree backbone with variable-order branching (1, 2, or 3 children)
    and asymmetric stochastic growth for a target total node count N_target.
    Enforces Jordan Curve planarity filtering to guarantee 0 edge crossings in tree skeleton.
    """
    N_tree = N_target

    # Resolve branching weights (default: 50% single-continuation, 45% bifurcation, 5% trifurcation)
    if branching_weights is None:
        branching_weights = params.get('branching_weights', (0.50, 0.45, 0.05))

    bw = np.array(branching_weights, dtype=float)
    p_weights = bw / bw.sum()

    G = nx.DiGraph()
    G.graph['N_target'] = N_target

    # Node 0: Apex / Inlet
    G.add_node(
        0,
        r=0.0,
        theta=0.0,
        x=0.0,
        y=0.0,
        node_type='apex',
        level=0
    )

    current_node_id = 1
    queue = deque([0])
    existing_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    # Estimate levels needed
    max_levels = int(math.ceil(math.log2(N_tree + 1))) + 1
    r_step_base = domain.r_shore / max(2, max_levels)

    step_min = params.get('radial_step_scale_min', 0.65)
    step_max = params.get('radial_step_scale_max', 1.35)
    spread_decay = params.get('angular_spread_decay', 0.70)
    single_spread_fac = params.get('single_branch_spread_factor', 0.35)
    bifurc_min = params.get('bifurcation_spread_min', 0.60)
    bifurc_max = params.get('bifurcation_spread_max', 1.40)

    while queue and current_node_id < N_tree:
        parent = queue.popleft()
        p_r = G.nodes[parent]['r']
        p_theta = G.nodes[parent]['theta']
        p_level = G.nodes[parent]['level']
        parent_pos = (G.nodes[parent]['x'], G.nodes[parent]['y'])

        # Determine variable branching out-degree capacity check
        remaining_nodes = N_tree - current_node_id
        if remaining_nodes >= 3:
            num_children = int(rng.choice([1, 2, 3], p=p_weights))
        elif remaining_nodes == 2:
            p12 = p_weights[:2] / p_weights[:2].sum()
            num_children = int(rng.choice([1, 2], p=p12))
        elif remaining_nodes == 1:
            num_children = 1
        else:
            break

        # Angular spreading angle narrows as tree deepens
        spread_base = (domain.theta_open / 2.5) / max(1.0, spread_decay * (p_level + 1))

        for i in range(num_children):
            if current_node_id >= N_tree:
                break

            # Asymmetric radial step per child branch
            r_step = r_step_base * rng.uniform(step_min, step_max)
            child_r = min(domain.r_shore * 0.95, p_r + r_step)

            # Angular spreading depending on branching order
            if num_children == 1:
                base_angle = rng.uniform(-spread_base * single_spread_fac, spread_base * single_spread_fac)
            elif num_children == 2:
                # Asymmetric bifurcation
                sign = 1.0 if i == 0 else -1.0
                base_angle = spread_base * sign * rng.uniform(bifurc_min, bifurc_max)
            else:  # num_children == 3 (trifurcation)
                if i == 0:
                    base_angle = spread_base * rng.uniform(0.7, 1.3)
                elif i == 1:
                    base_angle = rng.uniform(-spread_base * 0.3, spread_base * 0.3)
                else:
                    base_angle = -spread_base * rng.uniform(0.7, 1.3)

            # Planar candidate placement search to prevent edge crossings
            placed = False
            child_theta = p_theta + base_angle
            child_pos = (0.0, 0.0)

            for factor in [1.0, 0.8, 1.2, 0.6, 1.4, 0.4, 1.6, 0.2, 0.0]:
                angle_offset = base_angle * factor
                cand_theta = domain.clamp_theta(p_theta + angle_offset, child_r)
                cx, cy = domain.polar_to_cartesian(child_r, cand_theta)
                cand_pos = (cx, cy)

                if domain.is_in_domain(child_r, cand_theta) and not edge_intersects_graph(parent_pos, cand_pos, existing_edges):
                    child_theta = cand_theta
                    child_pos = cand_pos
                    placed = True
                    break

            if not placed:
                for r_fac in [0.8, 1.1, 0.7, 1.2]:
                    alt_r = min(domain.r_shore * 0.95, p_r + r_step * r_fac)
                    cand_theta = domain.clamp_theta(p_theta + base_angle, alt_r)
                    cx, cy = domain.polar_to_cartesian(alt_r, cand_theta)
                    if domain.is_in_domain(alt_r, cand_theta) and not edge_intersects_graph(parent_pos, (cx, cy), existing_edges):
                        child_r = alt_r
                        child_theta = cand_theta
                        child_pos = (cx, cy)
                        placed = True
                        break

            if not placed:
                continue

            cx, cy = child_pos
            if child_r >= domain.r_shore * 0.90:
                node_type = 'outlet'
            else:
                node_type = 'internal'

            G.add_node(
                current_node_id,
                r=child_r,
                theta=child_theta,
                x=cx,
                y=cy,
                node_type=node_type,
                level=p_level + 1
            )

            # Compute Euclidean length and sinuosity
            straight_distance = max(10.0, math.hypot(cx - parent_pos[0], cy - parent_pos[1]))
            sinuosity = float(rng.lognormal(0.25, 0.15))
            length = straight_distance * sinuosity

            G.add_edge(
                parent,
                current_node_id,
                length=length,
                sinuosity=sinuosity,
                straight_distance=straight_distance,
                edge_type='tree',
                width=params.get('apex_width', 250.0) / max(1, p_level + 1)
            )

            existing_edges.append((parent_pos, child_pos))
            queue.append(current_node_id)
            current_node_id += 1

        # Update parent type based on actual out-degree (preserve apex designation)
        actual_out = G.out_degree(parent)
        if actual_out >= 2 and G.nodes[parent]['node_type'] not in ('apex',):
            G.nodes[parent]['node_type'] = 'bifurcation' if actual_out == 2 else 'trifurcation'

    # Extend all terminal leaf nodes (outlets) directly to shoreline r_shore without crossing
    extend_leaf_to_shoreline(G, domain, existing_edges)

    return G
