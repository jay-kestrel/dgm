"""
Primary polar growth tree backbone generator for DGM.
Synthesizes a planar bifurcating tree skeleton across the polar domain.
"""

import math
from collections import deque
from typing import Any

import networkx as nx
import numpy as np

from .domain import PolarDomain
from .geometry_utils import (
    edge_intersects_graph,
    edge_too_close_to_graph,
    extend_leaf_to_shoreline,
    incident_angle_between_segments,
)


def generate_polar_tree_backbone(
    N_target: int,
    R_rt: float,
    domain: PolarDomain,
    params: dict[str, Any],
    rng: np.random.Generator,
    branching_weights: tuple[float, float, float] | None = None
) -> nx.DiGraph:
    """
    Generates a polar tree backbone with variable-order branching (1, 2, or 3 children)
    and asymmetric stochastic growth for a target total node count N_target.
    Enforces Jordan Curve planarity filtering to guarantee 0 edge crossings in tree skeleton.
    """
    N_tree = max(1, int(N_target))

    # Resolve branching weights (default: 50% single-continuation, 45% bifurcation, 5% trifurcation)
    if branching_weights is None:
        branching_weights = params.get('branching_weights', (0.50, 0.45, 0.05))

    bw = np.array(branching_weights, dtype=float)
    if bw.sum() <= 0.0 or np.any(bw < 0.0):
        bw = np.array([0.50, 0.45, 0.05], dtype=float)
    p_weights = bw / bw.sum()

    G = nx.DiGraph()
    G.graph['N_target'] = N_tree

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

    if N_tree == 1:
        return G

    current_node_id = 1
    queue = deque([0])
    existing_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []

    # Estimate levels needed
    max_levels = math.ceil(math.log2(N_tree + 1)) + 1
    r_step_base = domain.r_shore / max(2, max_levels)

    step_min = float(params.get('radial_step_scale_min', 0.65))
    step_max = float(params.get('radial_step_scale_max', 1.35))
    coastal_node_decay = float(params.get('coastal_node_decay', 0.40))
    spread_decay = float(params.get('angular_spread_decay', 0.70))
    single_spread_fac = float(params.get('single_branch_spread_factor', 0.35))
    bifurc_min = float(params.get('bifurcation_spread_min', 0.60))
    bifurc_max = float(params.get('bifurcation_spread_max', 1.40))
    base_sin_mu = float(params.get('base_sinuosity_mean', 0.25))
    base_sin_sigma = max(1e-4, float(params.get('base_sinuosity_std', 0.15)))
    coastal_sin_decay = float(params.get('coastal_sinuosity_decay', 0.30))
    apex_width = max(0.1, float(params.get('apex_width', 250.0)))
    min_width = max(1e-4, float(params.get('min_channel_width', 0.1)))
    min_bif_angle_deg = float(params.get('min_bifurcation_angle_deg', 15.0))
    min_bif_angle = math.radians(min_bif_angle_deg)
    min_sibling_dist = max(20.0, float(params.get('min_sibling_distance', 0.005 * domain.r_shore)))
    min_edge_clearance = max(15.0, float(params.get('min_edge_clearance', 0.003 * domain.r_shore)))

    failed_parents = set()

    while current_node_id < N_tree:
        if not queue:
            # Recovery: find existing upstream nodes with available out-degree capacity (< 3)
            eligible_parents = [
                n for n in G.nodes()
                if n not in failed_parents and G.out_degree(n) < 3 and float(G.nodes[n].get('r', 0.0)) < domain.r_shore * 0.85
            ]
            if not eligible_parents:
                break
            # Sort eligible parents by radial distance ascending to grow downstream
            eligible_parents.sort(key=lambda n: float(G.nodes[n].get('r', 0.0)))
            queue.extend(eligible_parents)

        parent = queue.popleft()
        p_r = float(G.nodes[parent]['r'])
        p_theta = float(G.nodes[parent]['theta'])
        p_level = int(G.nodes[parent]['level'])
        parent_pos = (float(G.nodes[parent]['x']), float(G.nodes[parent]['y']))

        # Determine variable branching out-degree capacity check
        remaining_nodes = N_tree - current_node_id
        current_out = G.out_degree(parent)
        max_possible_children = max(1, 3 - current_out)

        if remaining_nodes >= 3 and max_possible_children >= 3:
            num_children = int(rng.choice([1, 2, 3], p=p_weights))
        elif remaining_nodes >= 2 and max_possible_children >= 2:
            p12_sum = float(p_weights[:2].sum())
            p12 = p_weights[:2] / p12_sum if p12_sum > 1e-12 else np.array([0.5, 0.5])
            num_children = int(rng.choice([1, 2], p=p12))
        else:
            num_children = 1

        num_children = min(num_children, max_possible_children)

        # Angular spreading angle narrows as tree deepens
        spread_base = (domain.theta_open / 2.5) / max(1.0, spread_decay * (p_level + 1))
        effective_min_bif = min(min_bif_angle, 0.40 * domain.theta_open)
        placed_children_for_parent: list[tuple[float, float, float]] = [
            (float(G.nodes[c]['theta']), float(G.nodes[c]['x']), float(G.nodes[c]['y']))
            for c in G.successors(parent)
        ]
        children_added = 0

        for i in range(num_children):
            if current_node_id >= N_tree:
                break

            # Asymmetric radial step per child branch with shoreward spatial contraction
            r_frac = min(1.0, max(0.0, p_r / max(1.0, domain.r_shore)))
            decay_factor = max(0.20, 1.0 - coastal_node_decay * r_frac)
            r_step = max(10.0, r_step_base * rng.uniform(step_min, step_max) * decay_factor)
            child_r = min(domain.r_shore * 0.95, p_r + r_step)
            if child_r <= p_r:
                child_r = min(domain.r_shore * 0.98, p_r + 5.0)

            # Scale minimal clearances proportionally to domain width if valley is extremely confined
            valley_w_est = child_r * min(math.pi, max(1e-4, domain.theta_open))
            if domain.lateral_valley_width is not None:
                valley_w_est = min(valley_w_est, domain.lateral_valley_width)
            eff_sib_dist = min(min_sibling_dist, max(1.0, 0.30 * valley_w_est))
            eff_edge_clr = min(min_edge_clearance, max(1.0, 0.20 * valley_w_est))

            # Angular spreading depending on branching order
            if num_children == 1:
                base_angle = float(rng.uniform(-spread_base * single_spread_fac, spread_base * single_spread_fac))
            elif num_children == 2:
                # Asymmetric bifurcation
                sign = 1.0 if i == 0 else -1.0
                base_angle = float(spread_base * sign * rng.uniform(bifurc_min, bifurc_max))
            else:  # num_children == 3 (trifurcation)
                if i == 0:
                    base_angle = float(spread_base * rng.uniform(0.7, 1.3))
                elif i == 1:
                    base_angle = float(rng.uniform(-spread_base * 0.3, spread_base * 0.3))
                else:
                    base_angle = float(-spread_base * rng.uniform(0.7, 1.3))

            # Planar candidate placement search to prevent edge crossings & secondary overlap
            placed = False
            child_theta = p_theta + base_angle
            child_pos = (0.0, 0.0)

            # Avoid degenerate zero-spread factors (minimum factor 0.45)
            for factor in [1.0, 0.85, 1.15, 0.70, 1.30, 0.55, 1.45]:
                angle_offset = base_angle * factor
                cand_theta = domain.clamp_theta(p_theta + angle_offset, child_r)
                cx, cy = domain.polar_to_cartesian(child_r, cand_theta)
                cand_pos = (cx, cy)

                # Enforce sibling angular divergence at parent junction and spatial separation
                violates_sibling = False
                for _s_th, s_x, s_y in placed_children_for_parent:
                    if incident_angle_between_segments(parent_pos, cand_pos, (s_x, s_y)) < effective_min_bif or math.hypot(cx - s_x, cy - s_y) < eff_sib_dist:
                        violates_sibling = True
                        break
                if violates_sibling:
                    continue

                if domain.is_in_domain(child_r, cand_theta) and \
                   not edge_intersects_graph(parent_pos, cand_pos, existing_edges) and \
                   not edge_too_close_to_graph(parent_pos, cand_pos, existing_edges, eff_edge_clr):
                    child_theta = cand_theta
                    child_pos = cand_pos
                    placed = True
                    break

            if not placed:
                for r_fac in [0.8, 1.1, 0.7, 1.2]:
                    alt_r = min(domain.r_shore * 0.95, p_r + r_step * r_fac)
                    if alt_r <= p_r:
                        alt_r = min(domain.r_shore * 0.98, p_r + 5.0)
                    cand_theta = domain.clamp_theta(p_theta + base_angle, alt_r)
                    cx, cy = domain.polar_to_cartesian(alt_r, cand_theta)
                    cand_pos = (cx, cy)

                    violates_sibling = False
                    for _s_th, s_x, s_y in placed_children_for_parent:
                        if incident_angle_between_segments(parent_pos, cand_pos, (s_x, s_y)) < effective_min_bif or math.hypot(cx - s_x, cy - s_y) < eff_sib_dist:
                            violates_sibling = True
                            break
                    if violates_sibling:
                        continue

                    if domain.is_in_domain(alt_r, cand_theta) and \
                       not edge_intersects_graph(parent_pos, cand_pos, existing_edges) and \
                       not edge_too_close_to_graph(parent_pos, cand_pos, existing_edges, eff_edge_clr):
                        child_r = alt_r
                        child_theta = cand_theta
                        child_pos = cand_pos
                        placed = True
                        break

            if not placed:
                continue

            placed_children_for_parent.append((child_theta, child_pos[0], child_pos[1]))
            children_added += 1

            cx, cy = child_pos
            node_type = 'outlet' if child_r >= domain.r_shore * 0.9 else 'internal'

            G.add_node(
                current_node_id,
                r=child_r,
                theta=child_theta,
                x=cx,
                y=cy,
                node_type=node_type,
                level=p_level + 1
            )

            # Compute Euclidean length and sinuosity with shoreward straightening
            straight_distance = max(10.0, math.hypot(cx - parent_pos[0], cy - parent_pos[1]))
            raw_sin = float(rng.lognormal(base_sin_mu, base_sin_sigma))
            r_mid = 0.5 * (p_r + child_r)
            r_ratio = min(1.0, max(0.0, r_mid / max(1.0, domain.r_shore)))
            sin_decay_mult = max(0.0, 1.0 - coastal_sin_decay * r_ratio)
            sinuosity = max(1.0, 1.0 + (raw_sin - 1.0) * sin_decay_mult)
            length = straight_distance * sinuosity

            edge_width = max(min_width, apex_width / max(1, p_level + 1))

            G.add_edge(
                parent,
                current_node_id,
                length=length,
                sinuosity=sinuosity,
                straight_distance=straight_distance,
                edge_type='tree',
                width=edge_width
            )

            existing_edges.append((parent_pos, child_pos))
            queue.append(current_node_id)
            current_node_id += 1

        if children_added == 0:
            failed_parents.add(parent)

        # Update parent type based on actual out-degree (preserve apex designation)
        actual_out = G.out_degree(parent)
        if actual_out >= 2 and G.nodes[parent].get('node_type') != 'apex':
            G.nodes[parent]['node_type'] = 'bifurcation' if actual_out == 2 else 'trifurcation'

    # Extend all terminal leaf nodes (outlets) directly to shoreline r_shore without crossing
    extend_leaf_to_shoreline(G, domain, existing_edges)

    return G
