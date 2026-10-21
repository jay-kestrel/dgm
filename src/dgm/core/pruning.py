"""
Pre-hydraulic network pruning and topological simplification for DGM.
Removes blind inland dead-ends, filters sub-threshold spurs, and contracts pass-through nodes
prior to Murray's Law width relaxation and Markov flux routing.
"""

import math
from typing import Any

import networkx as nx
import numpy as np

from .domain import PolarDomain
from .geometry_utils import edge_intersects_graph


def refresh_node_types(G: nx.DiGraph, r_shore: float, outlet_tolerance: float = 0.05) -> None:
    """
    Refreshes node_type attributes ('apex', 'bifurcation', 'trifurcation', 'confluence', 'outlet', 'internal')
    based on current in/out degrees and radial distance.
    """
    if G.number_of_nodes() == 0:
        return

    shore_cutoff = max(0.0, float(r_shore) * (1.0 - float(outlet_tolerance)))

    for n in G.nodes():
        if G.nodes[n].get('node_type') == 'apex' or n == 0:
            G.nodes[n]['node_type'] = 'apex'
            continue

        in_deg = G.in_degree(n)
        out_deg = G.out_degree(n)
        r = float(G.nodes[n].get('r', 0.0))

        if out_deg == 0:
            if r >= shore_cutoff:
                G.nodes[n]['node_type'] = 'outlet'
            else:
                G.nodes[n]['node_type'] = 'dead_end'
        elif out_deg >= 3:
            G.nodes[n]['node_type'] = 'trifurcation'
        elif out_deg == 2:
            G.nodes[n]['node_type'] = 'bifurcation'
        elif in_deg >= 2:
            G.nodes[n]['node_type'] = 'confluence'
        else:
            G.nodes[n]['node_type'] = 'internal'


def prune_dead_end_spurs(
    G: nx.DiGraph,
    r_shore: float,
    outlet_tolerance: float = 0.05,
    max_iterations: int = 10,
    min_spur_length: float = 0.0,
    prune_inland_dead_ends: bool = True
) -> tuple[nx.DiGraph, int]:
    """
    Prunes blind channel spurs that terminate inland without reaching the shoreline.
    Guarantees that apex and valid shoreline outlets (r >= r_shore * (1 - tol)) are never removed.

    Returns:
      (pruned_G, count_of_pruned_nodes)
    """
    if G.number_of_nodes() == 0:
        return G.copy(), 0

    pruned_G = G.copy()
    total_pruned = 0
    shore_cutoff = max(0.0, float(r_shore) * (1.0 - float(outlet_tolerance)))

    for _ in range(max(1, max_iterations)):
        candidates_to_prune: list[Any] = []

        for n in list(pruned_G.nodes()):
            # Never prune apex
            if pruned_G.nodes[n].get('node_type') == 'apex' or n == 0:
                continue

            # Only consider terminal leaves (out_degree == 0)
            if pruned_G.out_degree(n) == 0:
                r = float(pruned_G.nodes[n].get('r', 0.0))

                # Legitimate shoreline outlet: preserve
                if r >= shore_cutoff:
                    continue

                # Inland dead-end: check length threshold or blind channel flag
                in_edges = list(pruned_G.in_edges(n, data=True))
                if not in_edges:
                    # Isolated orphan node
                    candidates_to_prune.append(n)
                else:
                    edge_len = max(float(e[2].get('length', 0.0)) for e in in_edges)
                    if prune_inland_dead_ends or (min_spur_length > 0 and edge_len <= min_spur_length):
                        candidates_to_prune.append(n)

        if not candidates_to_prune:
            break

        # Remove candidates
        for n in candidates_to_prune:
            if n in pruned_G:
                pruned_G.remove_node(n)
                total_pruned += 1

    refresh_node_types(pruned_G, r_shore, outlet_tolerance)
    return pruned_G, total_pruned


def simplify_degree_two_chains(
    G: nx.DiGraph,
    r_shore: float,
    outlet_tolerance: float = 0.05,
    recalculate_reach_sinuosity: bool = True,
    params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None
) -> tuple[nx.DiGraph, int]:
    """
    Contracts all non-branching pass-through nodes (in_degree == 1 and out_degree == 1)
    into direct links between morphological junctions, inlets, and outlets.

    When recalculate_reach_sinuosity is True (default), calculates a single unified
    reach-scale sinuosity for the contracted link directly from the process model,
    preventing double-counting of geometric growth knees and aligning with DIDNT
    reach-scale conventions.

    Never contracts apex, outlets (out_degree == 0), confluences (in_degree >= 2),
    or bifurcations (out_degree >= 2).

    Returns:
      (simplified_G, count_of_contracted_nodes)
    """
    if G.number_of_nodes() == 0:
        return G.copy(), 0

    simplified_G = G.copy()
    contracted_count = 0

    while True:
        candidates = [
            n for n in simplified_G.nodes()
            if simplified_G.in_degree(n) == 1
            and simplified_G.out_degree(n) == 1
            and n != 0
            and simplified_G.nodes[n].get('node_type') != 'apex'
        ]

        contracted_in_pass = 0
        for n in candidates:
            if n not in simplified_G:
                continue

            if simplified_G.in_degree(n) != 1 or simplified_G.out_degree(n) != 1:
                continue

            in_edges = list(simplified_G.in_edges(n, data=True))
            out_edges = list(simplified_G.out_edges(n, data=True))

            if len(in_edges) != 1 or len(out_edges) != 1:
                continue

            p, _, d_in = in_edges[0]
            _, c, d_out = out_edges[0]

            # Prevent multi-edges or self-loops
            if p == c or simplified_G.has_edge(p, c):
                continue

            # Node coordinates
            x_p = float(simplified_G.nodes[p].get('x', 0.0))
            y_p = float(simplified_G.nodes[p].get('y', 0.0))
            x_n = float(simplified_G.nodes[n].get('x', 0.0))
            y_n = float(simplified_G.nodes[n].get('y', 0.0))
            x_c = float(simplified_G.nodes[c].get('x', 0.0))
            y_c = float(simplified_G.nodes[c].get('y', 0.0))

            straight_dist = max(1.0, math.hypot(x_c - x_p, y_c - y_p))

            len_in = float(d_in.get('length', math.hypot(x_n - x_p, y_n - y_p)))
            len_out = float(d_out.get('length', math.hypot(x_c - x_n, y_c - y_n)))
            total_len = len_in + len_out
            combined_sin = max(1.0, total_len / straight_dist)

            # Check for existing multi-vertex polylines (>2 points)
            has_curved_in = 'coordinates' in d_in and len(d_in['coordinates']) > 2
            has_curved_out = 'coordinates' in d_out and len(d_out['coordinates']) > 2

            if not (has_curved_in or has_curved_out):
                # Planarity guard: ensure straight chord (p, c) does not intersect any existing edge
                other_edges = [
                    ((float(simplified_G.nodes[u]['x']), float(simplified_G.nodes[u]['y'])),
                     (float(simplified_G.nodes[v]['x']), float(simplified_G.nodes[v]['y'])))
                    for u, v in simplified_G.edges()
                    if u != n and v != n
                ]
                if edge_intersects_graph((x_p, y_p), (x_c, y_c), other_edges):
                    continue

            if has_curved_in or has_curved_out:
                reach_len = max(straight_dist, total_len)
                reach_sin = combined_sin
                coords_in = d_in.get('coordinates', [[x_p, y_p], [x_n, y_n]])
                coords_out = d_out.get('coordinates', [[x_n, y_n], [x_c, y_c]])
                combined_coords = coords_in[:-1] + coords_out
            elif recalculate_reach_sinuosity:
                p_params = params or {}
                base_sin_mu = float(p_params.get('base_sinuosity_mean', p_params.get('base_sinuosity_mu', 0.25)))
                base_sin_sigma = max(1e-4, float(p_params.get('base_sinuosity_std', p_params.get('base_sinuosity_sigma', 0.15))))
                coastal_sin_decay = float(p_params.get('coastal_sinuosity_decay', 0.30))

                if rng is not None:
                    raw_sin = float(rng.lognormal(base_sin_mu, base_sin_sigma))
                else:
                    edge_seed = (int(p) * 31337 + int(c) * 7919) % (2**31) if isinstance(p, (int, np.integer)) and isinstance(c, (int, np.integer)) else hash(f"{p}_{c}") % (2**31)
                    local_rng = np.random.default_rng(edge_seed)
                    raw_sin = float(local_rng.lognormal(base_sin_mu, base_sin_sigma))

                r_p = float(simplified_G.nodes[p].get('r', math.hypot(x_p, y_p)))
                r_c = float(simplified_G.nodes[c].get('r', math.hypot(x_c, y_c)))
                r_mid = 0.5 * (r_p + r_c)
                r_ratio = min(1.0, max(0.0, r_mid / max(1.0, r_shore)))
                sin_decay_mult = max(0.0, 1.0 - coastal_sin_decay * r_ratio)
                reach_sin = max(1.0, 1.0 + (raw_sin - 1.0) * sin_decay_mult)
                reach_len = max(straight_dist, straight_dist * reach_sin)
                combined_coords = [[x_p, y_p], [x_c, y_c]]
            else:
                reach_len = max(straight_dist, total_len)
                reach_sin = combined_sin
                coords_in = d_in.get('coordinates', [[x_p, y_p], [x_n, y_n]])
                coords_out = d_out.get('coordinates', [[x_n, y_n], [x_c, y_c]])
                combined_coords = coords_in[:-1] + coords_out

            edge_type = 'tree' if (d_in.get('edge_type') == 'tree' and d_out.get('edge_type') == 'tree') else 'loop'
            width_val = max(0.1, float(d_in.get('width', d_out.get('width', 1.0))))

            # Remove node n and add direct edge p -> c
            simplified_G.remove_node(n)
            simplified_G.add_edge(
                p,
                c,
                length=reach_len,
                straight_distance=straight_dist,
                sinuosity=reach_sin,
                edge_type=edge_type,
                width=width_val,
                coordinates=combined_coords,
            )

            contracted_in_pass += 1
            contracted_count += 1

        if contracted_in_pass == 0:
            break

    refresh_node_types(simplified_G, r_shore, outlet_tolerance)
    return simplified_G, contracted_count


def prune_and_simplify_network(
    G: nx.DiGraph,
    domain: PolarDomain | None = None,
    params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None
) -> nx.DiGraph:
    """
    Orchestrator function called before hydraulics:
    Optionally prunes blind dead-end inland spurs and simplifies degree-2 pass-through chains.
    Strictly preserves weak connectivity and shoreline delivery invariants.
    """
    if params is None:
        params = {}

    enable_pruning = params.get('enable_pruning', False)
    if not enable_pruning:
        return G

    r_shore = domain.r_shore if domain else float(params.get('r_shore', 10000.0))
    outlet_tol = float(params.get('outlet_tolerance', 0.05))
    prune_inland = params.get('prune_inland_dead_ends', True)
    min_spur_len = float(params.get('min_spur_length', 0.0))
    simplify_chains = params.get('simplify_pass_through_nodes', False)
    recalc_sin = params.get('recalculate_reach_sinuosity', True)
    max_iters = int(params.get('max_prune_iterations', 5))

    current_G = G.copy()
    stats: dict[str, Any] = {
        'initial_nodes': current_G.number_of_nodes(),
        'initial_edges': current_G.number_of_edges(),
        'pruned_nodes': 0,
        'contracted_nodes': 0,
    }

    # Step 1: Prune dead-end spurs
    current_G, pruned_count = prune_dead_end_spurs(
        current_G,
        r_shore=r_shore,
        outlet_tolerance=outlet_tol,
        max_iterations=max_iters,
        min_spur_length=min_spur_len,
        prune_inland_dead_ends=prune_inland
    )
    stats['pruned_nodes'] = pruned_count

    # Step 2: Contract pass-through chains
    if simplify_chains:
        current_G, contracted_count = simplify_degree_two_chains(
            current_G,
            r_shore=r_shore,
            outlet_tolerance=outlet_tol,
            recalculate_reach_sinuosity=recalc_sin,
            params=params,
            rng=rng
        )
        stats['contracted_nodes'] = contracted_count

    # Enforce weak connectivity invariant: retain connected component of apex if any orphan remained
    if not nx.is_weakly_connected(current_G) and current_G.number_of_nodes() > 1:
        apex_nodes = [n for n in current_G.nodes() if current_G.nodes[n].get('node_type') == 'apex']
        apex_node = apex_nodes[0] if apex_nodes else (0 if 0 in current_G else next(iter(current_G.nodes())))
        apex_comp = nx.node_connected_component(current_G.to_undirected(), apex_node)
        to_remove = [n for n in current_G.nodes() if n not in apex_comp]
        for n in to_remove:
            current_G.remove_node(n)
        stats['pruned_nodes'] += len(to_remove)

    stats['final_nodes'] = current_G.number_of_nodes()
    stats['final_edges'] = current_G.number_of_edges()

    current_G.graph['pruning_stats'] = stats
    return current_G
