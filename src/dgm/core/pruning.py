"""
Pre-hydraulic network pruning and topological simplification for DGM.
Removes blind inland dead-ends, filters sub-threshold spurs, and contracts pass-through nodes
prior to Murray's Law width relaxation and Markov flux routing.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import math
import networkx as nx

from .domain import PolarDomain


def refresh_node_types(G: nx.DiGraph, r_shore: float, outlet_tolerance: float = 0.05) -> None:
    """
    Refreshes node_type attributes ('apex', 'bifurcation', 'trifurcation', 'confluence', 'outlet', 'internal')
    based on current in/out degrees and radial distance.
    """
    shore_cutoff = r_shore * (1.0 - outlet_tolerance)

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
) -> Tuple[nx.DiGraph, int]:
    """
    Prunes blind channel spurs that terminate inland without reaching the shoreline.
    Guarantees that apex and valid shoreline outlets (r >= r_shore * (1 - tol)) are never removed.

    Returns:
      (pruned_G, count_of_pruned_nodes)
    """
    pruned_G = G.copy()
    total_pruned = 0
    shore_cutoff = r_shore * (1.0 - outlet_tolerance)

    for _ in range(max(1, max_iterations)):
        candidates_to_prune: List[Any] = []

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
                    edge_len = float(in_edges[0][2].get('length', 0.0))
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
    r_shore: float
) -> Tuple[nx.DiGraph, int]:
    """
    Contracts pass-through nodes with in_degree == 1 and out_degree == 1 into unified edges,
    accumulating lengths and merging polyline coordinates if present.
    Never contracts apex, outlets, or junctions.

    Returns:
      (simplified_G, count_of_contracted_nodes)
    """
    simplified_G = G.copy()
    contracted_count = 0

    pass_through_nodes = [
        n for n in simplified_G.nodes()
        if simplified_G.in_degree(n) == 1
        and simplified_G.out_degree(n) == 1
        and n != 0
        and simplified_G.nodes[n].get('node_type') not in ('apex', 'outlet')
    ]

    for n in pass_through_nodes:
        if n not in simplified_G:
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

        # Combine edge properties
        x_p, y_p = float(simplified_G.nodes[p]['x']), float(simplified_G.nodes[p]['y'])
        x_c, y_c = float(simplified_G.nodes[c]['x']), float(simplified_G.nodes[c]['y'])
        straight_dist = max(1.0, math.hypot(x_c - x_p, y_c - y_p))

        len_in = float(d_in.get('length', straight_dist / 2.0))
        len_out = float(d_out.get('length', straight_dist / 2.0))
        total_len = len_in + len_out
        combined_sin = max(1.0, total_len / straight_dist)

        # Merge polyline coordinates if available
        combined_coords = None
        if 'coordinates' in d_in and 'coordinates' in d_out:
            coords_in = d_in['coordinates']
            coords_out = d_out['coordinates']
            # Concatenate coordinates avoiding duplicate joint point
            combined_coords = coords_in[:-1] + coords_out

        edge_type = 'tree' if (d_in.get('edge_type') == 'tree' and d_out.get('edge_type') == 'tree') else 'loop'

        # Remove node n and add direct edge p -> c
        simplified_G.remove_node(n)
        simplified_G.add_edge(
            p,
            c,
            length=total_len,
            straight_distance=straight_dist,
            sinuosity=combined_sin,
            edge_type=edge_type,
            width=d_in.get('width', d_out.get('width', 1.0))
        )
        if combined_coords:
            simplified_G.edges[p, c]['coordinates'] = combined_coords

        contracted_count += 1

    refresh_node_types(simplified_G, r_shore)
    return simplified_G, contracted_count


def prune_and_simplify_network(
    G: nx.DiGraph,
    domain: Optional[PolarDomain] = None,
    params: Optional[Dict[str, Any]] = None
) -> nx.DiGraph:
    """
    Orchestrator function called before hydraulics:
    Optionally prunes blind dead-end inland spurs and simplifies degree-2 pass-through chains.
    """
    if params is None:
        params = {}

    enable_pruning = params.get('enable_pruning', False)
    if not enable_pruning:
        return G

    r_shore = domain.r_shore if domain else float(params.get('r_shore', 10000.0))
    prune_inland = params.get('prune_inland_dead_ends', True)
    min_spur_len = float(params.get('min_spur_length', 0.0))
    simplify_chains = params.get('simplify_pass_through_nodes', False)
    max_iters = int(params.get('max_prune_iterations', 5))

    current_G = G.copy()
    stats: Dict[str, Any] = {
        'initial_nodes': current_G.number_of_nodes(),
        'initial_edges': current_G.number_of_edges(),
        'pruned_nodes': 0,
        'contracted_nodes': 0,
    }

    # Step 1: Prune dead-end spurs
    current_G, pruned_count = prune_dead_end_spurs(
        current_G,
        r_shore=r_shore,
        max_iterations=max_iters,
        min_spur_length=min_spur_len,
        prune_inland_dead_ends=prune_inland
    )
    stats['pruned_nodes'] = pruned_count

    # Step 2: Contract pass-through chains
    if simplify_chains:
        current_G, contracted_count = simplify_degree_two_chains(current_G, r_shore=r_shore)
        stats['contracted_nodes'] = contracted_count

    stats['final_nodes'] = current_G.number_of_nodes()
    stats['final_edges'] = current_G.number_of_edges()

    current_G.graph['pruning_stats'] = stats
    return current_G
