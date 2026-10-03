"""
Emergent loop generation engines for DGM.
Provides:
  - Mechanism A: Spatial Potential-Field Reconnection Engine (Konkol et al. 2022)
  - Mechanism B: Thermodynamic nER* Optimality Pruning Engine (Tejedor et al. 2017)
Enforces Jordan Curve Theorem planarity filtering and degree constraints.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

from .domain import PolarDomain
from .geometry_utils import edge_intersects_graph, extend_leaf_to_shoreline
from .hydraulics import apply_emergent_hydraulics, solve_markov_flux


def apply_loop_mechanism(
    G: Optional[nx.DiGraph] = None,
    domain: Optional[PolarDomain] = None,
    mechanism: str = "potential_field",
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> nx.DiGraph:
    """Routes loop engine execution to Mechanism A (potential_field) or Mechanism B (ner_optimality)."""
    if G is None:
        G = kwargs.get('G_tree', kwargs.get('G', None))
    if G is None:
        raise ValueError("A NetworkX DiGraph G must be provided to apply_loop_mechanism.")

    if 'mechanism_type' in kwargs:
        mechanism = kwargs['mechanism_type']

    if params is None:
        params = {}
    else:
        params = params.copy()

    if 'N_target' in kwargs and 'N_target' not in params:
        params['N_target'] = kwargs['N_target']
    if 'R_rt' in kwargs and 'R_rt' not in params:
        params['R_rt'] = kwargs['R_rt']

    if rng is None:
        rng = np.random.default_rng(seed)

    if mechanism == "ner_optimality":
        return generate_loops_ner_optimality(G, domain, params, rng=rng, seed=seed)
    else:
        return generate_loops_potential_field(G, domain, params, rng=rng, seed=seed)


def generate_loops_potential_field(
    G: nx.DiGraph,
    domain: Optional[PolarDomain],
    params: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    seed: int = 42
) -> nx.DiGraph:
    """
    Mechanism A: Spatial Potential-Field Reconnection Engine.
    Generates cross-ties governed by probability field P_loop(u, v | R_rt).
    Enforces Jordan Curve planarity filtering and degree limits (k_in <= max_in_deg, k_out <= max_out_deg).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    current_G = G.copy()

    N_target = params.get('N_target', G.graph.get('N_target', len(current_G.nodes())))
    R_rt = params.get('R_rt', domain.R_rt if domain else 0.5)
    l0_ratio = float(params.get('candidate_dist_scale', 0.25))
    c0 = float(params.get('reconnection_prob_base', 0.55))
    P0 = float(params.get('anastomosis_prob_base', 0.0))
    max_in = int(params.get('max_in_degree', 2))
    max_out = int(params.get('max_out_degree', 3))

    effective_c = P0 + c0 * R_rt
    r_shore = domain.r_shore if domain else 10000.0
    l0 = l0_ratio * r_shore

    nodes = list(current_G.nodes())
    existing_edges = [
        ((current_G.nodes[u]['x'], current_G.nodes[u]['y']),
         (current_G.nodes[v]['x'], current_G.nodes[v]['y']))
        for u, v in current_G.edges()
    ]

    coords = np.array([[current_G.nodes[n]['x'], current_G.nodes[n]['y']] for n in nodes])
    tree = KDTree(coords)
    max_dist = max(l0, -l0 * math.log(0.02 / max(1e-4, effective_c)))
    pairs = tree.query_pairs(max_dist)

    candidates = []
    for i, j in pairs:
        u, v = nodes[i], nodes[j]

        # Enforce downstream orientation (r_src < r_tgt)
        r_u, r_v = current_G.nodes[u]['r'], current_G.nodes[v]['r']
        if abs(r_u - r_v) < 1e-4:
            src, tgt = (u, v) if u < v else (v, u)
        elif r_u < r_v:
            src, tgt = u, v
        else:
            src, tgt = v, u

        if current_G.has_edge(src, tgt) or current_G.has_edge(tgt, src):
            continue

        if current_G.out_degree(src) >= max_out or current_G.in_degree(tgt) >= max_in:
            continue

        dist = domain.law_of_cosines_distance(
            current_G.nodes[src]['r'], current_G.nodes[src]['theta'],
            current_G.nodes[tgt]['r'], current_G.nodes[tgt]['theta']
        ) if domain else math.hypot(
            current_G.nodes[src]['x'] - current_G.nodes[tgt]['x'],
            current_G.nodes[src]['y'] - current_G.nodes[tgt]['y']
        )

        d_theta = abs(current_G.nodes[src]['theta'] - current_G.nodes[tgt]['theta']) if domain else 0.0
        p_loop = min(0.95, effective_c * math.exp(-dist / l0) * max(0.0, math.cos(d_theta)))

        if p_loop <= 0.02:
            continue

        # Downstream monotonicity: since delta channels flow downstream (r_src < r_tgt),
        # directed cycles are impossible unless radial coordinates are equal within tolerance.
        same_r = abs(r_u - r_v) < 1e-4
        if same_r and nx.has_path(current_G, tgt, src):
            continue

        candidates.append((src, tgt, p_loop, dist, same_r))

    candidates.sort(key=lambda x: x[2], reverse=True)

    for src, tgt, p_loop, dist, same_r in candidates:
        if current_G.out_degree(src) >= max_out or current_G.in_degree(tgt) >= max_in:
            continue
        if same_r and nx.has_path(current_G, tgt, src):
            continue
        if rng.uniform(0.0, 1.0) <= p_loop:
            p1 = (current_G.nodes[src]['x'], current_G.nodes[src]['y'])
            p2 = (current_G.nodes[tgt]['x'], current_G.nodes[tgt]['y'])

            if not edge_intersects_graph(p1, p2, existing_edges):
                current_G.add_edge(src, tgt, edge_type='loop')
                existing_edges.append((p1, p2))

                if current_G.out_degree(src) >= 2:
                    current_G.nodes[src]['node_type'] = 'bifurcation'
                if current_G.in_degree(tgt) >= 2:
                    current_G.nodes[tgt]['node_type'] = 'confluence'

    extend_leaf_to_shoreline(current_G, domain, existing_edges)
    return current_G


def generate_loops_ner_optimality(
    G: nx.DiGraph,
    domain: Optional[PolarDomain],
    params: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    seed: int = 42
) -> nx.DiGraph:
    """
    Mechanism B: Thermodynamic nER* Optimality Engine (Tejedor et al. 2017).
    Constructs a candidate mesh, then uses Simulated Annealing to prune links to
    maximize Non-Local Entropy Rate (nER*) under channel volume/count constraints.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    current_G = G.copy()

    N_target = params.get('N_target', G.graph.get('N_target', len(current_G.nodes())))
    R_rt = params.get('R_rt', domain.R_rt if domain else 0.5)
    l0_ratio = float(params.get('candidate_dist_scale', 0.35))
    r_shore = domain.r_shore if domain else 10000.0
    l0 = l0_ratio * r_shore
    max_in = int(params.get('max_in_degree', 2))
    max_out = int(params.get('max_out_degree', 3))

    nodes = list(current_G.nodes())
    existing_edges = [
        ((current_G.nodes[u]['x'], current_G.nodes[u]['y']),
         (current_G.nodes[v]['x'], current_G.nodes[v]['y']))
        for u, v in current_G.edges()
    ]

    coords = np.array([[current_G.nodes[n]['x'], current_G.nodes[n]['y']] for n in nodes])
    tree = KDTree(coords)
    pairs = tree.query_pairs(l0)

    all_candidates = []
    for i, j in pairs:
        u, v = nodes[i], nodes[j]

        r_u, r_v = current_G.nodes[u]['r'], current_G.nodes[v]['r']
        if abs(r_u - r_v) < 1e-4:
            src, tgt = (u, v) if u < v else (v, u)
        elif r_u < r_v:
            src, tgt = u, v
        else:
            src, tgt = v, u

        if current_G.has_edge(src, tgt) or current_G.has_edge(tgt, src):
            continue
        if current_G.out_degree(src) >= max_out or current_G.in_degree(tgt) >= max_in:
            continue
        if nx.has_path(current_G, tgt, src):
            continue

        dist = domain.law_of_cosines_distance(
            current_G.nodes[src]['r'], current_G.nodes[src]['theta'],
            current_G.nodes[tgt]['r'], current_G.nodes[tgt]['theta']
        ) if domain else math.hypot(
            current_G.nodes[src]['x'] - current_G.nodes[tgt]['x'],
            current_G.nodes[src]['y'] - current_G.nodes[tgt]['y']
        )

        if dist <= l0:
            all_candidates.append((src, tgt, dist))

    rng.shuffle(all_candidates)

    for src, tgt, dist in all_candidates:
        if current_G.out_degree(src) >= max_out or current_G.in_degree(tgt) >= max_in:
            continue
        if nx.has_path(current_G, tgt, src):
            continue
        p1 = (current_G.nodes[src]['x'], current_G.nodes[src]['y'])
        p2 = (current_G.nodes[tgt]['x'], current_G.nodes[tgt]['y'])

        if not edge_intersects_graph(p1, p2, existing_edges):
            current_G.add_edge(src, tgt, edge_type='loop')
            existing_edges.append((p1, p2))

            if current_G.out_degree(src) >= 2:
                current_G.nodes[src]['node_type'] = 'bifurcation'
            if current_G.in_degree(tgt) >= 2:
                current_G.nodes[tgt]['node_type'] = 'confluence'

    current_G = apply_emergent_hydraulics(current_G, params, seed=seed)
    _, W, phi = solve_markov_flux(current_G, params)

    def calc_ner_star(W_mat, phi_vec):
        nonzero_mask = W_mat > 1e-12
        out_degrees = np.sum(nonzero_mask, axis=1)
        valid_branching = out_degrees > 1
        max_ner = float(np.sum(phi_vec[valid_branching] * np.log2(out_degrees[valid_branching])))

        i_indices, j_indices = np.nonzero(nonzero_mask)
        w_vals = W_mat[i_indices, j_indices]
        phi_vals = phi_vec[i_indices]
        ner = -float(np.sum(phi_vals * w_vals * np.log2(w_vals)))
        return (ner / max_ner) if max_ner > 1e-6 else 0.0

    current_ner_star = calc_ner_star(W, phi)

    max_loops_slope = float(params.get('target_max_loops_slope', 0.22))
    max_loops_intercept = float(params.get('target_max_loops_intercept', 0.05))
    max_loops_target = max(1, int(round((max_loops_intercept + max_loops_slope * R_rt) * N_target)))

    T = float(params.get('sa_initial_temp', 1.0))
    alpha = float(params.get('sa_cooling_rate', 0.975))
    sa_steps = int(params.get('sa_max_steps', 250))

    for step in range(sa_steps):
        loop_edges = [(u, v) for u, v, d in current_G.edges(data=True) if d.get('edge_type') == 'loop']
        if not loop_edges:
            break

        edge_to_remove = loop_edges[rng.integers(0, len(loop_edges))]
        test_G = current_G.copy()
        test_G.remove_edge(*edge_to_remove)

        if not nx.is_weakly_connected(test_G):
            continue

        test_G = apply_emergent_hydraulics(test_G, params, seed=seed)
        _, W_test, phi_test = solve_markov_flux(test_G, params)
        test_ner_star = calc_ner_star(W_test, phi_test)

        cost_current = -current_ner_star
        cost_test = -test_ner_star
        delta_cost = cost_test - cost_current

        accept = False
        if delta_cost <= 0:
            accept = True
        elif len(loop_edges) > max_loops_target:
            if rng.uniform(0.0, 1.0) < math.exp(-delta_cost / max(1e-4, T)):
                accept = True

        if accept:
            current_G = test_G
            current_ner_star = test_ner_star

            src, tgt = edge_to_remove
            if current_G.out_degree(src) == 1:
                current_G.nodes[src]['node_type'] = 'internal'
            if current_G.in_degree(tgt) <= 1:
                current_G.nodes[tgt]['node_type'] = 'internal'

        T *= alpha

    extend_leaf_to_shoreline(current_G, domain, existing_edges)
    return current_G
