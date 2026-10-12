"""
Emergent loop generation engines for DGM.
Provides:
  - Mechanism A: Spatial Potential-Field Reconnection Engine (Konkol et al. 2022)
  - Mechanism B: Thermodynamic nER* Optimality Pruning Engine (Tejedor et al. 2017)
Enforces Jordan Curve Theorem planarity filtering and degree constraints.
"""

from typing import Dict, Any, Optional
import math
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

from .domain import PolarDomain
from .geometry_utils import (
    edge_intersects_graph,
    extend_leaf_to_shoreline,
    edge_too_close_to_graph,
    incident_angle_between_segments
)
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

    if current_G.number_of_nodes() < 2 or current_G.number_of_edges() == 0:
        return current_G

    R_rt = max(0.0, min(1.0, float(params.get('R_rt', domain.R_rt if domain else 0.5))))
    l0_ratio = max(0.01, float(params.get('candidate_dist_scale', 0.25)))
    c0 = max(0.0, float(params.get('reconnection_prob_base', 0.55)))
    P0 = max(0.0, float(params.get('anastomosis_prob_base', 0.0)))
    max_in = max(1, int(params.get('max_in_degree', 2)))
    max_out = max(1, int(params.get('max_out_degree', 3)))

    effective_c = P0 + c0 * R_rt
    if effective_c <= 1e-4 and R_rt <= 1e-4:
        return current_G

    r_shore = float(domain.r_shore if domain else 10000.0)
    l0 = max(10.0, l0_ratio * r_shore)
    base_sin_mu = float(params.get('loop_sinuosity_mu', params.get('base_sinuosity_mean', params.get('base_sinuosity_mu', 0.25))))
    base_sin_sigma = float(params.get('loop_sinuosity_sigma', params.get('base_sinuosity_std', params.get('base_sinuosity_sigma', 0.15))))
    coastal_sin_decay = float(params.get('coastal_sinuosity_decay', 0.30))

    nodes = list(current_G.nodes())
    existing_edges = [
        ((float(current_G.nodes[u]['x']), float(current_G.nodes[u]['y'])),
         (float(current_G.nodes[v]['x']), float(current_G.nodes[v]['y'])))
        for u, v in current_G.edges()
    ]

    coords = np.array([[float(current_G.nodes[n]['x']), float(current_G.nodes[n]['y'])] for n in nodes])
    tree = KDTree(coords)
    max_dist = max(l0, -l0 * math.log(0.02 / max(1e-4, effective_c))) if effective_c > 0.02 else l0
    pairs = tree.query_pairs(max_dist)

    candidates = []
    for i, j in pairs:
        u, v = nodes[i], nodes[j]

        # Enforce downstream orientation (r_src < r_tgt)
        r_u, r_v = float(current_G.nodes[u].get('r', 0.0)), float(current_G.nodes[v].get('r', 0.0))
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
            float(current_G.nodes[src]['r']), float(current_G.nodes[src]['theta']),
            float(current_G.nodes[tgt]['r']), float(current_G.nodes[tgt]['theta'])
        ) if domain else math.hypot(
            float(current_G.nodes[src]['x']) - float(current_G.nodes[tgt]['x']),
            float(current_G.nodes[src]['y']) - float(current_G.nodes[tgt]['y'])
        )

        d_theta = abs(float(current_G.nodes[src]['theta']) - float(current_G.nodes[tgt]['theta'])) if domain else 0.0
        p_loop = min(0.95, max(0.0, effective_c * math.exp(-dist / max(1.0, l0)) * max(0.0, math.cos(d_theta))))

        if p_loop <= 0.02:
            continue

        # Downstream monotonicity: since delta channels flow downstream (r_src < r_tgt),
        # directed cycles are impossible unless radial coordinates are equal within tolerance.
        same_r = abs(r_u - r_v) < 1e-4
        if same_r and nx.has_path(current_G, tgt, src):
            continue

        candidates.append((src, tgt, p_loop, dist, same_r))

    candidates.sort(key=lambda x: x[2], reverse=True)

    min_width = max(1e-4, float(params.get('min_channel_width', 0.1)))
    min_edge_clearance = max(10.0, float(params.get('min_edge_clearance', 0.0025 * r_shore)))
    min_bif_angle_deg = float(params.get('min_bifurcation_angle_deg', 15.0))
    min_bif_angle = math.radians(min_bif_angle_deg)
    theta_open = float(domain.theta_open if domain else math.pi)
    effective_min_bif = min(min_bif_angle, 0.40 * theta_open)
    min_sibling_dist = max(20.0, float(params.get('min_sibling_distance', 0.005 * r_shore)))

    for src, tgt, p_loop, _dist, same_r in candidates:
        if current_G.out_degree(src) >= max_out or current_G.in_degree(tgt) >= max_in:
            continue
        if same_r and nx.has_path(current_G, tgt, src):
            continue

        p1 = (float(current_G.nodes[src]['x']), float(current_G.nodes[src]['y']))
        p2 = (float(current_G.nodes[tgt]['x']), float(current_G.nodes[tgt]['y']))

        # Enforce angular divergence and spatial separation from existing outgoing branches at src
        violates_bif = False
        for succ in current_G.successors(src):
            succ_pos = (float(current_G.nodes[succ]['x']), float(current_G.nodes[succ]['y']))
            if incident_angle_between_segments(p1, p2, succ_pos) < effective_min_bif or math.hypot(p2[0] - succ_pos[0], p2[1] - succ_pos[1]) < min_sibling_dist:
                violates_bif = True
                break
        if violates_bif:
            continue

        # Enforce angular convergence from existing incoming branches at tgt
        violates_conf = False
        for pred in current_G.predecessors(tgt):
            pred_pos = (float(current_G.nodes[pred]['x']), float(current_G.nodes[pred]['y']))
            if incident_angle_between_segments(p2, p1, pred_pos) < effective_min_bif:
                violates_conf = True
                break
        if violates_conf:
            continue

        if rng.uniform(0.0, 1.0) <= p_loop:
            if not edge_intersects_graph(p1, p2, existing_edges) and not edge_too_close_to_graph(p1, p2, existing_edges, min_edge_clearance):
                straight_dist = max(10.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                r_src = float(current_G.nodes[src].get('r', math.hypot(p1[0], p1[1])))
                r_tgt = float(current_G.nodes[tgt].get('r', math.hypot(p2[0], p2[1])))
                r_mid = 0.5 * (r_src + r_tgt)
                r_ratio = min(1.0, max(0.0, r_mid / max(1.0, r_shore)))
                sin_decay_mult = max(0.0, 1.0 - coastal_sin_decay * r_ratio)
                raw_sin = float(rng.lognormal(base_sin_mu, base_sin_sigma))
                loop_sin = max(1.0, 1.0 + (raw_sin - 1.0) * sin_decay_mult)
                loop_len = straight_dist * loop_sin

                current_G.add_edge(
                    src, tgt,
                    edge_type='loop',
                    straight_distance=straight_dist,
                    sinuosity=loop_sin,
                    length=loop_len,
                    width=min_width
                )
                existing_edges.append((p1, p2))

                if current_G.out_degree(src) >= 2 and current_G.nodes[src].get('node_type') != 'apex':
                    current_G.nodes[src]['node_type'] = 'bifurcation'
                if current_G.in_degree(tgt) >= 2 and current_G.nodes[tgt].get('node_type') != 'apex':
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

    if current_G.number_of_nodes() < 2 or current_G.number_of_edges() == 0:
        return current_G

    N_target = int(params.get('N_target', G.graph.get('N_target', len(current_G.nodes()))))
    R_rt = max(0.0, min(1.0, float(params.get('R_rt', domain.R_rt if domain else 0.5))))
    l0_ratio = max(0.01, float(params.get('candidate_dist_scale', 0.35)))
    r_shore = float(domain.r_shore if domain else 10000.0)
    l0 = max(10.0, l0_ratio * r_shore)
    max_in = max(1, int(params.get('max_in_degree', 2)))
    max_out = max(1, int(params.get('max_out_degree', 3)))
    base_sin_mu = float(params.get('loop_sinuosity_mu', params.get('base_sinuosity_mean', params.get('base_sinuosity_mu', 0.25))))
    base_sin_sigma = float(params.get('loop_sinuosity_sigma', params.get('base_sinuosity_std', params.get('base_sinuosity_sigma', 0.15))))
    coastal_sin_decay = float(params.get('coastal_sinuosity_decay', 0.30))

    nodes = list(current_G.nodes())
    existing_edges = [
        ((float(current_G.nodes[u]['x']), float(current_G.nodes[u]['y'])),
         (float(current_G.nodes[v]['x']), float(current_G.nodes[v]['y'])))
        for u, v in current_G.edges()
    ]

    coords = np.array([[float(current_G.nodes[n]['x']), float(current_G.nodes[n]['y'])] for n in nodes])
    tree = KDTree(coords)
    pairs = tree.query_pairs(l0)

    all_candidates = []
    for i, j in pairs:
        u, v = nodes[i], nodes[j]

        r_u, r_v = float(current_G.nodes[u].get('r', 0.0)), float(current_G.nodes[v].get('r', 0.0))
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
            float(current_G.nodes[src]['r']), float(current_G.nodes[src]['theta']),
            float(current_G.nodes[tgt]['r']), float(current_G.nodes[tgt]['theta'])
        ) if domain else math.hypot(
            float(current_G.nodes[src]['x']) - float(current_G.nodes[tgt]['x']),
            float(current_G.nodes[src]['y']) - float(current_G.nodes[tgt]['y'])
        )

        if dist <= l0:
            all_candidates.append((src, tgt, dist))

    rng.shuffle(all_candidates)
    min_width = max(1e-4, float(params.get('min_channel_width', 0.1)))
    min_edge_clearance = max(10.0, float(params.get('min_edge_clearance', 0.0025 * r_shore)))
    min_bif_angle_deg = float(params.get('min_bifurcation_angle_deg', 15.0))
    min_bif_angle = math.radians(min_bif_angle_deg)
    theta_open = float(domain.theta_open if domain else math.pi)
    effective_min_bif = min(min_bif_angle, 0.40 * theta_open)
    min_sibling_dist = max(20.0, float(params.get('min_sibling_distance', 0.005 * r_shore)))

    for src, tgt, _dist in all_candidates:
        if current_G.out_degree(src) >= max_out or current_G.in_degree(tgt) >= max_in:
            continue
        if nx.has_path(current_G, tgt, src):
            continue
        p1 = (float(current_G.nodes[src]['x']), float(current_G.nodes[src]['y']))
        p2 = (float(current_G.nodes[tgt]['x']), float(current_G.nodes[tgt]['y']))

        # Enforce angular divergence and spatial separation from existing outgoing branches at src
        violates_bif = False
        for succ in current_G.successors(src):
            succ_pos = (float(current_G.nodes[succ]['x']), float(current_G.nodes[succ]['y']))
            if incident_angle_between_segments(p1, p2, succ_pos) < effective_min_bif or math.hypot(p2[0] - succ_pos[0], p2[1] - succ_pos[1]) < min_sibling_dist:
                violates_bif = True
                break
        if violates_bif:
            continue

        # Enforce angular convergence from existing incoming branches at tgt
        violates_conf = False
        for pred in current_G.predecessors(tgt):
            pred_pos = (float(current_G.nodes[pred]['x']), float(current_G.nodes[pred]['y']))
            if incident_angle_between_segments(p2, p1, pred_pos) < effective_min_bif:
                violates_conf = True
                break
        if violates_conf:
            continue

        if not edge_intersects_graph(p1, p2, existing_edges) and not edge_too_close_to_graph(p1, p2, existing_edges, min_edge_clearance):
            straight_dist = max(10.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            r_src = float(current_G.nodes[src].get('r', math.hypot(p1[0], p1[1])))
            r_tgt = float(current_G.nodes[tgt].get('r', math.hypot(p2[0], p2[1])))
            r_mid = 0.5 * (r_src + r_tgt)
            r_ratio = min(1.0, max(0.0, r_mid / max(1.0, r_shore)))
            sin_decay_mult = max(0.0, 1.0 - coastal_sin_decay * r_ratio)
            raw_sin = float(rng.lognormal(base_sin_mu, base_sin_sigma))
            loop_sin = max(1.0, 1.0 + (raw_sin - 1.0) * sin_decay_mult)
            loop_len = straight_dist * loop_sin

            current_G.add_edge(
                src, tgt,
                edge_type='loop',
                straight_distance=straight_dist,
                sinuosity=loop_sin,
                length=loop_len,
                width=min_width
            )
            existing_edges.append((p1, p2))

            if current_G.out_degree(src) >= 2 and current_G.nodes[src].get('node_type') != 'apex':
                current_G.nodes[src]['node_type'] = 'bifurcation'
            if current_G.in_degree(tgt) >= 2 and current_G.nodes[tgt].get('node_type') != 'apex':
                current_G.nodes[tgt]['node_type'] = 'confluence'

    current_G = apply_emergent_hydraulics(current_G, params, seed=seed)
    _, W, phi = solve_markov_flux(current_G, params)

    def calc_ner_star(W_mat: np.ndarray, phi_vec: np.ndarray) -> float:
        nonzero_mask = W_mat > 1e-12
        out_degrees = np.sum(nonzero_mask, axis=1)
        valid_branching = out_degrees > 1
        max_ner = float(np.sum(phi_vec[valid_branching] * np.log2(out_degrees[valid_branching])))

        i_indices, j_indices = np.nonzero(nonzero_mask)
        w_vals = np.clip(W_mat[i_indices, j_indices], 1e-12, 1.0)
        phi_vals = np.maximum(0.0, phi_vec[i_indices])
        ner = -float(np.sum(phi_vals * w_vals * np.log2(w_vals)))
        return float(np.nan_to_num((ner / max_ner) if max_ner > 1e-6 else 0.0, nan=0.0))

    current_ner_star = calc_ner_star(W, phi)

    max_loops_slope = float(params.get('target_max_loops_slope', 0.22))
    max_loops_intercept = float(params.get('target_max_loops_intercept', 0.05))
    max_loops_target = max(1, int(round((max_loops_intercept + max_loops_slope * R_rt) * N_target)))

    T = max(1e-4, float(params.get('sa_initial_temp', 1.0)))
    alpha = max(0.5, min(0.999, float(params.get('sa_cooling_rate', 0.975))))
    sa_steps = max(1, int(params.get('sa_max_steps', 250)))

    for _ in range(sa_steps):
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
            sa_exp_arg = min(50.0, max(-50.0, -delta_cost / max(1e-4, T)))
            if rng.uniform(0.0, 1.0) < math.exp(sa_exp_arg):
                accept = True

        if accept:
            current_G = test_G
            current_ner_star = test_ner_star

            src, tgt = edge_to_remove
            if current_G.out_degree(src) == 1 and current_G.nodes[src].get('node_type') != 'apex':
                current_G.nodes[src]['node_type'] = 'internal'
            if current_G.in_degree(tgt) <= 1 and current_G.nodes[tgt].get('node_type') != 'apex':
                current_G.nodes[tgt]['node_type'] = 'internal'

        T = max(1e-6, T * alpha)

    # Recompute existing_edges to remove pruned phantom edges before shoreline extension
    existing_edges = [
        ((float(current_G.nodes[u]['x']), float(current_G.nodes[u]['y'])),
         (float(current_G.nodes[v]['x']), float(current_G.nodes[v]['y'])))
        for u, v in current_G.edges()
    ]

    extend_leaf_to_shoreline(current_G, domain, existing_edges)
    return current_G
