"""
Emergent Hydraulics, Channel Width Scaling, Conductance, and Markov Flux Solver for DGM.
"""

import math
import warnings
from typing import Any

import networkx as nx
import numpy as np
import scipy.linalg


def _get_dag_topological_sort(G: nx.DiGraph) -> list[Any]:
    """
    Returns a topological sorting of nodes in G.
    If G contains directed cycles, extracts a DAG backbone (separating tree backbone
    from cyclic loop edges or removing cycle-causing back-edges) and returns its topological sort.
    Uses iterative DFS to avoid Python recursion limit errors on deep delta networks.
    """
    if G.number_of_nodes() == 0:
        return []

    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # G contains directed cycles. Extract DAG backbone by omitting loop edges
        G_dag = nx.DiGraph()
        G_dag.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            if d.get('edge_type') != 'loop':
                G_dag.add_edge(u, v)
        try:
            return list(nx.topological_sort(G_dag))
        except nx.NetworkXUnfeasible:
            # Iterative DFS to break cycles without recursion depth limits
            apex_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'apex']
            root = apex_nodes[0] if apex_nodes else (0 if 0 in G.nodes() else next(iter(G.nodes())))

            visited = set()
            rec_stack = set()
            dag_edges = []

            for start_node in [root] + [n for n in G.nodes() if n != root]:
                if start_node in visited:
                    continue
                stack = [(start_node, iter(list(G.successors(start_node))))]
                visited.add(start_node)
                rec_stack.add(start_node)

                while stack:
                    parent_node, succ_iter = stack[-1]
                    try:
                        child_node = next(succ_iter)
                        if child_node not in visited:
                            dag_edges.append((parent_node, child_node))
                            visited.add(child_node)
                            rec_stack.add(child_node)
                            stack.append((child_node, iter(list(G.successors(child_node)))))
                        elif child_node not in rec_stack:
                            dag_edges.append((parent_node, child_node))
                    except StopIteration:
                        rec_stack.remove(parent_node)
                        stack.pop()

            G_dfs = nx.DiGraph()
            G_dfs.add_nodes_from(G.nodes())
            G_dfs.add_edges_from(dag_edges)
            try:
                return list(nx.topological_sort(G_dfs))
            except nx.NetworkXUnfeasible:
                return list(G.nodes())


def apply_emergent_hydraulics(
    G: nx.DiGraph | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    **kwargs
) -> nx.DiGraph:
    """
    Applies Murray's Law width splitting, confluence power accumulation,
    exponential tidal mouth flaring, open-channel conductance, and Markov flux throughput.
    Uses an iterative relaxation solver to guarantee Murray's Law width conservation.
    """
    if G is None:
        G = kwargs.get('G')
    if G is None:
        raise ValueError("A NetworkX DiGraph G must be provided to apply_emergent_hydraulics.")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return G

    if params is None:
        params = {}

    w_apex = max(0.1, float(params.get('apex_width', 250.0)))
    p_exp = max(0.1, float(params.get('murray_p', 1.70)))
    beta = max(0.01, min(0.99, float(params.get('width_beta', 0.745))))
    R_rt = max(0.0, min(1.0, float(params.get('R_rt', 0.5))))
    r_shore = max(1.0, float(params.get('r_shore', 10000.0)))
    k_flare = float(params.get('tidal_flare_k', 1.40))
    tidal_flare_onset = max(0.0, min(1.0, float(params.get('tidal_flare_onset', 0.30))))
    loop_flow_penalty = max(0.01, min(1.0, float(params.get('loop_flow_penalty', 0.40))))
    manning_eta = max(0.01, float(params.get('manning_eta', 1.65)))
    length_gamma = max(0.01, float(params.get('length_gamma', 0.50)))
    sigma_w = max(0.0, float(params.get('width_noise', 0.30)))
    min_width = max(1e-4, float(params.get('min_channel_width', 0.1)))
    max_iter = max(1, int(params.get('relaxation_max_iter', 15)))
    tol = max(1e-9, float(params.get('relaxation_tol', 1e-4)))
    base_sin_mu = float(params.get('base_sinuosity_mean', 0.25))
    base_sin_sigma = max(1e-4, float(params.get('base_sinuosity_std', 0.15)))
    coastal_sin_decay = float(params.get('coastal_sinuosity_decay', 0.30))

    G.graph['manning_eta'] = manning_eta
    G.graph['length_gamma'] = length_gamma

    # Identify exact apex node
    apex_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'apex']
    apex_node = apex_nodes[0] if apex_nodes else (0 if 0 in G.nodes() else next(iter(G.nodes())))

    # Pre-generate fractional weights for Murray's law conservation at bifurcations
    branch_fractions: dict[tuple[Any, Any], float] = {}
    for u in G.nodes():
        out_edges = list(G.out_edges(u, data=True))
        n_out = len(out_edges)
        if n_out == 1:
            branch_fractions[(out_edges[0][0], out_edges[0][1])] = 1.0
        elif n_out >= 2:
            node_seed = (seed + int(u) * 7919) % 2 ** 31 if isinstance(u, (int, np.integer)) else (seed + abs(hash(str(u)))) % 2 ** 31
            node_rng = np.random.default_rng(node_seed)

            primary_frac = node_rng.beta(beta * 4.0, max(0.1, (1.0 - beta) * 4.0))
            primary_frac = max(0.50, min(0.99, float(primary_frac)))
            other_frac = (1.0 - primary_frac) / (n_out - 1)
            raw_fracs = [primary_frac] + [other_frac] * (n_out - 1)
            node_rng.shuffle(raw_fracs)

            weights = []
            for i, (_, _, d) in enumerate(out_edges):
                penalty_factor = (loop_flow_penalty ** p_exp) if d.get('edge_type') == 'loop' else 1.0
                weights.append(raw_fracs[i] * penalty_factor)

            total_weight = sum(weights)
            for i, (src, tgt, _) in enumerate(out_edges):
                branch_fractions[(src, tgt)] = float(weights[i] / total_weight) if total_weight > 0 else (1.0 / n_out)

    topo_nodes = _get_dag_topological_sort(G)

    # Initialize edge river widths
    for u, v in G.edges():
        G.edges[u, v]['width_river'] = 0.0

    # Iterative relaxation solver for Murray's Law width accumulation across DAG & cyclic loop edges
    with np.errstate(divide='ignore', over='ignore', invalid='ignore', under='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for iteration in range(max_iter):
            max_change = 0.0

            for u in topo_nodes:
                in_edges = list(G.in_edges(u, data=True))
                if not in_edges:
                    w_u = w_apex if (u == apex_node or G.nodes[u].get('node_type') == 'apex') else 0.0
                else:
                    w_in_sum = sum((d.get('width_river', 0.0) ** p_exp) for _, _, d in in_edges)
                    if u == apex_node or G.nodes[u].get('node_type') == 'apex':
                        w_in_sum += w_apex ** p_exp
                    w_u = float(w_in_sum ** (1.0 / p_exp)) if w_in_sum > 1e-12 else (
                        w_apex if (u == apex_node or G.nodes[u].get('node_type') == 'apex') else 0.0
                    )

                out_edges = list(G.out_edges(u, data=True))
                n_out = len(out_edges)
                if n_out == 0:
                    continue
                for src, tgt, _ in out_edges:
                    f = branch_fractions.get((src, tgt), 1.0)
                    w_child = w_u * (f ** (1.0 / p_exp))
                    old_w = G.edges[src, tgt].get('width_river', 0.0)
                    G.edges[src, tgt]['width_river'] = w_child
                    max_change = max(max_change, abs(w_child - old_w))

            if max_change < tol and iteration > 0:
                break

        # Compute final emergent width and conductance for all edges
        for u, v, d in G.edges(data=True):
            if isinstance(u, (int, np.integer)) and isinstance(v, (int, np.integer)):
                edge_seed = (seed + int(u) * 7919 + int(v) * 6271) % (2**31)
            else:
                edge_seed = (seed + abs(hash(str((u, v))))) % (2**31)
            edge_rng = np.random.default_rng(edge_seed)

            r_u = float(G.nodes[u].get('r', 0.0))
            r_v = float(G.nodes[v].get('r', 0.0))
            r_mid = (r_u + r_v) / 2.0

            w_river = float(d.get('width_river', 0.0))

            # Multiplicative estuarine flaring with numerical exponent overflow guard
            flare_arg = min(50.0, max(0.0, k_flare * (r_mid / r_shore - tidal_flare_onset)))
            flare_factor = math.exp(flare_arg)
            w_emergent = w_river * ((1.0 - R_rt) + R_rt * flare_factor)

            if 'width_noise' in d:
                noise = float(d['width_noise'])
            else:
                noise = float(edge_rng.lognormal(0.0, sigma_w))
                d['width_noise'] = noise

            w_final = max(min_width, w_emergent * noise)
            d['width'] = w_final

            # Compute Euclidean physical length
            x_u, y_u = float(G.nodes[u].get('x', 0.0)), float(G.nodes[u].get('y', 0.0))
            x_v, y_v = float(G.nodes[v].get('x', 0.0)), float(G.nodes[v].get('y', 0.0))
            straight_dist = max(1.0, math.sqrt((x_v - x_u)**2 + (y_v - y_u)**2))

            if 'sinuosity' in d:
                sinuosity = float(d['sinuosity'])
            else:
                r_ratio = min(1.0, max(0.0, r_mid / max(1.0, r_shore)))
                raw_sin = float(edge_rng.lognormal(base_sin_mu, base_sin_sigma))
                sin_decay_mult = max(0.0, 1.0 - coastal_sin_decay * r_ratio)
                sinuosity = max(1.0, 1.0 + (raw_sin - 1.0) * sin_decay_mult)
                d['sinuosity'] = sinuosity

            sinuosity = max(1.0, sinuosity)
            length = max(1.0, straight_dist * sinuosity)
            d['straight_distance'] = straight_dist
            d['length'] = length

            # Open-channel conductance C_e = w_e^eta / L_e^gamma
            d['conductance'] = max(1e-12, (w_final ** manning_eta) / (length ** length_gamma))

    return G


def solve_markov_flux(
    G: nx.DiGraph,
    params: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the linear Markov flux throughput system using the transient submatrix formulation
    (Tejedor et al., 2017).

    Returns:
        Tuple of (C, W, phi):
            - C: (N, N) conductance / flow partition matrix
            - W: (N, N) full Markov transition probability matrix
            - phi: (N,) steady-state flux throughput vector (with phi[apex] = 1.0)
    """
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)

    n_map = {n: idx for idx, n in enumerate(nodes)}

    # Identify exact apex node
    apex_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'apex']
    apex_node = apex_nodes[0] if apex_nodes else (0 if 0 in n_map else nodes[0])
    apex_idx = n_map[apex_node]

    routing_mode = params.get('routing_mode', 'width') if params else 'width'
    eta = max(0.01, float(params.get('manning_eta', params.get('eta', 1.65)) if params else 1.65))

    with np.errstate(divide='ignore', over='ignore', invalid='ignore', under='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # Conductance / Flow Partition Matrix C
        C = np.zeros((N, N))
        for u, v, d in G.edges(data=True):
            if routing_mode == 'width':
                w_val = max(1e-4, float(d.get('width', 1.0)))
                C[n_map[u], n_map[v]] = max(0.0, float(w_val ** eta))
            else:
                C[n_map[u], n_map[v]] = max(0.0, float(d.get('conductance', 1.0)))

        # Markov transition matrix W
        W = np.zeros((N, N))
        for i in range(N):
            row_sum = np.sum(C[i, :])
            if np.isfinite(row_sum) and row_sum > 1e-12:
                W[i, :] = C[i, :] / row_sum
            else:
                W[i, i] = 1.0

        # Source flux vector S0: unit flux at apex
        S0 = np.zeros(N)
        S0[apex_idx] = 1.0

        T_indices = [idx for idx, n in enumerate(nodes) if G.out_degree(n) > 0]
        A_indices = [idx for idx, n in enumerate(nodes) if G.out_degree(n) == 0]

        if len(T_indices) == 0:
            return C, W, S0

        Q = W[np.ix_(T_indices, T_indices)]
        S0_T = S0[T_indices]

        I_Q_T = np.eye(len(T_indices)) - Q.T

        try:
            phi_T = np.linalg.solve(I_Q_T, S0_T)
            if not np.all(np.isfinite(phi_T)):
                phi_T = scipy.linalg.lstsq(I_Q_T, S0_T, rcond=1e-10)[0]
        except np.linalg.LinAlgError:
            try:
                # Fast regularized solve for near-singular Markov submatrices
                reg = 1e-10 * np.eye(len(T_indices))
                phi_T = np.linalg.solve(I_Q_T.T @ I_Q_T + reg, I_Q_T.T @ S0_T)
                if not np.all(np.isfinite(phi_T)):
                    phi_T = scipy.linalg.lstsq(I_Q_T, S0_T, rcond=1e-10)[0]
            except Exception:
                phi_T = scipy.linalg.lstsq(I_Q_T, S0_T, rcond=1e-10)[0]

        phi_T = np.maximum(0.0, np.nan_to_num(phi_T, nan=0.0, posinf=1.0, neginf=0.0))

        phi = np.zeros(N)
        phi[T_indices] = phi_T

        if len(A_indices) > 0:
            W_TA = W[np.ix_(T_indices, A_indices)]
            S0_A = S0[A_indices]
            phi_A = S0_A + np.dot(W_TA.T, phi_T)
            phi[A_indices] = np.maximum(0.0, np.nan_to_num(phi_A, nan=0.0, posinf=1.0, neginf=0.0))

    # Strictly guarantee Invariant 6: unit flux at apex and non-negativity everywhere
    phi = np.maximum(0.0, phi)
    phi[apex_idx] = 1.0
    return C, W, phi


solve_markov_hydraulics = solve_markov_flux
solve_steady_state_fluxes = solve_markov_flux

