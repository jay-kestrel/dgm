"""
Emergent Hydraulics, Channel Width Scaling, Conductance, and Markov Flux Solver for DGM.
"""

from typing import Dict, Any, Tuple, Optional, List
import math
import warnings
import numpy as np
import networkx as nx

warnings.filterwarnings('ignore', category=RuntimeWarning)


def _get_dag_topological_sort(G: nx.DiGraph) -> List[Any]:
    """
    Returns a topological sorting of nodes in G.
    If G contains directed cycles, extracts a DAG backbone (separating tree backbone
    from cyclic loop edges or removing cycle-causing back-edges) and returns its topological sort.
    """
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
            # If still cyclic, break cycles via DFS back-edge removal
            apex_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'apex']
            root = apex_nodes[0] if apex_nodes else (0 if 0 in G.nodes() else list(G.nodes())[0])

            visited = set()
            rec_stack = set()
            dag_edges = []

            def dfs(u):
                visited.add(u)
                rec_stack.add(u)
                for v in G.successors(u):
                    if v not in visited:
                        dag_edges.append((u, v))
                        dfs(v)
                    elif v not in rec_stack:
                        dag_edges.append((u, v))
                rec_stack.remove(u)

            dfs(root)
            for n in G.nodes():
                if n not in visited:
                    dfs(n)

            G_dfs = nx.DiGraph()
            G_dfs.add_nodes_from(G.nodes())
            G_dfs.add_edges_from(dag_edges)
            try:
                return list(nx.topological_sort(G_dfs))
            except nx.NetworkXUnfeasible:
                return list(G.nodes())


def apply_emergent_hydraulics(
    G: Optional[nx.DiGraph] = None,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    **kwargs
) -> nx.DiGraph:
    """
    Applies Murray's Law width splitting, confluence power accumulation,
    exponential tidal mouth flaring, open-channel conductance, and Markov flux throughput.
    Uses an iterative relaxation solver to guarantee Murray's Law width conservation.
    """
    if G is None:
        G = kwargs.get('G', kwargs.get('G_in', None))
    if G is None:
        raise ValueError("A NetworkX DiGraph G must be provided to apply_emergent_hydraulics.")

    if params is None:
        params = {}

    rng = np.random.default_rng(seed)

    w_apex = float(params.get('apex_width', params.get('w_apex', 250.0)))
    p_exp = float(params.get('murray_p', 1.70))
    beta = float(params.get('width_beta', params.get('width_asymmetry', 0.745)))
    R_rt = float(params.get('R_rt', 0.5))
    r_shore = float(params.get('r_shore', 10000.0))
    k_flare = float(params.get('tidal_flare_k', params.get('k_flare', 1.40)))
    manning_eta = float(params.get('manning_eta', params.get('eta', 1.65)))
    length_gamma = float(params.get('length_gamma', params.get('gamma', 0.50)))
    sigma_w = float(params.get('width_noise', params.get('width_noise_sigma', 0.30)))
    min_width = float(params.get('min_channel_width', 0.1))
    max_iter = int(params.get('relaxation_max_iter', 100))
    tol = float(params.get('relaxation_tol', 1e-6))

    # Identify exact apex node
    apex_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'apex']
    apex_node = apex_nodes[0] if apex_nodes else (0 if 0 in G.nodes() else list(G.nodes())[0])

    # Pre-generate fractional weights for Murray's law conservation at bifurcations
    branch_fractions: Dict[Tuple[Any, Any], float] = {}
    for u in G.nodes():
        out_edges = list(G.out_edges(u, data=True))
        n_out = len(out_edges)
        if n_out == 1:
            branch_fractions[(out_edges[0][0], out_edges[0][1])] = 1.0
        elif n_out >= 2:
            if isinstance(u, (int, np.integer)):
                node_seed = (seed + int(u) * 7919) % (2**31)
            else:
                node_seed = (seed + abs(hash(str(u)))) % (2**31)
            node_rng = np.random.default_rng(node_seed)

            primary_frac = node_rng.beta(beta * 4.0, max(0.1, (1.0 - beta) * 4.0))
            primary_frac = max(0.50, min(0.99, float(primary_frac)))
            other_frac = (1.0 - primary_frac) / (n_out - 1)
            raw_fracs = [primary_frac] + [other_frac] * (n_out - 1)
            node_rng.shuffle(raw_fracs)

            weights = []
            for i, (src, tgt, d) in enumerate(out_edges):
                penalty_factor = (0.40 ** p_exp) if d.get('edge_type') == 'loop' else 1.0
                weights.append(raw_fracs[i] * penalty_factor)

            total_weight = sum(weights)
            for i, (src, tgt, d) in enumerate(out_edges):
                branch_fractions[(src, tgt)] = float(weights[i] / total_weight) if total_weight > 0 else (1.0 / n_out)

    topo_nodes = _get_dag_topological_sort(G)

    # Initialize edge river widths
    for u, v in G.edges():
        G.edges[u, v]['width_river'] = 0.0

    # Iterative relaxation solver for Murray's Law width accumulation across DAG & cyclic loop edges
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
                w_u = float(w_in_sum ** (1.0 / p_exp)) if w_in_sum > 0 else (
                    w_apex if (u == apex_node or G.nodes[u].get('node_type') == 'apex') else 0.0
                )

            out_edges = list(G.out_edges(u, data=True))
            n_out = len(out_edges)
            if n_out == 0:
                continue
            for src, tgt, d in out_edges:
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

        r_u = G.nodes[u]['r']
        r_v = G.nodes[v]['r']
        r_mid = (r_u + r_v) / 2.0

        w_river = d.get('width_river', 0.0)

        # Multiplicative estuarine flaring
        flare_factor = math.exp(k_flare * max(0.0, r_mid / r_shore - 0.3))
        w_emergent = w_river * ((1.0 - R_rt) + R_rt * flare_factor)

        if 'width_noise' in d:
            noise = float(d['width_noise'])
        else:
            noise = float(edge_rng.lognormal(0.0, sigma_w))
            d['width_noise'] = noise

        w_final = max(min_width, w_emergent * noise)
        d['width'] = w_final

        # Compute Euclidean physical length
        x_u, y_u = G.nodes[u]['x'], G.nodes[u]['y']
        x_v, y_v = G.nodes[v]['x'], G.nodes[v]['y']
        straight_dist = math.sqrt((x_v - x_u)**2 + (y_v - y_u)**2)

        if 'sinuosity' in d:
            sinuosity = float(d['sinuosity'])
        else:
            sinuosity = float(edge_rng.lognormal(0.25, 0.15))
            d['sinuosity'] = sinuosity

        length = max(1.0, straight_dist * sinuosity)
        d['straight_distance'] = max(1.0, straight_dist)
        d['length'] = length

        # Open-channel conductance C_e = w_e^eta / L_e^gamma
        d['conductance'] = (w_final ** manning_eta) / (length ** length_gamma)

    return G


def solve_markov_flux(
    G: nx.DiGraph,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    eta = float(params.get('manning_eta', params.get('eta', 1.65)) if params else 1.65)

    # Conductance / Flow Partition Matrix C
    C = np.zeros((N, N))
    for u, v, d in G.edges(data=True):
        if routing_mode == 'width':
            w_val = float(d.get('width', 1.0))
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

    with np.errstate(divide='ignore', over='ignore', invalid='ignore', under='ignore'):
        try:
            phi_T = np.linalg.solve(I_Q_T, S0_T)
            if not np.all(np.isfinite(phi_T)):
                phi_T = np.dot(np.linalg.pinv(I_Q_T), S0_T)
        except np.linalg.LinAlgError:
            phi_T = np.dot(np.linalg.pinv(I_Q_T), S0_T)

        phi_T = np.nan_to_num(phi_T, nan=0.0, posinf=1.0, neginf=0.0)

        phi = np.zeros(N)
        phi[T_indices] = phi_T

        if len(A_indices) > 0:
            W_TA = W[np.ix_(T_indices, A_indices)]
            S0_A = S0[A_indices]
            phi_A = S0_A + np.dot(W_TA.T, phi_T)
            phi[A_indices] = np.nan_to_num(phi_A, nan=0.0, posinf=1.0, neginf=0.0)

    phi = np.maximum(0.0, phi)
    return C, W, phi


solve_markov_hydraulics = solve_markov_flux
