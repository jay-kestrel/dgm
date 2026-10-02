"""
Diagnostic Metric Battery module for DeltaGraphModel (DGM).
Evaluates complete literature suite across Topological, Dynamic & Transport,
Entropic Complexity, Vulnerability, and Geomorphic physical categories.
"""

from typing import Dict, Any, List, Tuple, Optional
import warnings
import math
import numpy as np
import networkx as nx

warnings.filterwarnings('ignore', category=RuntimeWarning)

from .core.hydraulics import solve_markov_flux


def _get_apex_node(G: nx.DiGraph):
    """Identifies the apex node using attributes (node_type=='apex' or is_apex==True) or topological in_degree==0."""
    if len(G) == 0:
        return None
    for n in G.nodes():
        if G.nodes[n].get('is_apex', False) or G.nodes[n].get('node_type') == 'apex':
            return n
    zero_in = [n for n in G.nodes() if G.in_degree(n) == 0]
    if zero_in:
        return zero_in[0]
    return 0 if 0 in G.nodes() else list(G.nodes())[0]


class MetricsBattery:
    """
    Comprehensive metric diagnostic battery for evaluating synthetic delta networks.
    Computes 20 metrics across four tiers:
      1. Topological (Betti-1, alpha, Vc/Vb, LSI, RD)
      2. Dynamic & Entropic (nER*, eBI, D_net, algebraic connectivity)
      3. Vulnerability & Resilience (V_net, H_out)
      4. Geomorphic & Hydraulic Form (width CV, radial sinuosity, angular dispersion)
    """

    def __init__(self, manning_eta: float = 1.65, length_gamma: float = 0.50):
        self.eta = manning_eta
        self.gamma = length_gamma

    def evaluate_all(self, G: nx.DiGraph, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluates complete multi-tier metric suite on graph G."""
        if params is None:
            params = {'manning_eta': self.eta, 'length_gamma': self.gamma}

        topo = self.evaluate_topology(G)
        dyn = self.evaluate_dynamics(G, params)
        vuln = self.evaluate_vulnerability(G, params)
        geom = self.evaluate_geomorphology(G)
        return {**topo, **dyn, **vuln, **geom}

    def evaluate_topology(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Tier 1: Topological graph metrics."""
        N = G.number_of_nodes()
        E = G.number_of_edges()

        if N == 0:
            return {
                'num_nodes': 0, 'num_edges': 0, 'betti_0': 0, 'betti_1': 0,
                'alpha_index': 0.0, 'num_bifurcations': 0, 'num_confluences': 0,
                'ratio_vc_vb': 0.0, 'recombination_factor': 0.0, 'topological_slope': 0.0,
                'num_outlets': 0, 'resistance_distance': 1.0, 'link_sharing_index': 0.5
            }

        betti_0 = nx.number_weakly_connected_components(G)
        betti_1 = max(0, E - N + betti_0)

        denom = max(1, 2 * N - 5)
        alpha_index = float(betti_1 / denom)

        n_bif = sum(1 for n in G.nodes() if G.out_degree(n) >= 2)
        n_con = sum(1 for n in G.nodes() if G.in_degree(n) >= 2)
        ratio_vc_vb = float(n_con / max(1, n_bif))
        recombination_factor = ratio_vc_vb

        horton_inversion_index = float((n_bif - n_con) / max(1, n_bif + n_con))
        topological_slope = float(alpha_index / max(1e-3, ratio_vc_vb))

        apex_node = _get_apex_node(G)
        outlets = [n for n in G.nodes() if G.out_degree(n) == 0 and n != apex_node]
        n_outlets = max(1, len(outlets))

        # Link Sharing Index (LSI): average normalized shortest-path depth (len(path) / N)
        lsi_vals = []
        for o in outlets:
            if nx.has_path(G, apex_node, o):
                path = nx.shortest_path(G, apex_node, o)
                lsi_vals.append(len(path) / float(N))
        link_sharing_index = float(np.mean(lsi_vals)) if lsi_vals else 0.5

        # Resistance Distance (RD): effective unweighted Laplacian pseudoinverse distance
        G_und = G.to_undirected()
        try:
            nodes_list = list(G.nodes())
            idx_0 = nodes_list.index(apex_node)
            N_g = len(nodes_list)
            L_unw = nx.laplacian_matrix(G_und).toarray().astype(float)
            J_shift = (1.0 / N_g) * np.ones((N_g, N_g))
            L_shifted = L_unw + J_shift
            L_pinv = np.linalg.inv(L_shifted) - J_shift

            rd_vals = []
            for o in outlets:
                if o in G.nodes():
                    idx_o = nodes_list.index(o)
                    rd_val = L_pinv[idx_0, idx_0] + L_pinv[idx_o, idx_o] - 2 * L_pinv[idx_0, idx_o]
                    rd_vals.append(max(0.1, float(rd_val)))
            resistance_distance = float(np.mean(rd_vals)) if rd_vals else 1.0
        except Exception:
            resistance_distance = 1.0

        return {
            'num_nodes': N,
            'num_edges': E,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'betti_1_norm': float(betti_1 / max(1, N)),
            'alpha_index': alpha_index,
            'num_bifurcations': n_bif,
            'num_confluences': n_con,
            'ratio_vc_vb': ratio_vc_vb,
            'horton_inversion_index': horton_inversion_index,
            'recombination_factor': recombination_factor,
            'topological_slope': topological_slope,
            'num_outlets': n_outlets,
            'log_num_outlets': float(math.log1p(n_outlets)),
            'resistance_distance': resistance_distance,
            'link_sharing_index': link_sharing_index
        }

    def evaluate_dynamics(self, G: nx.DiGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 2: Entropic, Energetic & Markovian dynamic metrics (nER*, eBI, D_net)."""
        nodes = list(G.nodes())
        N = len(nodes)
        if N == 0:
            return {
                'nER': 0.0, 'nER_star': 0.0, 'eBI': 1.0, 'log_eBI': 0.693147,
                'algebraic_connectivity': 0.0, 'energy_dissipation': 0.0, 'log_energy_dissipation': 0.0,
                'energy_entropy_ratio': 0.0
            }

        C, W, phi = solve_markov_flux(G, params)
        n_map = {n: idx for idx, n in enumerate(nodes)}

        # Non-Local Entropy Rate (nER)
        ner = 0.0
        max_ner = 0.0
        for i in range(N):
            out_degrees = np.count_nonzero(W[i, :])
            if out_degrees > 1:
                max_ner += phi[i] * math.log2(out_degrees)
            for j in range(N):
                if W[i, j] > 1e-12:
                    ner -= phi[i] * W[i, j] * math.log2(W[i, j])

        ner_star = float(ner / max_ner) if max_ner > 1e-12 else 0.0
        ner_star = min(1.0, max(0.0, ner_star))

        # Entropic Braiding Index (eBI)
        r_coords = [G.nodes[n].get('r', 0.0) for n in nodes]
        r_max = max(r_coords) if r_coords else 0.0

        ebi_vals = []
        bi_vals = []

        if r_max > 10.0:
            transects = np.linspace(0.15 * r_max, 0.85 * r_max, 15)
            for r_t in transects:
                crossing_fluxes = []
                for u, v, d in G.edges(data=True):
                    ru, rv = G.nodes[u].get('r', 0.0), G.nodes[v].get('r', 0.0)
                    if (ru <= r_t <= rv) or (rv <= r_t <= ru):
                        i_idx, j_idx = n_map[u], n_map[v]
                        f_e = float(W[i_idx, j_idx] * phi[i_idx])
                        if f_e > 1e-12:
                            crossing_fluxes.append(f_e)

                if crossing_fluxes:
                    bi_vals.append(len(crossing_fluxes))
                    tot_f = sum(crossing_fluxes)
                    p_f = [q / tot_f for q in crossing_fluxes]
                    h_t = -sum(p * math.log2(p) for p in p_f if p > 1e-12)
                    ebi_vals.append(2.0 ** h_t)

        if ebi_vals:
            eBI = float(np.mean(ebi_vals))
            traditional_BI = float(np.mean(bi_vals))
        else:
            eBI = float(2.0 ** (ner_star * 2.0))
            traditional_BI = max(1.0, eBI * 1.5)

        log_eBI = float(math.log1p(eBI))

        # Network Energy Dissipation: D_net = sum Q_e^2 / C_e
        energy_dissipation = 0.0
        for u, v, d in G.edges(data=True):
            i_idx, j_idx = n_map[u], n_map[v]
            flow_e = W[i_idx, j_idx] * phi[i_idx]
            cond_e = max(1e-4, d.get('conductance', 1.0))
            energy_dissipation += (flow_e ** 2) / cond_e
        log_energy_dissipation = float(math.log1p(energy_dissipation))

        energy_entropy_ratio = float(ner_star / max(1e-3, energy_dissipation))

        # Algebraic connectivity lambda_2 of normalized symmetric Laplacian
        C_sym = C + C.T
        d_vals = C_sym.sum(axis=1)
        d_inv_sqrt = np.zeros_like(d_vals)
        mask = (d_vals > 1e-12) & np.isfinite(d_vals)
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d_vals[mask])
        L_norm = np.eye(N) - (d_inv_sqrt[:, None] * C_sym * d_inv_sqrt[None, :])
        L_norm = np.nan_to_num(L_norm, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            eigvals = np.linalg.eigvalsh(L_norm)
            alg_conn = float(eigvals[1]) if len(eigvals) > 1 else 0.0
        except Exception:
            alg_conn = 0.0

        return {
            'nER': float(ner),
            'nER_star': ner_star,
            'eBI': eBI,
            'log_eBI': log_eBI,
            'traditional_BI': traditional_BI,
            'algebraic_connectivity': max(0.0, alg_conn),
            'energy_dissipation': float(energy_dissipation),
            'log_energy_dissipation': log_energy_dissipation,
            'energy_entropy_ratio': energy_entropy_ratio
        }

    def evaluate_vulnerability(self, G: nx.DiGraph, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 3: Vulnerability & Resilience metrics (Global Vulnerability V_net, Outlet Flux Entropy H_out)."""
        N = G.number_of_nodes()
        if N <= 2:
            return {'global_vulnerability': 0.50, 'outlet_flux_entropy': 0.0}

        C, W, phi = solve_markov_flux(G, params)
        apex_node = _get_apex_node(G)
        outlets = [n for n in G.nodes() if G.out_degree(n) == 0 and n != apex_node]

        if not outlets:
            return {'global_vulnerability': 0.50, 'outlet_flux_entropy': 0.0}

        vuln_vals = []
        nodes = list(G.nodes())
        outlet_fluxes = []
        for o in outlets:
            o_idx = nodes.index(o) if o in nodes else 0
            o_flux = phi[o_idx] if o_idx < len(phi) else 1.0 / len(outlets)
            outlet_fluxes.append(max(0.0, float(o_flux)))
            vuln_vals.append(min(1.0, max(0.1, float(1.0 / (1.0 + math.log1p(o_flux * N))))))

        global_vulnerability = float(np.mean(vuln_vals))

        total_outlet_flux = sum(outlet_fluxes)
        if total_outlet_flux > 1e-12:
            probs = [q / total_outlet_flux for q in outlet_fluxes]
            outlet_flux_entropy = float(-sum(p * math.log2(p) for p in probs if p > 1e-12))
        else:
            outlet_flux_entropy = 0.0

        return {
            'global_vulnerability': global_vulnerability,
            'outlet_flux_entropy': outlet_flux_entropy
        }

    def evaluate_geomorphology(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Tier 4: Geomorphic form, sinuosity, dispersion & width scaling metrics."""
        widths = [d.get('width', 10.0) for u, v, d in G.edges(data=True)]
        if not widths:
            return {
                'mean_width': 0.0, 'std_width': 0.0, 'width_cv': 0.0,
                'radial_sinuosity': 1.0, 'angular_dispersion': 0.5, 'loop_length_anisotropy': 1.0
            }

        mean_w = float(np.mean(widths))
        std_w = float(np.std(widths))
        width_cv = float(std_w / max(1e-3, mean_w))

        apex_node = _get_apex_node(G)
        outlets = [n for n in G.nodes() if G.out_degree(n) == 0 and n != apex_node]
        sinuosities = []
        for o in outlets:
            r_o = G.nodes[o].get('r', 0.0)
            if r_o > 10.0 and nx.has_path(G, apex_node, o):
                path = nx.shortest_path(G, apex_node, o)
                path_len = sum(G.edges[u, v]['length'] for u, v in zip(path[:-1], path[1:]))
                sinuosities.append(path_len / r_o)

        radial_sinuosity = float(np.mean(sinuosities)) if sinuosities else 1.0

        bifurcations = [n for n in G.nodes() if G.out_degree(n) >= 2]
        dispersions = []
        for b in bifurcations:
            children = list(G.successors(b))
            theta_c = [G.nodes[c].get('theta', 0.0) for c in children]
            dispersions.append(float(np.std(theta_c)))
        angular_dispersion = float(np.mean(dispersions)) if dispersions else 0.5

        tree_lens = [d.get('length', 10.0) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'tree']
        loop_lens = [d.get('length', 10.0) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'loop']
        mean_tree = float(np.mean(tree_lens)) if tree_lens else 10.0
        mean_loop = float(np.mean(loop_lens)) if loop_lens else 10.0
        loop_length_anisotropy = float(mean_tree / max(1.0, mean_loop)) if loop_lens else 1.0

        return {
            'mean_width': mean_w,
            'std_width': std_w,
            'width_cv': width_cv,
            'radial_sinuosity': radial_sinuosity,
            'angular_dispersion': angular_dispersion,
            'loop_length_anisotropy': loop_length_anisotropy
        }


# Backwards compatibility alias
V2MetricsBattery = MetricsBattery
