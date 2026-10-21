"""
Diagnostic Metric Battery module for DeltaGraphModel (DGM).
Evaluates complete literature suite across Topological, Dynamic & Transport,
Entropic Complexity, Vulnerability, and Geomorphic physical categories.
"""

import itertools
import math
import warnings
from typing import Any

import networkx as nx
import numpy as np

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
    return 0 if 0 in G.nodes() else next(iter(G.nodes()))


def _clean_val(v: Any, default: float = 0.0) -> float:
    """Ensures value is a finite float, replacing NaN, Inf, and None with a robust default."""
    if v is None:
        return float(default)
    try:
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            return float(default)
        return val
    except (ValueError, TypeError):
        return float(default)


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

    def evaluate_all(self, G: nx.DiGraph, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluates complete multi-tier metric suite on graph G."""
        if params is None:
            params = {'manning_eta': self.eta, 'length_gamma': self.gamma}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            markov_sol = solve_markov_flux(G, params) if G.number_of_nodes() > 0 else None
            topo = self.evaluate_topology(G)
            dyn = self.evaluate_dynamics(G, params, markov_solution=markov_sol)
            vuln = self.evaluate_vulnerability(G, params, markov_solution=markov_sol)
            geom = self.evaluate_geomorphology(G)

        combined = {**topo, **dyn, **vuln, **geom}

        # Sanitize all outputs to guarantee clean, finite numeric returns
        default_one_keys = {
            'eBI', 'traditional_BI', 'radial_sinuosity',
            'loop_length_anisotropy', 'resistance_distance'
        }
        clean_res: dict[str, Any] = {}
        for k, v in combined.items():
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                clean_res[k] = int(v)
            else:
                default_val = 1.0 if k in default_one_keys else 0.0
                clean_res[k] = _clean_val(v, default=default_val)
        return clean_res

    def evaluate_topology(self, G: nx.DiGraph) -> dict[str, Any]:
        """
        Tier 1: Topological graph metrics.
        Computes node/edge counts, Betti numbers (betti_0, betti_1), alpha index,
        junction ratios (num_bifurcations, num_confluences, ratio_vc_vb, recombination_factor),
        horton_inversion_index, topological_slope, num_outlets, log_num_outlets, resistance_distance,
        and link_sharing_index.

        Link Sharing Index (LSI) measures network tree-like branch depth and topological
        path redundancy as the mean normalized shortest-path hop distance from apex to all
        shoreline outlets: (1 / |outlets|) * sum((d(apex, o) + 1) / N).
        """
        N = G.number_of_nodes()
        E = G.number_of_edges()

        if N == 0:
            return {
                'num_nodes': 0, 'num_edges': 0, 'betti_0': 0, 'betti_1': 0,
                'alpha_index': 0.0, 'num_bifurcations': 0, 'num_confluences': 0,
                'ratio_vc_vb': 0.0, 'recombination_factor': 0.0, 'topological_slope': 0.0,
                'num_outlets': 0, 'resistance_distance': 1.0, 'link_sharing_index': 0.5
            }

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
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
            n_outlets = len(outlets)

            # Link Sharing Index (LSI): average normalized shortest-path depth ((len(path) + 1) / N)
            sp_lengths = nx.single_source_shortest_path_length(G, apex_node) if apex_node is not None else {}
            lsi_vals = [(sp_lengths[o] + 1) / float(N) for o in outlets if o in sp_lengths]
            link_sharing_index = float(np.mean(lsi_vals)) if lsi_vals else 0.5

            # Resistance Distance (RD): effective unweighted Laplacian pseudoinverse distance
            G_und = G.to_undirected()
            try:
                nodes_list = list(G.nodes())
                node_to_idx = {n: idx for idx, n in enumerate(nodes_list)}
                idx_0 = node_to_idx[apex_node]
                N_g = len(nodes_list)
                L_unw = nx.laplacian_matrix(G_und, nodelist=nodes_list).toarray().astype(float)
                J_shift = (1.0 / N_g) * np.ones((N_g, N_g))
                L_shifted = L_unw + J_shift
                L_pinv = np.linalg.inv(L_shifted) - J_shift

                rd_vals = []
                for o in outlets:
                    idx_o = node_to_idx.get(o)
                    if idx_o is not None:
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

    def evaluate_dynamics(
        self,
        G: nx.DiGraph,
        params: dict[str, Any],
        markov_solution: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Tier 2: Entropic, Energetic & Markovian dynamic metrics (nER*, eBI, D_net)."""
        nodes = list(G.nodes())
        N = len(nodes)
        if N == 0:
            return {
                'nER': 0.0, 'nER_star': 0.0, 'eBI': 1.0, 'log_eBI': 0.693147,
                'algebraic_connectivity': 0.0, 'energy_dissipation': 0.0, 'log_energy_dissipation': 0.0,
                'energy_entropy_ratio': 0.0
            }

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if markov_solution is not None:
                C, W, phi = markov_solution
            else:
                C, W, phi = solve_markov_flux(G, params)
            n_map = {n: idx for idx, n in enumerate(nodes)}

            # Non-Local Entropy Rate (nER) - Vectorized over non-zero transitions
            nonzero_mask = W > 1e-12
            out_degrees = np.sum(nonzero_mask, axis=1)
            valid_branching = out_degrees > 1
            max_ner = float(np.sum(phi[valid_branching] * np.log2(out_degrees[valid_branching])))

            i_indices, j_indices = np.nonzero(nonzero_mask)
            w_vals = W[i_indices, j_indices]
            phi_vals = phi[i_indices]
            ner = -float(np.sum(phi_vals * w_vals * np.log2(w_vals)))

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
                    for u, v in G.edges():
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

    def evaluate_vulnerability(
        self,
        G: nx.DiGraph,
        params: dict[str, Any],
        markov_solution: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Tier 3: Vulnerability & Resilience metrics (Global Vulnerability V_net, Outlet Flux Entropy H_out).

        Global Vulnerability (V_net) measures network fragility to channel disruptions based on
        the distribution of outlet Markov fluxes. For each shoreline outlet o with flux Phi_o,
        the vulnerability contribution is v_o = clamp(1 / (1 + ln(1 + Phi_o * N)), [0.1, 1.0]).
        V_net is the mean across all outlets: V_net = (1 / |outlets|) * sum(v_o).
        Higher V_net indicates elevated sensitivity to single-channel failure.

        Outlet Flux Entropy (H_out) measures the Shannon entropy in bits of the normalized
        flux distribution across shoreline outlets: -sum(p_o * log2(p_o)).
        """
        N = G.number_of_nodes()
        if N <= 2:
            return {'global_vulnerability': 0.50, 'outlet_flux_entropy': 0.0}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if markov_solution is not None:
                _C, _W, phi = markov_solution
            else:
                _C, _W, phi = solve_markov_flux(G, params)
            apex_node = _get_apex_node(G)
            outlets = [n for n in G.nodes() if G.out_degree(n) == 0 and n != apex_node]

            if not outlets:
                return {'global_vulnerability': 0.50, 'outlet_flux_entropy': 0.0}

            vuln_vals = []
            nodes = list(G.nodes())
            node_to_idx = {n: idx for idx, n in enumerate(nodes)}
            outlet_fluxes = []
            for o in outlets:
                o_idx = node_to_idx.get(o, 0)
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

    def evaluate_geomorphology(self, G: nx.DiGraph) -> dict[str, Any]:
        """
        Tier 4: Geomorphic form, sinuosity, dispersion & width scaling metrics.
        Computes mean_width, std_width, width_cv, radial_sinuosity, angular_dispersion,
        and loop_length_anisotropy.

        Angular dispersion computes the standard deviation of local bifurcation branch
        deflection angles relative to the incoming channel vector at each branching junction,
        averaged across all junctions with out-degree >= 2.
        """
        widths = [d.get('width', 10.0) for u, v, d in G.edges(data=True)]
        if not widths:
            return {
                'mean_width': 0.0, 'std_width': 0.0, 'width_cv': 0.0,
                'radial_sinuosity': 1.0, 'angular_dispersion': 0.5, 'loop_length_anisotropy': 1.0
            }

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_w = float(np.mean(widths))
            std_w = float(np.std(widths))
            width_cv = float(std_w / max(1e-3, mean_w))

            apex_node = _get_apex_node(G)
            outlets = [n for n in G.nodes() if G.out_degree(n) == 0 and n != apex_node]
            paths = nx.single_source_shortest_path(G, apex_node) if apex_node is not None else {}
            sinuosities = []
            for o in outlets:
                r_o = G.nodes[o].get('r', 0.0)
                if r_o > 10.0 and o in paths:
                    path = paths[o]
                    path_len = sum(
                        float(G.edges[u, v].get(
                            'length',
                            math.hypot(
                                G.nodes[v].get('x', 0.0) - G.nodes[u].get('x', 0.0),
                                G.nodes[v].get('y', 0.0) - G.nodes[u].get('y', 0.0)
                            )
                        ))
                        for u, v in itertools.pairwise(path)
                    )
                    if path_len > 0:
                        sinuosities.append(path_len / r_o)

            radial_sinuosity = float(np.mean(sinuosities)) if sinuosities else 1.0

            bifurcations = [n for n in G.nodes() if G.out_degree(n) >= 2]
            dispersions = []
            for b in bifurcations:
                xb, yb = float(G.nodes[b].get('x', 0.0)), float(G.nodes[b].get('y', 0.0))
                preds = list(G.predecessors(b))
                if preds:
                    xp, yp = float(G.nodes[preds[0]].get('x', 0.0)), float(G.nodes[preds[0]].get('y', 0.0))
                    v_in_x, v_in_y = xb - xp, yb - yp
                else:
                    v_in_x, v_in_y = xb, yb

                if math.hypot(v_in_x, v_in_y) < 1e-6:
                    v_in_x, v_in_y = 0.0, 1.0
                in_angle = math.atan2(v_in_y, v_in_x)

                branch_deflections = []
                for c in G.successors(b):
                    xc, yc = float(G.nodes[c].get('x', 0.0)), float(G.nodes[c].get('y', 0.0))
                    out_angle = math.atan2(yc - yb, xc - xb)
                    deflection = (out_angle - in_angle + math.pi) % (2.0 * math.pi) - math.pi
                    branch_deflections.append(deflection)

                if len(branch_deflections) >= 2:
                    dispersions.append(float(np.std(branch_deflections)))

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
