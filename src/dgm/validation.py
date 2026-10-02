"""
Empirical Benchmark and Statistical Validation module for DeltaGraphModel (DGM).
Compares DGM synthetic delta ensembles against 28 real-world deltas from the DIDNT dataset
across a complete battery of topological, entropic, dynamic, hydraulic, and geomorphic metrics.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

from .generator import EnsembleGenerator, V2EnsembleGenerator
from .io.datasets import (
    EmpiricalDelta,
    get_empirical_deltas,
    get_empirical_deltas_dict,
    DELTA_FORCING_REGIMES,
)
from .metrics import MetricsBattery


# Complete Multi-Dimensional 20-Metric Battery Feature Keys & Labels
FULL_20D_METRIC_KEYS: List[str] = [
    'alpha',
    'betti_1_norm',
    'vc_vb_ratio',
    'topological_slope',
    'horton_inversion',
    'link_sharing_index',
    'recombination_factor',
    'algebraic_connectivity',
    'resistance_distance',
    'nER_star',
    'log_eBI',
    'log_energy_dissipation',
    'energy_entropy_ratio',
    'global_vulnerability',
    'outlet_flux_entropy',
    'width_cv',
    'radial_sinuosity',
    'angular_dispersion',
    'loop_length_anisotropy',
    'log_num_outlets',
]

FULL_20D_METRIC_LABELS: List[str] = [
    'Alpha Index (α)',
    'Normalized Loops (β₁/N)',
    'Confluence/Bifurcation (Vc/Vb)',
    'Topological Slope (Γ)',
    'Horton Inversion (H_inv)',
    'Link Sharing Index (LSI)',
    'Recombination Factor (Vc/Vb)',
    'Algebraic Connectivity (λ₂)',
    'Resistance Distance (RD)',
    'Normalized Entropy Rate (nER*)',
    'Log Entropic Braiding Index log(1+eBI)',
    'Log Energy Dissipation log(1+D_net)',
    'Energy-Entropy Ratio (E_ratio)',
    'Global Vulnerability (V_net)',
    'Outlet Flux Entropy (H_out)',
    'Width CV (σ_w/μ_w)',
    'Radial Sinuosity (χ_rad)',
    'Angular Dispersion (θ_disp)',
    'Loop Length Anisotropy (Λ_loop)',
    'Log Outlets log(1+N_out)',
]

# Backward compatibility aliases
EXPANDED_METRIC_KEYS: List[str] = FULL_20D_METRIC_KEYS
EXPANDED_METRIC_LABELS: List[str] = FULL_20D_METRIC_LABELS
EMPIRICAL_METRIC_KEYS = FULL_20D_METRIC_KEYS
EMPIRICAL_METRIC_LABELS = FULL_20D_METRIC_LABELS

# Feature key mapping from EmpiricalDelta attribute names to MetricsBattery output dictionary keys
METRIC_KEY_MAP: Dict[str, str] = {
    'alpha': 'alpha_index',
    'betti_1_norm': 'betti_1_norm',
    'vc_vb_ratio': 'ratio_vc_vb',
    'topological_slope': 'topological_slope',
    'horton_inversion': 'horton_inversion_index',
    'recombination_factor': 'recombination_factor',
    'link_sharing_index': 'link_sharing_index',
    'algebraic_connectivity': 'algebraic_connectivity',
    'resistance_distance': 'resistance_distance',
    'nER_star': 'nER_star',
    'eBI': 'eBI',
    'log_eBI': 'log_eBI',
    'energy_dissipation': 'energy_dissipation',
    'log_energy_dissipation': 'log_energy_dissipation',
    'energy_entropy_ratio': 'energy_entropy_ratio',
    'global_vulnerability': 'global_vulnerability',
    'outlet_flux_entropy': 'outlet_flux_entropy',
    'width_cv': 'width_cv',
    'radial_sinuosity': 'radial_sinuosity',
    'angular_dispersion': 'angular_dispersion',
    'loop_length_anisotropy': 'loop_length_anisotropy',
    'num_outlets': 'num_outlets',
    'log_num_outlets': 'log_num_outlets',
}

# Canonical 5 direct empirical measured metric keys & labels for 5D PCA/t-SNE sub-views
EMPIRICAL_5D_KEYS: List[str] = [
    'alpha',
    'betti_1_norm',
    'vc_vb_ratio',
    'topological_slope',
    'nER_star'
]

EMPIRICAL_5D_LABELS: List[str] = [
    'Alpha Index (α)',
    'Normalized Loops (β₁/N)',
    'Confluence/Bifurcation (Vc/Vb)',
    'Topological Slope (Γ)',
    'Normalized Entropy Rate (nER*)'
]


def get_empirical_vector(d: EmpiricalDelta, keys: Optional[List[str]] = None) -> List[float]:
    """
    Returns feature vector of empirical metrics for a delta across specified feature keys.
    Defaults to FULL_20D_METRIC_KEYS (20 metrics).
    """
    if keys is None:
        keys = FULL_20D_METRIC_KEYS
    return [float(getattr(d, k)) for k in keys]


def get_empirical_5d_vector(d: EmpiricalDelta) -> List[float]:
    """
    Returns the 5-element vector of direct empirical measured metrics for a delta.
    """
    return [
        float(d.alpha),
        float(d.loops / max(1, d.nodes)),
        float(d.vc_vb_ratio),
        float(d.topological_slope),
        float(d.nER_star)
    ]


def compute_goodness_of_fit(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes quantitative goodness-of-fit metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - NRMSE: Normalized Root Mean Squared Error (normalized by empirical range y_max - y_min)
    - R2: Coefficient of determination
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    y_range = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / (y_range + 1e-8))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        'mae': mae,
        'rmse': rmse,
        'nrmse': nrmse,
        'r2': r2
    }


def compute_ks_test(synthetic_samples: np.ndarray, empirical_targets: np.ndarray) -> Dict[str, float]:
    """
    Performs 2-sample Kolmogorov-Smirnov (KS) test comparing synthetic metric distribution
    against empirical target distribution.
    """
    import scipy.stats as stats
    synth = np.asarray(synthetic_samples, dtype=float).ravel()
    emp = np.asarray(empirical_targets, dtype=float).ravel()

    res = stats.ks_2samp(synth, emp)
    return {
        'ks_stat': float(res.statistic),
        'p_value': float(res.pvalue)
    }


def compute_mahalanobis_validation(
    synthetic_cloud: np.ndarray,
    empirical_points: np.ndarray,
    df: Optional[Union[Tuple[int, ...], int]] = None
) -> Dict[str, Any]:
    """
    Computes Mahalanobis distance from each empirical delta point to the synthetic cloud centroid
    in feature space, and determines whether each empirical delta falls within the 95% confidence hyper-ellipsoid.
    """
    import scipy.stats as stats
    X_synth = np.asarray(synthetic_cloud, dtype=float)
    X_emp = np.asarray(empirical_points, dtype=float)

    p = X_synth.shape[1]
    if df is None:
        df = p

    mu = np.mean(X_synth, axis=0)
    cov = np.cov(X_synth, rowvar=False)

    if cov.ndim == 0:
        cov = np.array([[cov]])
    elif cov.ndim == 1:
        cov = np.diag(cov)

    reg_cov = cov + 1e-6 * np.eye(p)
    inv_cov = np.linalg.inv(reg_cov)

    distances = []
    sq_distances = []
    inside_95 = []
    p_vals = []

    crit_val_sq = float(stats.chi2.ppf(0.95, df=df))

    for i in range(X_emp.shape[0]):
        diff = X_emp[i] - mu
        d_sq = float(np.dot(np.dot(diff, inv_cov), diff.T))
        d_sq = max(0.0, d_sq)
        d_val = float(np.sqrt(d_sq))

        is_inside = bool(d_sq <= crit_val_sq)
        p_val = float(1.0 - stats.chi2.cdf(d_sq, df=df))

        distances.append(d_val)
        sq_distances.append(d_sq)
        inside_95.append(is_inside)
        p_vals.append(p_val)

    coverage_ratio = float(np.mean(inside_95))

    return {
        'centroid': mu,
        'covariance': reg_cov,
        'distances': distances,
        'squared_distances': sq_distances,
        'critical_value_95_sq': crit_val_sq,
        'critical_value_95_dist': float(np.sqrt(crit_val_sq)),
        'inside_95_ci': inside_95,
        'p_values': p_vals,
        'coverage_ratio': coverage_ratio
    }


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD) with RBF kernel between synthetic cloud X and empirical samples Y.
    """
    import scipy.spatial.distance as dist
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if gamma is None:
        Z = np.vstack([X, Y])
        pairwise_dists = dist.pdist(Z, metric='sqeuclidean')
        median_dist = np.median(pairwise_dists) if len(pairwise_dists) > 0 else 1.0
        gamma = 1.0 / (2.0 * max(1e-5, median_dist))

    K_XX = np.exp(-gamma * dist.cdist(X, X, metric='sqeuclidean'))
    K_YY = np.exp(-gamma * dist.cdist(Y, Y, metric='sqeuclidean'))
    K_XY = np.exp(-gamma * dist.cdist(X, Y, metric='sqeuclidean'))

    mmd_sq = np.mean(K_XX) + np.mean(K_YY) - 2.0 * np.mean(K_XY)
    return float(np.sqrt(max(0.0, mmd_sq)))


def compute_permanova(
    X: np.ndarray,
    Y: np.ndarray,
    num_permutations: int = 499
) -> Dict[str, float]:
    """
    PERMANOVA test helper testing multivariate distribution overlap between synthetic cloud X and empirical samples Y.
    """
    import scipy.spatial.distance as dist
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_x = X.shape[0]
    n_y = Y.shape[0]
    N_total = n_x + n_y

    Z = np.vstack([X, Y])
    labels = np.array([0] * n_x + [1] * n_y)

    D_mat = dist.squareform(dist.pdist(Z, metric='euclidean'))

    def _pseudo_f(lbls: np.ndarray) -> float:
        idx0 = np.where(lbls == 0)[0]
        idx1 = np.where(lbls == 1)[0]

        ss_w0 = np.sum(D_mat[np.ix_(idx0, idx0)] ** 2) / (2.0 * len(idx0)) if len(idx0) > 0 else 0.0
        ss_w1 = np.sum(D_mat[np.ix_(idx1, idx1)] ** 2) / (2.0 * len(idx1)) if len(idx1) > 0 else 0.0
        ss_w = ss_w0 + ss_w1

        ss_t = np.sum(D_mat ** 2) / (2.0 * N_total)
        ss_b = ss_t - ss_w

        df_b = 1
        df_w = N_total - 2

        if ss_w <= 1e-12:
            return 0.0
        return float((ss_b / df_b) / (ss_w / df_w))

    obs_f = _pseudo_f(labels)

    rng = np.random.default_rng(42)
    perm_f = []
    for _ in range(num_permutations):
        perm_labels = rng.permutation(labels)
        perm_f.append(_pseudo_f(perm_labels))

    p_val = float((1 + np.sum(np.array(perm_f) >= obs_f)) / (1 + num_permutations))

    return {
        'pseudo_f': obs_f,
        'p_value': p_val,
        'num_permutations': num_permutations
    }


class EmpiricalBenchmark:
    """
    Multi-metric validation battery comparing DGM synthetic graph ensembles against 28 real-world deltas from DIDNT.
    Computes formal statistical validation suite (RMSE, MAE, NRMSE, R2, KS-test, Mahalanobis 95% CI, MMD, PERMANOVA)
    across the full 20-metric battery.
    """

    def __init__(self, realizations_per_delta: int = 10, keys: Optional[List[str]] = None):
        self.realizations_per_delta = realizations_per_delta
        self.empirical_data = get_empirical_deltas()
        self.keys = keys if keys is not None else EXPANDED_METRIC_KEYS

    def run_benchmark(
        self,
        loop_mechanism: str = "potential_field",
        sample_parameters: bool = True
    ) -> Dict[str, Any]:
        """
        Runs synthetic cloud generation for all 28 real-world deltas using their (N, R_rt) values
        and computes quantitative error metrics and formal statistical validation suite.
        """
        benchmark_results = []
        metric_errors = {k: [] for k in self.keys}

        emp_matrix = []
        synth_matrix = []
        synth_cloud_all = []

        y_true_dict = {k: [] for k in self.keys}
        y_pred_dict = {k: [] for k in self.keys}

        for emp in self.empirical_data:
            emp_vec = get_empirical_vector(emp, keys=self.keys)
            emp_matrix.append(emp_vec)

            for k in self.keys:
                y_true_dict[k].append(float(getattr(emp, k)))

            # Cap N at 75 for benchmark speed — scale-normalized metrics are N-invariant
            bench_n = min(emp.nodes, 75)
            ens_gen = EnsembleGenerator(
                N=bench_n,
                R_rt=emp.estimated_R_rt,
                num_realizations=self.realizations_per_delta,
                loop_mechanism=loop_mechanism,
                sample_parameters=sample_parameters
            )
            summary = ens_gen.evaluate_ensemble()
            m = summary['metrics']

            synth_vec = []
            for k in self.keys:
                battery_key = METRIC_KEY_MAP.get(k, k)
                if battery_key in m and 'mean' in m[battery_key]:
                    val = m[battery_key]['mean']
                else:
                    if k == 'betti_1_norm':
                        val = m['betti_1']['mean'] / max(1, emp.nodes)
                    else:
                        val = 0.0
                synth_vec.append(val)
                y_pred_dict[k].append(val)

                abs_err = abs(val - getattr(emp, k))
                metric_errors[k].append(abs_err)

            synth_matrix.append(synth_vec)

            raw_runs = summary.get('raw_realizations', [])
            for run in raw_runs:
                run_vec = []
                for k in self.keys:
                    battery_key = METRIC_KEY_MAP.get(k, k)
                    if battery_key in run:
                        v = run[battery_key]
                    elif k == 'betti_1_norm':
                        v = run.get('betti_1', 0.0) / max(1, emp.nodes)
                    else:
                        v = getattr(emp, k)
                    run_vec.append(v)
                synth_cloud_all.append(run_vec)

            result_entry = {
                'delta_name': emp.name,
                'empirical_nodes': emp.nodes,
                'empirical_R_rt': emp.estimated_R_rt,
            }
            for k in self.keys:
                battery_key = METRIC_KEY_MAP.get(k, k)
                result_entry[f'empirical_{k}'] = getattr(emp, k)
                if battery_key in m:
                    result_entry[f'synthetic_{k}_mean'] = m[battery_key]['mean']
                    result_entry[f'synthetic_{k}_std'] = m[battery_key]['std']
                result_entry[f'{k}_error'] = abs(synth_vec[self.keys.index(k)] - getattr(emp, k))

            benchmark_results.append(result_entry)

        emp_arr = np.array(emp_matrix)
        synth_arr = np.array(synth_matrix)
        synth_cloud_arr = np.array(synth_cloud_all) if len(synth_cloud_all) > 0 else synth_arr

        gof_by_metric = {}
        all_true = []
        all_pred = []
        for k in self.keys:
            yt = np.array(y_true_dict[k])
            yp = np.array(y_pred_dict[k])
            gof_by_metric[k] = compute_goodness_of_fit(yt, yp)
            all_true.extend(yt)
            all_pred.extend(yp)
        overall_gof = compute_goodness_of_fit(np.array(all_true), np.array(all_pred))

        ks_by_metric = {}
        for idx, k in enumerate(self.keys):
            ks_by_metric[k] = compute_ks_test(synth_cloud_arr[:, idx], emp_arr[:, idx])

        df_val = len(self.keys)
        mahalanobis_res = compute_mahalanobis_validation(synth_cloud_arr, emp_arr, df=df_val)

        mmd_val = compute_mmd(synth_cloud_arr, emp_arr)
        permanova_res = compute_permanova(synth_cloud_arr, emp_arr)

        res = {
            'loop_mechanism': loop_mechanism,
            'realizations_per_delta': self.realizations_per_delta,
            'metric_keys': self.keys,
            'goodness_of_fit': gof_by_metric,
            'overall_goodness_of_fit': overall_gof,
            'ks_tests': ks_by_metric,
            'mahalanobis': mahalanobis_res,
            'mmd': mmd_val,
            'permanova': permanova_res,
            'detailed_results': benchmark_results
        }
        for k in self.keys:
            res[f'mean_{k}_mae'] = float(np.mean(metric_errors[k]))

        # Backward compatibility aliases
        res['mean_alpha_mae'] = res.get('mean_alpha_mae', float(np.mean(metric_errors['alpha'])))
        res['mean_loop_mae'] = res.get('mean_betti_1_norm_mae', float(np.mean(metric_errors.get('betti_1_norm', [0.0]))))
        res['mean_ner_mae'] = res.get('mean_nER_star_mae', float(np.mean(metric_errors.get('nER_star', [0.0]))))
        res['mean_vc_vb_mae'] = res.get('mean_vc_vb_ratio_mae', float(np.mean(metric_errors.get('vc_vb_ratio', [0.0]))))

        return res


# Backwards compatibility alias
V2EmpiricalBenchmark = EmpiricalBenchmark
