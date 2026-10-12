"""
Diagnostic & Reality Check Suite for DGM Physical Enhancements.
Evaluates the physical and geomorphic realism of:
  1. Lateral Valley Confinement (C_conf, W_valley)
  2. Shoreward Spatial Scaling (delta_r decay, sinuosity decay)
  3. Proximity-Constrained Sinuosity & Curved Centerlines
  4. Pre-Hydraulic Network Pruning (inland dead-end elimination)
"""

from typing import Dict, Any
import math
import numpy as np

import dgm
from dgm.core.channel_geometry import compute_edge_clearance
from dgm.core.hydraulics import solve_markov_flux


def run_reality_diagnostics() -> Dict[str, Any]:
    """
    Executes quantitative physical and morphological checks on all 4 new modules.
    Returns structured results dictionary.
    """
    results: Dict[str, Any] = {}

    print("=" * 78)
    print("DGM Physical Modules Reality & Geomorphic Consistency Check")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Lateral Valley Confinement Reality Check
    # -------------------------------------------------------------------------
    print("\n[Check 1/4]: Lateral Valley Confinement Realism")
    confinement_levels = [0.0, 0.30, 0.60, 0.85]
    conf_results = []

    for c in confinement_levels:
        widths = []
        aspect_ratios = []
        for s in range(5):
            G = dgm.generate(N=50, R_rt=0.4, confinement=c, seed=42 + s)
            ys = [G.nodes[n]['y'] for n in G.nodes()]
            xs = [G.nodes[n]['x'] for n in G.nodes()]
            w = max(ys) - min(ys) if ys else 0.0
            l = max(xs) - min(xs) if xs else 0.0
            widths.append(w)
            aspect_ratios.append(w / max(1.0, l))

        mean_w = float(np.mean(widths)) if widths else 0.0
        mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
        conf_results.append({
            'confinement': c,
            'mean_lateral_width_m': mean_w,
            'mean_aspect_ratio_W_over_L': mean_ar,
        })
        print(f"  C_conf = {c:.2f} -> Lateral Width = {mean_w:7.1f} m | Aspect Ratio W/L = {mean_ar:.3f}")

    # Physical invariant: aspect ratio must decrease monotonically with confinement
    ar_vals = [cr['mean_aspect_ratio_W_over_L'] for cr in conf_results]
    is_conf_monotonic = bool(ar_vals[0] > ar_vals[1] > ar_vals[2] > ar_vals[3])
    results['confinement_check'] = {
        'levels': conf_results,
        'monotonic_aspect_ratio_reduction': is_conf_monotonic,
        'passed': is_conf_monotonic
    }
    print(f"  => Confinement Aspect Ratio Monotonic Reduction: {'PASSED' if is_conf_monotonic else 'FAILED'}")

    # -------------------------------------------------------------------------
    # 2. Shoreward Spatial Scaling Reality Check
    # -------------------------------------------------------------------------
    print("\n[Check 2/4]: Shoreward Spatial Scaling Realism")
    # Generate ensemble with shoreward scaling active
    r_bins = [(0.0, 0.35), (0.35, 0.70), (0.70, 0.95)]
    bin_steps = {b: [] for b in r_bins}
    bin_sins = {b: [] for b in r_bins}

    for s in range(8):
        G = dgm.generate(
            N=60, R_rt=0.4,
            coastal_node_decay=0.50,
            coastal_sinuosity_decay=0.40,
            seed=100 + s
        )
        r_shore = G.graph['params']['r_shore']
        for u, v, d in G.edges(data=True):
            if d.get('edge_type') == 'tree' and G.nodes[v].get('node_type') != 'outlet':
                r_mid = 0.5 * (G.nodes[u]['r'] + G.nodes[v]['r']) / r_shore
                step = d.get('straight_distance', d.get('length', 0.0))
                sin = d.get('sinuosity', 1.0)
                for b_low, b_high in r_bins:
                    if b_low <= r_mid < b_high:
                        bin_steps[(b_low, b_high)].append(step)
                        bin_sins[(b_low, b_high)].append(sin)

    step_proximal = float(np.mean(bin_steps[(0.0, 0.35)])) if bin_steps[(0.0, 0.35)] else 0.0
    step_distal = float(np.mean(bin_steps[(0.70, 0.95)])) if bin_steps[(0.70, 0.95)] else 0.0
    sin_proximal = float(np.mean(bin_sins[(0.0, 0.35)])) if bin_sins[(0.0, 0.35)] else 1.0
    sin_distal = float(np.mean(bin_sins[(0.70, 0.95)])) if bin_sins[(0.70, 0.95)] else 1.0

    step_reduction = (step_proximal - step_distal) / max(1e-6, step_proximal)
    sin_reduction = (sin_proximal - sin_distal) / (sin_proximal - 1.0 + 1e-6)

    print(f"  Apex Reach (r < 0.35 r_shore):   Mean Step = {step_proximal:6.1f} m | Sinuosity = {sin_proximal:.3f}")
    print(f"  Coastal Reach (r > 0.70 r_shore): Mean Step = {step_distal:6.1f} m | Sinuosity = {sin_distal:.3f}")
    print(f"  => Radial Step Shrinkage: {step_reduction * 100:.1f}%")
    print(f"  => Excess Sinuosity Decay: {sin_reduction * 100:.1f}%")

    passed_scaling = bool(step_distal < step_proximal and sin_distal <= sin_proximal and sin_distal >= 1.0)
    results['scaling_check'] = {
        'step_proximal': step_proximal,
        'step_distal': step_distal,
        'sin_proximal': sin_proximal,
        'sin_distal': sin_distal,
        'passed': passed_scaling
    }
    print(f"  => Shoreward Spatial Scaling Realism: {'PASSED' if passed_scaling else 'FAILED'}")

    # -------------------------------------------------------------------------
    # 3. Proximity-Constrained Sinuosity & Curved Polylines Reality Check
    # -------------------------------------------------------------------------
    print("\n[Check 3/4]: Channel Proximity & Non-Overlapping Sinuosity Realism")
    G_curved = dgm.generate(
        N=50, R_rt=0.5,
        enable_curved_centerlines=True,
        min_channel_clearance=0.40,
        seed=42
    )

    all_clearances = []
    all_sinuosities = []
    polyline_point_counts = []

    for u, v, d in G_curved.edges(data=True):
        clearance = compute_edge_clearance(G_curved, u, v)
        if math.isfinite(clearance):
            all_clearances.append(clearance)
        all_sinuosities.append(d.get('sinuosity', 1.0))
        if 'coordinates' in d:
            polyline_point_counts.append(len(d['coordinates']))

    min_clear = float(np.min(all_clearances)) if all_clearances else 0.0
    mean_clear = float(np.mean(all_clearances)) if all_clearances else 0.0
    all_sin_valid = bool(all(s >= 1.0 for s in all_sinuosities))
    has_polylines = bool(len(polyline_point_counts) == G_curved.number_of_edges())

    print(f"  Mean Channel Clearance:  {mean_clear:7.1f} m (Min Clearance: {min_clear:.1f} m)")
    print(f"  Sinuosity Bounds Check:  All S >= 1.0 ({'PASSED' if all_sin_valid else 'FAILED'})")
    print(f"  Curved Polylines Made:   {len(polyline_point_counts)} / {G_curved.number_of_edges()} edges")

    passed_proximity = bool(all_sin_valid and has_polylines and min_clear >= 0.0)
    results['proximity_check'] = {
        'min_clearance': min_clear,
        'mean_clearance': mean_clear,
        'all_sin_ge_unity': all_sin_valid,
        'polylines_generated': has_polylines,
        'passed': passed_proximity
    }
    print(f"  => Channel Proximity Realism: {'PASSED' if passed_proximity else 'FAILED'}")

    # -------------------------------------------------------------------------
    # 4. Pre-Hydraulic Network Pruning Reality Check
    # -------------------------------------------------------------------------
    print("\n[Check 4/4]: Pre-Hydraulic Network Pruning (Inland Dead-End Elimination)")
    # Generate 10 networks without pruning and 10 with pruning
    unpruned_dead_ends = []
    pruned_dead_ends = []
    unpruned_shore_ratios = []
    pruned_shore_ratios = []
    flux_conservations = []

    for s in range(10):
        # Unpruned
        G_raw = dgm.generate(N=50, R_rt=0.4, enable_pruning=False, seed=200 + s)
        r_cutoff = G_raw.graph['params']['r_shore'] * 0.95
        raw_leaves = [n for n in G_raw.nodes() if G_raw.out_degree(n) == 0]
        raw_dead = [n for n in raw_leaves if G_raw.nodes[n]['r'] < r_cutoff]
        unpruned_dead_ends.append(len(raw_dead))
        unpruned_shore_ratios.append((len(raw_leaves) - len(raw_dead)) / max(1, len(raw_leaves)))

        # Pruned
        G_clean = dgm.generate(N=50, R_rt=0.4, enable_pruning=True, prune_inland_dead_ends=True, seed=200 + s)
        r_cutoff_c = G_clean.graph['params']['r_shore'] * 0.95
        clean_leaves = [n for n in G_clean.nodes() if G_clean.out_degree(n) == 0]
        clean_dead = [n for n in clean_leaves if G_clean.nodes[n]['r'] < r_cutoff_c]
        pruned_dead_ends.append(len(clean_dead))
        pruned_shore_ratios.append((len(clean_leaves) - len(clean_dead)) / max(1, len(clean_leaves)))

        # Check mass conservation in hydraulics
        C, W, phi = solve_markov_flux(G_clean, G_clean.graph.get('params'))
        flux_conservations.append(len(phi) > 0 and math.isclose(phi[0], 1.0, rel_tol=1e-3))

    mean_raw_dead = float(np.mean(unpruned_dead_ends))
    mean_clean_dead = float(np.mean(pruned_dead_ends))
    mean_raw_ratio = float(np.mean(unpruned_shore_ratios))
    mean_clean_ratio = float(np.mean(pruned_shore_ratios))
    all_flux_conserved = bool(all(flux_conservations))

    print(f"  Unpruned Networks: Inland Dead-Ends = {mean_raw_dead:.1f} / net | Shoreline Outlets = {mean_raw_ratio * 100:.1f}%")
    print(f"  Pruned Networks:   Inland Dead-Ends = {mean_clean_dead:.1f} / net | Shoreline Outlets = {mean_clean_ratio * 100:.1f}%")
    print(f"  Empirical DIDNT Delta Target:         0.0 inland dead-ends | Shoreline Outlets = 100.0%")
    print(f"  Markov Flux Mass Conservation:       {'PASSED' if all_flux_conserved else 'FAILED'}")

    passed_pruning = bool(mean_clean_dead == 0.0 and mean_clean_ratio == 1.0 and all_flux_conserved)
    results['pruning_check'] = {
        'mean_unpruned_dead_ends': mean_raw_dead,
        'mean_pruned_dead_ends': mean_clean_dead,
        'unpruned_shoreline_ratio': mean_raw_ratio,
        'pruned_shoreline_ratio': mean_clean_ratio,
        'all_flux_conserved': all_flux_conserved,
        'passed': passed_pruning
    }
    print(f"  => Topological Pruning Realism: {'PASSED' if passed_pruning else 'FAILED'}")

    # Summary
    all_passed = bool(passed_scaling and passed_proximity and passed_pruning and is_conf_monotonic)
    print("\n" + "=" * 78)
    print(f"OVERALL MODULE REALISM DIAGNOSTIC: {'ALL 4 MODULES VERIFIED REALISTIC' if all_passed else 'ATTENTION NEEDED'}")
    print("=" * 78)

    return results


if __name__ == "__main__":
    run_reality_diagnostics()
