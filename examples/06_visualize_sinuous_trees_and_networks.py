"""
Example 06: Visualizing Sinuous Channel Centerlines, Tree Backbones, and Delta Networks.
Demonstrates:
  1. Generating delta realizations with proximity-constrained sinuous polyline centerlines
  2. Rendering high-fidelity 2D cartographic maps using `dgm.plot_network`
  3. Comparing the initial Polar Tree Backbone against the Final Network (`dgm.plot_tree_vs_network`)
  4. Visualizing progressive multi-stage build-up evolution (`dgm.plot_stages`)
  5. Comparing river-dominated vs tide-dominated channel patterns with meanders
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dgm


def main():
    print("=" * 70)
    print("DeltaGraphModel (DGM) - Spatial Network & Sinuous Channel Visualizer")
    print("=" * 70)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Single Delta Realization with Sinuous Polylines & Murray's Law
    # -------------------------------------------------------------------------
    print("\n[1/4] Generating single realization with sinuous meanders (N=45, R_rt=0.35)...")
    G_single = dgm.generate(
        N=45,
        R_rt=0.35,
        enable_curved_centerlines=True,
        min_channel_clearance=0.40,
        enable_pruning=True,
        seed=101
    )

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
    dgm.plot_network(
        G_single,
        ax=ax,
        curved=True,
        edge_width="width",
        width_scaling="sqrt",
        min_linewidth=1.0,
        max_linewidth=6.5,
        show_nodes=True,
        show_domain=True,
        orientation="cartesian",
        theme="light",
        title="DGM Synthetic River Delta with Sinuous Channel Centerlines (R_rt = 0.35)",
        show_stats=True,
        show_legend=True
    )
    p1 = output_dir / "example_sinuous_delta_network.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved single realization map to: {p1}")

    # -------------------------------------------------------------------------
    # 2. Stage 1 Tree Backbone vs. Stage 6 Final Network Comparison
    # -------------------------------------------------------------------------
    print("\n[2/4] Generating comparative Tree Backbone vs Final Network (N=50, R_rt=0.50)...")
    G_final, stages = dgm.generate(
        N=50,
        R_rt=0.50,
        enable_curved_centerlines=True,
        enable_pruning=True,
        return_stages=True,
        seed=42
    )

    fig_comp, (ax_tree, ax_net) = dgm.plot_tree_vs_network(
        G_tree=stages["tree"],
        G_final=G_final,
        curved=True,
        orientation="cartesian",
        theme="light",
        figsize=(18, 8),
        dpi=300,
        title="DGM Generative Transition: Dendritic Tree Backbone to Emergent Delta Network",
        show_stats=True,
        show_legend=True
    )
    p2 = output_dir / "example_tree_vs_network_comparison.png"
    fig_comp.savefig(p2, bbox_inches="tight")
    plt.close(fig_comp)
    print(f"  -> Saved tree vs network comparison to: {p2}")

    # -------------------------------------------------------------------------
    # 3. Progressive Multi-Stage Generative Evolution (4 Stages)
    # -------------------------------------------------------------------------
    print("\n[3/4] Generating 4-stage progressive evolution pipeline (N=40, R_rt=0.55)...")
    _, stages_4 = dgm.generate(
        N=40,
        R_rt=0.55,
        enable_curved_centerlines=True,
        enable_pruning=True,
        return_stages=True,
        seed=77
    )

    fig_stages, axes = dgm.plot_stages(
        stages=stages_4,
        curved=True,
        orientation="cartesian",
        theme="light",
        figsize=(24, 6.5),
        dpi=300,
        title="DGM Multi-Stage Generative Assembly: Tree -> Loops -> Pruning -> Hydraulics & Meanders"
    )
    p3 = output_dir / "example_progressive_stages_evolution.png"
    fig_stages.savefig(p3, bbox_inches="tight")
    plt.close(fig_stages)
    print(f"  -> Saved progressive stages evolution to: {p3}")

    # -------------------------------------------------------------------------
    # 4. Fluvial vs. Mixed vs. Tidal Forcing Spectrum with Meanders
    # -------------------------------------------------------------------------
    print("\n[4/4] Comparing River-dominated vs Tidal-dominated regimes with sinuous channels...")
    regimes = [
        ("Fluvial-Dominated (R_rt = 0.05)", 0.05, 301),
        ("Mixed Fluvial-Tidal (R_rt = 0.50)", 0.50, 302),
        ("Tide-Dominated (R_rt = 0.90)", 0.90, 303),
    ]

    fig_reg, axes_reg = plt.subplots(1, 3, figsize=(22, 7), dpi=300)
    for idx, (label, r_val, s_val) in enumerate(regimes):
        g_reg = dgm.generate(
            N=45,
            R_rt=r_val,
            enable_curved_centerlines=True,
            enable_pruning=True,
            seed=s_val
        )
        dgm.plot_network(
            g_reg,
            ax=axes_reg[idx],
            curved=True,
            theme="light",
            title=label,
            show_stats=True,
            show_legend=(idx == 0)
        )

    plt.suptitle("DGM Hydrodynamic Forcing Spectrum with Sinuous Polyline Centerlines",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    p4 = output_dir / "example_river_vs_tide_sinuous_comparison.png"
    fig_reg.savefig(p4, bbox_inches="tight")
    plt.close(fig_reg)
    print(f"  -> Saved hydrodynamic forcing comparison to: {p4}")

    print("\n" + "=" * 70)
    print("All visualization examples successfully generated in outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
