"""
Example 01: Generating a Single Synthetic Delta Graph Realization.
Demonstrates:
  1. Instantiating DeltaGenerator with DGMConfig
  2. Synthesizing a realistic delta network
  3. Evaluating the 20-metric geomorphic battery
  4. Exporting the graph to GraphML and GeoJSON
"""

import dgm


def main():
    print("=" * 60)
    print("DeltaGraphModel (DGM) - Single Realization Example")
    print("=" * 60)

    # 1. Configure the model
    # N: target node budget
    # R_rt: River-to-Tidal forcing ratio (0.0 = river-dominated, 1.0 = tide-dominated)
    config = dgm.DGMConfig.load_default()
    config.N = 45
    config.R_rt = 0.45
    config.seed = 42

    print(f"Generating delta with N={config.N}, R_rt={config.R_rt:.2f}...")

    # 2. Generate graph
    generator = dgm.DeltaGenerator(config)
    G = generator.generate()

    # 3. Evaluate multi-tier metrics battery
    metrics = generator.evaluate_metrics(G)

    print("\n--- Network Structure ---")
    print(f"Total Nodes: {G.number_of_nodes()} (Apex: 1, Outlets: {metrics['num_outlets']})")
    print(f"Total Edges: {G.number_of_edges()}")
    print(f"Bifurcations (Vb): {metrics['num_bifurcations']} | Confluences (Vc): {metrics['num_confluences']}")
    print(f"Loops (Betti-1): {metrics['betti_1']} | Alpha Index (α): {metrics['alpha_index']:.4f}")

    print("\n--- Entropic & Hydraulic Dynamics ---")
    print(f"Normalized Shannon Entropy Rate (nER*): {metrics['nER_star']:.4f}")
    print(f"Entropic Braiding Index (eBI): {metrics['eBI']:.4f}")
    print(f"Global Network Vulnerability (V_net): {metrics['global_vulnerability']:.4f}")
    print(f"Mean Channel Width: {metrics['mean_width']:.1f} m (CV: {metrics['width_cv']:.3f})")

    # 4. Export graph
    dgm.to_graphml(G, "outputs/single_realization.graphml")
    dgm.to_geojson(G, "outputs/single_realization.geojson")
    print("\nSaved network outputs to outputs/single_realization.graphml and .geojson")


if __name__ == "__main__":
    main()
