# Tutorial 1: Quickstart

In this tutorial, you will generate your first synthetic river delta network, inspect its geomorphic metrics, and plot a 2D cartographic map.

> [!TIP]
> For a detailed guide covering installation, custom YAML configurations, multi-stage pipeline extraction, and advanced plotting, see the [End-to-End Walkthrough](../walkthrough.md).

---

## 1. Quickstart Code

```python
import matplotlib.pyplot as plt
import dgm

# 1. Generate a delta realization with sinuous centerlines and pruning
G = dgm.generate(
    N=50,
    R_rt=0.45,
    enable_curved_centerlines=True,
    enable_pruning=True,
    seed=42
)

print(f"Generated delta: {G.number_of_nodes()} nodes, {G.number_of_edges()} channels.")

# 2. Evaluate the 20-metric geomorphic battery
battery = dgm.MetricsBattery()
metrics = battery.evaluate_all(G)

print(f"Alpha Index (Loopiness):          {metrics['alpha_index']:.4f}")
print(f"Normalized Entropy Rate (nER*):   {metrics['nER_star']:.4f}")
print(f"Entropic Braiding Index (eBI):    {metrics['eBI']:.4f}")
print(f"Mean Channel Sinuosity:           {metrics['mean_sinuosity']:.3f}")

# 3. Plot a publication-quality 2D map
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
dgm.plot_network(
    G,
    ax=ax,
    curved=True,
    edge_width="width",
    width_scaling="sqrt",
    theme="light",
    title="DGM Quickstart: Synthetic Delta Network (R_rt = 0.45)",
    show_stats=True,
    show_legend=True
)
fig.savefig("quickstart_delta.png", bbox_inches="tight")
plt.close(fig)
print("Saved map to quickstart_delta.png")

# 4. Export graph for GIS and network analysis
dgm.to_graphml(G, "quickstart_delta.graphml")  # For Gephi / Cytoscape
dgm.to_geojson(G, "quickstart_delta.geojson")  # For QGIS / ArcGIS
```

---

## 2. Using the CLI

You can also generate deltas and render plots directly from the command line:

```bash
# Export network topology to GraphML or GeoJSON
dgm generate -N 60 -R 0.5 --curved --enable-pruning -o delta_60.graphml
dgm generate -N 60 -R 0.5 --curved --enable-pruning -o delta_60.geojson

# Render directly to an image (.png, .pdf, or .svg)
dgm generate -N 60 -R 0.5 --curved --enable-pruning -o delta_60.png
```

---

## Next Steps

- Follow the [End-to-End Walkthrough](../walkthrough.md) for a complete step-by-step tour from installation to advanced visualization.
- Proceed to [Tutorial 2: Sampling Ensemble Clouds](02_ensemble_clouds.md) to explore stochastic parameter variation.
- Review the [Python API Reference](../api_reference.md) for full method signatures.
- Understand the underlying physics in [Architecture & Mathematical Formulation](../architecture.md).
- Return to the [Documentation Index](../index.md).
