# Tutorial 1: Quickstart

In this tutorial, you will generate your first synthetic delta network and inspect its topological and entropic metrics.

## 1. Quickstart Code

```python
import dgm

# Generate a single realization with N=50 nodes and R_rt=0.4
G = dgm.generate(N=50, R_rt=0.4, seed=42)

print(f"Generated delta with {G.number_of_nodes()} nodes and {G.number_of_edges()} channels.")

# Evaluate 20-metric geomorphic battery
battery = dgm.MetricsBattery()
metrics = battery.evaluate_all(G)

print(f"Alpha Index (Loopiness): {metrics['alpha_index']:.4f}")
print(f"Normalized Entropy Rate: {metrics['nER_star']:.4f}")
print(f"Entropic Braiding Index: {metrics['eBI']:.4f}")

# Export graph for visualization in QGIS or Gephi
dgm.to_graphml(G, "my_delta.graphml")
dgm.to_geojson(G, "my_delta.geojson")
```

## 2. Using the CLI

You can also generate deltas directly from the command line:

```bash
uv run dgm generate -N 60 -R 0.5 -s 42 -o delta_60.graphml
```

---

## Next Steps

- Proceed to [Tutorial 2: Sampling Ensemble Clouds](02_ensemble_clouds.md) to explore stochastic parameter variation.
- Review the [Python API Reference](../api_reference.md) for full method signatures.
- Understand the underlying physics in [Architecture & Mathematical Formulation](../architecture.md).
- Return to the [Documentation Index](../index.md).

