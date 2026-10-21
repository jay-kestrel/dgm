# Tutorial 2: Sampling Ensemble Clouds

Delta environments exhibit natural variability driven by heterogeneous grain size, vegetation density, and hydrodynamic fluctuations. DGM allows sampling large ensembles where physical parameters vary across realistic bounded distributions.

> [!TIP]
> For a complete tour from installation to custom plotting, see the [End-to-End Walkthrough](../walkthrough.md).

---

## 1. Ensemble Sampling Code

```python
import matplotlib.pyplot as plt
import dgm

# 1. Configure ensemble with stochastic parameter sampling enabled
cfg = dgm.DGMConfig.load_default()
cfg.N = 45
cfg.R_rt = 0.65
cfg.sample_parameters = True

# Customize active distributions using set_parameter
cfg.set_parameter("tidal_flare_onset", (0.20, 0.45))
cfg.set_parameter("loop_flow_penalty", (0.25, 0.60))
cfg.set_parameter("apex_width", {"type": "normal", "mean": 250.0, "std": 30.0, "min": 50.0})

# 2. Run ensemble generator for 100 realizations
ensemble = dgm.EnsembleGenerator(
    N=cfg.N,
    R_rt=cfg.R_rt,
    num_realizations=100,
    config=cfg,
    base_seed=100
)

# 3. Compute cloud summary statistics
summary = ensemble.evaluate_ensemble()

print("Ensemble Metric Summary (100 realizations):")
for metric in ['alpha_index', 'nER_star', 'eBI', 'global_vulnerability']:
    stat = summary['metrics'][metric]
    print(f"  {metric:22s} | Mean: {stat['mean']:.4f} ± {stat['std']:.4f}")

# 4. Save metrics to CSV
dgm.metrics_to_csv(summary, "ensemble_100.csv")
print("Saved summary metrics to ensemble_100.csv")
```

---

## 2. Visualizing Ensemble Realizations

You can generate a side-by-side comparative panel of distinct realizations from the sampled ensemble:

```python
# Sample 3 representative realizations from the parameter space
graphs = dgm.sample_ensemble(
    N=45,
    R_rt=0.65,
    num_realizations=3,
    config=cfg,
    base_seed=42
)

fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=300)
for idx, G in enumerate(graphs):
    dgm.plot_network(
        G,
        ax=axes[idx],
        curved=True,
        theme="light",
        title=f"Ensemble Realization #{idx + 1}",
        show_stats=True,
        show_legend=(idx == 0)
    )

fig.suptitle("DGM Stochastic Ensemble Realizations (N=45, R_rt=0.65)", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig("ensemble_samples.png", bbox_inches="tight")
plt.close(fig)
print("Saved ensemble plot to ensemble_samples.png")
```

---

## 3. Using the CLI

You can also run ensemble sampling directly from the command line:

```bash
dgm ensemble -N 50 -R 0.65 -n 100 --sample-parameters -o ensemble_50.csv
```

---

## Next Steps

- Follow the [End-to-End Walkthrough](../walkthrough.md) for a comprehensive, runnable walkthrough.
- Explore the [Configuration Guide & YAML Schema](../configuration.md) to customize parameter distributions.
- Run the empirical validation benchmark (`examples/04_empirical_validation_benchmark.py`) to compare ensembles with real-world deltas.
- Return to [Tutorial 1: Quickstart](01_quickstart.md) or the [Documentation Index](../index.md).
