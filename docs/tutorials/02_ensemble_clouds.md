# Tutorial 2: Sampling Ensemble Clouds

Delta environments exhibit natural variability. DGM allows sampling large ensembles where physical parameters vary across realistic bounded distributions.

## 1. Ensemble Sampling Code

```python
import dgm

# 1. Configure ensemble with stochastic parameter sampling
cfg = dgm.DGMConfig.load_default()
cfg.N = 45
cfg.R_rt = 0.65
cfg.sample_parameters = True

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
```

## 2. Using the CLI

```bash
uv run dgm ensemble -N 50 -R 0.65 -n 100 --sample-parameters -o ensemble_50.csv
```
