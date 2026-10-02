"""
Example 02: Running a Stochastic Ensemble Simulation with Parameter Clouds.
Demonstrates:
  1. Loading and modifying YAML configuration
  2. Sampling a cloud of realizations with stochastic parameter distributions
  3. Computing ensemble summary statistics (mean, std, percentiles)
  4. Saving metrics to CSV
"""

import dgm


def main():
    print("=" * 60)
    print("DeltaGraphModel (DGM) - Stochastic Ensemble Example")
    print("=" * 60)

    # 1. Initialize configuration with parameter distribution sampling enabled
    cfg = dgm.DGMConfig.load_default()
    cfg.N = 40
    cfg.R_rt = 0.60
    cfg.sample_parameters = True
    cfg.seed = 100

    num_realizations = 20
    print(f"Sampling {num_realizations} realizations for N={cfg.N}, R_rt={cfg.R_rt:.2f}...")

    # 2. Run ensemble generator
    ensemble = dgm.EnsembleGenerator(
        N=cfg.N,
        R_rt=cfg.R_rt,
        num_realizations=num_realizations,
        sample_parameters=True,
        base_seed=100,
        config=cfg
    )

    summary = ensemble.evaluate_ensemble()

    # 3. Print summary statistics
    print("\n--- Ensemble Metric Summary ---")
    for key in ['alpha_index', 'nER_star', 'eBI', 'ratio_vc_vb', 'global_vulnerability']:
        stat = summary['metrics'].get(key, {})
        if stat:
            print(f"{key:22s} | Mean: {stat['mean']:.4f} ± {stat['std']:.4f} "
                  f"| Range: [{stat['min']:.4f}, {stat['max']:.4f}]")

    # 4. Save results to CSV
    dgm.metrics_to_csv(summary, "outputs/ensemble_summary.csv")
    print("\nSaved ensemble metrics to outputs/ensemble_summary.csv")


if __name__ == "__main__":
    main()
