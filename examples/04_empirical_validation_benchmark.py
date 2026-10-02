"""
Example 04: Empirical Validation Benchmark Against Real-World Deltas.
Demonstrates:
  1. Loading the 28 empirical real-world deltas from the DIDNT dataset
  2. Conditioning DGM on empirical node count N and marine forcing ratio R_rt
  3. Generating stochastic ensemble clouds with parameter sampling enabled
  4. Evaluating the 20-metric geomorphic battery across empirical and synthetic networks
  5. Computing quantitative goodness-of-fit (MAE, RMSE, NRMSE, R²)
  6. Verifying cloud coverage via [p10, p90] envelopes and 95% Mahalanobis hyper-ellipsoid
  7. Exporting presentation-ready CSV, JSON, PNG, and Markdown deliverables to dgm/outputs/
"""

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure dgm and local modules are importable
current_dir = Path(__file__).resolve().parent
dgm_root = current_dir.parent
repo_root = dgm_root.parent

for p in [str(dgm_root / "src"), str(dgm_root), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dgm
from dgm.benchmarks.empirical_validation import (
    EmpiricalValidationBenchmark,
    run_empirical_benchmark,
    PRIMARY_METRICS,
    METRIC_LABELS,
)


def resolve_output_dir(custom_path: str = None) -> Path:
    """Intelligently resolves the target dgm/outputs directory."""
    if custom_path:
        out = Path(custom_path).resolve()
    else:
        # Check standard locations
        candidates = [
            dgm_root / "outputs",
            Path.cwd() / "dgm" / "outputs",
            Path.cwd() / "outputs",
        ]
        out = None
        for cand in candidates:
            if cand.parent.exists():
                out = cand
                break
        if out is None:
            out = dgm_root / "outputs"

    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run DGM Empirical Real-World Delta Validation Benchmark Suite"
    )
    parser.add_argument(
        "--realizations", "-r", type=int, default=10,
        help="Number of stochastic realizations per empirical delta (default: 10)"
    )
    parser.add_argument(
        "--max-nodes", "-m", type=int, default=75,
        help="Node budget cap for synthetic realization generation (default: 75)"
    )
    parser.add_argument(
        "--mechanism", type=str, default="potential_field",
        choices=["potential_field", "ner_optimality"],
        help="Loop creation mechanism (default: potential_field)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--enable-pruning", action="store_true", default=False,
        help="Enable pre-hydraulic dead-end spur pruning (default: False)"
    )
    parser.add_argument(
        "--simplify-chains", action="store_true", default=False,
        help="Simplify degree-2 pass-through chains (default: False)"
    )
    parser.add_argument(
        "--confinement", type=float, default=None,
        help="Override lateral valley confinement factor in [0, 0.95]"
    )
    parser.add_argument(
        "--curved", action="store_true", default=False,
        help="Enable 2D curved centerline coordinates"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to custom YAML config file"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Custom destination directory for benchmark deliverables"
    )
    args = parser.parse_args()

    out_dir = resolve_output_dir(args.output_dir)

    print("=" * 78)
    print("DeltaGraphModel (DGM) - Global Empirical Validation Benchmark")
    print("=" * 78)
    print(f"Output Directory:      {out_dir}")
    print(f"Realizations / Delta:  {args.realizations}")
    print(f"Node Budget Cap:       {args.max_nodes}")
    print(f"Loop Mechanism:        {args.mechanism}")
    print(f"Pruning Enabled:       {args.enable_pruning}")
    print(f"Simplify Chains:       {args.simplify_chains}")
    print(f"Base Random Seed:      {args.seed}")
    print("-" * 78)

    t0 = time.time()

    # Build config overrides
    overrides = {}
    if args.enable_pruning:
        overrides["enable_pruning"] = True
    if args.simplify_chains:
        overrides["simplify_pass_through_nodes"] = True
    if args.confinement is not None:
        overrides["confinement"] = args.confinement
    if args.curved:
        overrides["enable_curved_centerlines"] = True

    # 1. Initialize benchmark suite
    suite = EmpiricalValidationBenchmark(
        realizations_per_delta=args.realizations,
        max_nodes=args.max_nodes,
        loop_mechanism=args.mechanism,
        sample_parameters=True,
        base_seed=args.seed,
        config=args.config,
        **overrides,
    )

    # 2. Run benchmark across all 28 DIDNT deltas
    results = suite.run(verbose=True)

    # 3. Export deliverables
    csv_file = out_dir / "empirical_validation_results.csv"
    json_file = out_dir / "empirical_validation_summary.json"
    fig_file = out_dir / "empirical_validation_benchmark.png"
    rpt_file = out_dir / "VALIDATION_REPORT.md"

    suite.save_csv(csv_file)
    suite.save_json(json_file)
    suite.plot_benchmark(fig_file)
    suite.generate_report(rpt_file)

    dt = time.time() - t0

    print("\n" + "=" * 78)
    print("ALL DELIVERABLES SUCCESSFULLY GENERATED")
    print("=" * 78)
    print(f"1. Detailed Results Table:   {csv_file.resolve()}")
    print(f"2. Statistical JSON Summary: {json_file.resolve()}")
    print(f"3. Publication Figure:       {fig_file.resolve()}")
    print(f"4. Scientific Report:        {rpt_file.resolve()}")
    print(f"\nTotal Elapsed Time: {dt:.2f} seconds")
    print("=" * 78)


if __name__ == "__main__":
    main()
