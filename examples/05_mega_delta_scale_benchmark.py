"""
Example 05: Planetary Mega-Delta Scale Benchmark (Lena, GBMD, Amazon, Niger).
Demonstrates:
  1. Benchmarking DGM at planetary scale (N >= 600 up to 3,000 nodes)
  2. Direct validation against the world's largest empirical distributary networks:
     - Lena River Delta (N = 2,914)
     - Ganges-Brahmaputra-Meghna Delta (N = 2,488)
     - Amazon River Delta (N = 1,094)
     - Niger River Delta (N = 651)
  3. Verifying that flow entropy (nER*) and entropic braiding (eBI) scale faithfully
  4. Exporting CSV, JSON, PNG, and Markdown benchmark reports
"""

import argparse
import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent
dgm_root = current_dir.parent

for p in [str(dgm_root / "src"), str(dgm_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dgm.benchmarks.mega_delta_benchmark import (
    DEFAULT_MEGA_DELTAS,
    MegaDeltaScaleBenchmark,
)


def resolve_output_dir(custom_path: str | None = None) -> Path:
    out = Path(custom_path).resolve() if custom_path else dgm_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run DGM Mega-Delta Scale Benchmark (Lena, GBMD, Amazon, Niger)"
    )
    parser.add_argument(
        "--deltas", "-d", nargs="+", default=DEFAULT_MEGA_DELTAS,
        help=f"Target mega-delta names (default: {DEFAULT_MEGA_DELTAS})"
    )
    parser.add_argument(
        "--realizations", "-r", type=int, default=2,
        help="Number of stochastic realizations per mega-delta (default: 2)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--no-pruning", action="store_true", default=False,
        help="Disable pre-hydraulic dead-end pruning"
    )
    parser.add_argument(
        "--no-simplify", action="store_true", default=False,
        help="Disable pass-through chain simplification"
    )
    parser.add_argument(
        "--confinement", "-C", type=float, default=None,
        help="Optional confinement factor override"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory for generated artifacts"
    )
    args = parser.parse_args()

    out_dir = resolve_output_dir(args.output_dir)

    benchmark = MegaDeltaScaleBenchmark(
        delta_names=args.deltas,
        realizations_per_delta=args.realizations,
        enable_pruning=not args.no_pruning,
        simplify_pass_through_nodes=not args.no_simplify,
        confinement=args.confinement,
        base_seed=args.seed,
    )

    t0 = time.time()
    benchmark.run(verbose=True)
    total_time = time.time() - t0

    # Export Artifacts
    csv_file = benchmark.export_csv(out_dir / "mega_delta_benchmark_results.csv")
    json_file = benchmark.export_json(out_dir / "mega_delta_benchmark_summary.json")
    plot_file = benchmark.export_plot(out_dir / "mega_delta_scale_benchmark.png")
    report_file = benchmark.export_report(out_dir / "MEGA_DELTA_BENCHMARK_REPORT.md")

    print("\n" + "=" * 80)
    print("BENCHMARK ARTIFACTS SUCCESSFULLY EXPORTED:")
    print(f"  - CSV Results:     {csv_file}")
    print(f"  - JSON Summary:    {json_file}")
    print(f"  - Comparison Plot: {plot_file}")
    print(f"  - Markdown Report: {report_file}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
