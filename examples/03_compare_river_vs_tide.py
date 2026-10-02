"""
Example 03: Comparing River-Dominated vs Tide-Dominated Networks.
Demonstrates:
  - Generating networks across the R_rt forcing spectrum (0.0 to 1.0)
  - Observing loop formation, mouth flaring, and topological differences
"""

import dgm


def main():
    print("=" * 60)
    print("DeltaGraphModel (DGM) - River vs. Tide Comparison")
    print("=" * 60)

    battery = dgm.MetricsBattery()
    regimes = [
        ("River-dominated (Mississippi-like)", 0.05),
        ("Mixed Fluvial-Tidal (Mahanadi-like)", 0.50),
        ("Tide-dominated (Niger/Fly-like)", 0.90),
    ]

    print(f"{'Regime':35s} | {'R_rt':5s} | {'Loops':5s} | {'Alpha':6s} | {'nER*':6s} | {'eBI':6s}")
    print("-" * 75)

    for label, r_val in regimes:
        G = dgm.generate(N=45, R_rt=r_val, seed=42)
        m = battery.evaluate_all(G)
        print(f"{label:35s} | {r_val:5.2f} | {m['betti_1']:5d} | {m['alpha_index']:6.3f} | {m['nER_star']:6.3f} | {m['eBI']:6.3f}")


if __name__ == "__main__":
    main()
