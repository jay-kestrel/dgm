# Changelog

All notable changes to **DeltaGraphModel (DGM)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.2.0] - 2026-10-21

### Added
- **Topological Planarity Invariant Engine**: Rigorous Jordan Curve planarity intersection testing with 2D Axis-Aligned Bounding Box (AABB) fast spatial rejection.
- **Dual Emergent Loop Engines**:
  - *Mechanism A (Potential Field Reconnection)*: Dynamic spatial probability field governing cross-ties, crevasses, and floodplain anastomosis.
  - *Mechanism B (Thermodynamic nER\* Optimality)*: Simulated annealing link pruning optimizing Non-Local Shannon Entropy Rate.
- **Hydraulics & Murray's Law Width Conservation**:
  - Conserves cross-sectional channel capacity at bifurcations ($w_0^p = w_1^p + w_2^p$).
  - Estuarine mouth flaring and open-channel conductance calculation.
  - Transient submatrix Markov flux throughput solver: $(I - Q^T)\Phi_T = S_{0,T}$.
- **Sinuous Polyline Centerline Generation**:
  - Proximity-constrained 2D cubic Hermite and sinusoidal meander path generation without geometric overlap.
- **Network Pruning & Reach Simplification**:
  - Pre-hydraulic dead-end removal guaranteeing 100% shoreline delivery.
  - Degree-2 pass-through chain contraction matching RivGraph delineation standards.
- **20-Metric Geomorphic Battery (`MetricsBattery`)**:
  - Quantitative topological, dynamic, vulnerability, and hydraulic form metrics benchmarked against empirical deltas.
- **Global Empirical Validation Suite**:
  - 28-delta benchmark covering the DIDNT global dataset (Hariharan et al., 2022).
  - Mega-delta benchmark scaled to Lena, Ganges-Brahmaputra-Meghna, Amazon, and Niger networks ($N \ge 600 - 3,000$).
- **Command Line Interface (`dgm`)**:
  - `dgm generate`: Instant single realization synthesis and export to GraphML, GeoJSON, or image.
  - `dgm ensemble`: Stochastic ensemble sampling and aggregated statistics export.
- **Packaging & Community**:
  - Bundled YAML configurations and canonical empirical dataset in standalone wheel distribution.
  - Added GitHub Actions CI workflow supporting Python 3.10, 3.11, and 3.12.
  - Added `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

### Changed
- **Modular Dependency Footprint**:
  - Decoupled `scikit-learn`, `pandas`, and `matplotlib` from core dependencies into optional extras (`[viz]`, `[benchmarks]`, `[all]`, `[dev]`).
  - Core package now depends only on `networkx`, `numpy`, `scipy`, and `pyyaml`.
  - Added standard library `csv` fallback for CSV export when `pandas` is not installed.
- Expanded comprehensive test suite to 212 passing unit and integration tests.

---

## [0.1.0] - 2026-10-02

### Added
- Initial proof-of-concept generative model for synthetic river delta channel networks.
- Basic polar tree backbone generation and radial domain confinement.
- Initial GraphML export and minimal metric battery.
