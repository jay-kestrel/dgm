# DeltaGraphModel (DGM)

**DeltaGraphModel (DGM)** is a modular, process-emergent generative graph framework for synthesizing realistic river delta channel networks (DCNs) and sampling large stochastic ensembles.

## Key Capabilities

1. **Topological Planarity Invariant**:
   Guarantees zero unphysical channel crossings using a Jordan Curve Theorem orientation filter and 2D bounding-box spatial indexing.
2. **Emergent Dual Loop Engines**:
   - **Mechanism A (Spatial Potential-Field Reconnection)**: Models hydrodynamic cross-tie reconnections and floodplain anastomosis.
   - **Mechanism B (Thermodynamic nER* Optimality)**: Simulates long-term topological reorganization under volume budgets by maximizing non-local entropy rate ($nER^*$).
3. **Pre-Hydraulic Simplification & Shoreline Protection**:
   Removes inland blind spurs while protecting coastal outlets via `outlet_tolerance`, and simplifies degree-2 pass-through chains with unified reach-scale sinuosity recalculation (`recalculate_reach_sinuosity`).
4. **Hydraulics & Murray's Law Conservation**:
   Conserves cross-sectional capacity at bifurcations ($w_0^p = w_1^p + w_2^p$), models estuarine tidal flaring with customizable onset distance (`tidal_flare_onset`), penalizes secondary loop resistance (`loop_flow_penalty`), and computes Manning conductance ($C_e = w^\eta / L^\gamma$).
5. **Harmonic Sinuous Centerlines & Proximity Constraints**:
   Synthesizes smooth, multi-vertex curved polyline channel reaches using harmonic sine profiles with Dirichlet boundary conditions, safely bounded away from neighboring channels via proximity clearance probes.
6. **Transient Submatrix Markov Flux Solver**:
   Solves the linear flux system $(I - Q^T)\Phi_T = S_{0,T}$ to compute steady-state discharge throughput, Shannon entropy, and vulnerability without heavy numerical morphodynamic simulation.
7. **Spatial Cartographic Visualization Suite**:
   Provides publication-grade 2D map rendering (`dgm.plot_network`), side-by-side tree-versus-network comparative figures (`dgm.plot_tree_vs_network`), and progressive multi-stage generative evolution visualization (`dgm.plot_stages`).
8. **YAML-Driven Extensibility & Dynamic Distributions**:
   All parameters and stochastic distributions are configured via YAML or dynamic Python methods (`cfg.set_parameter()`).
9. **20-Metric Geomorphic Validation Battery**:
   Evaluates topological, entropic, energetic, vulnerability, and hydraulic form metrics benchmarking directly against 28 empirical deltas from the global DIDNT dataset.

## Documentation Contents

- **[End-to-End Walkthrough (Installation to Plotting)](walkthrough.md)**: A complete, step-by-step code walkthrough from installation to configuration, running the model, extracting stages, and plotting results.
- **[Architecture & Mathematical Formulation](architecture.md)**: Detailed 6-stage generative assembly pipeline, mathematical invariants, and hydraulic equations.
- **[Configuration Guide & YAML Schema](configuration.md)**: Full parameter reference, YAML schema, inline distribution formats, and runtime overrides.
- **[Python API Reference](api_reference.md)**: Complete signature documentation for all top-level functions, classes, and visualization utilities.
- **[Tutorial 1: Quickstart](tutorials/01_quickstart.md)**: Rapid 5-minute introduction to generating a delta and viewing its metrics.
- **[Tutorial 2: Sampling Ensembles](tutorials/02_ensemble_clouds.md)**: Sampling stochastic clouds and exporting statistical distributions.

## Developer & Benchmark Ecosystem

- [Developer & Agent Guide (AGENTS.md)](../AGENTS.md)
