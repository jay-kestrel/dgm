# DeltaGraphModel (DGM)

**DeltaGraphModel (DGM)** is a modular, process-emergent generative graph framework for synthesizing realistic river delta channel networks (DCNs) and sampling large stochastic ensembles.

## Key Capabilities

1. **Topological Planarity Invariant**:
   Guarantees zero unphysical channel crossings using a Jordan Curve Theorem orientation filter.
2. **Emergent Dual Loop Engines**:
   - **Mechanism A (Spatial Potential-Field Reconnection)**: Models hydrodynamic cross-tie reconnections and floodplain anastomosis.
   - **Mechanism B (Thermodynamic nER* Optimality)**: Simulates long-term topological reorganization under volume budgets by maximizing non-local entropy rate ($nER^*$).
3. **Hydraulics & Murray's Law Conservation**:
   Calculates emergent channel widths conserving cross-sectional capacity at bifurcations ($w_0^p = w_1^p + w_2^p$), models estuarine tidal flaring, and computes conductance.
4. **Transient Submatrix Markov Flux Solver**:
   Solves the linear flux system $(I - Q^T)\Phi_T = S_{0,T}$ to compute steady-state discharge throughput, Shannon entropy, and vulnerability without heavy numerical morphodynamic simulation.
5. **YAML-Driven Extensibility**:
   All parameters and stochastic distributions are configured via YAML, enabling seamless customization and extension.
6. **20-Metric Geomorphic Validation Battery**:
   Evaluates topological, entropic, energetic, vulnerability, and hydraulic form metrics benchmarking directly against 28 empirical deltas from the DIDNT dataset.

## Documentation Contents

- [Architecture & Mathematical Formulation](architecture.md)
- [Configuration Guide & YAML Schema](configuration.md)
- [Parameter Assumptions & Literature Justifications](parameter_justifications.md)
- [Python API Reference](api_reference.md)
- [Tutorial 1: Quickstart](tutorials/01_quickstart.md)
- [Tutorial 2: Sampling Ensembles](tutorials/02_ensemble_clouds.md)
