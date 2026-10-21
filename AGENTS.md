# AGENTS.md: Developer & Agent Guide for DeltaGraphModel (DGM)

Welcome, AI agent! This document specifies the architectural conventions, testing protocols, configuration rules, and development guidelines for interacting with the `dgm/` repository.

---

## 1. Project Mission & Identity

**DeltaGraphModel (DGM)** is a fast, forward generative graph framework for synthesizing river delta channel networks (DCNs) and generating large stochastic ensembles.
- **Acronym**: DGM
- **Package Name**: `delta-graph-model`
- **Import Namespace**: `import dgm`
- **Core Philosophy**: Fast, purely graph-theoretic and topological delta synthesis (nodes, edges, bifurcations, confluences, loops, and flux routing) without heavy Navier-Stokes or 3D hydrodynamic numerical solvers.

---

## 2. Directory Layout & Architecture

```
dgm/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI matrix (Python 3.10, 3.11, 3.12, 3.13)
├── config/
│   └── default_config.yaml    # Master YAML configuration file (root copy)
├── src/
│   └── dgm/
│       ├── __init__.py        # Public exports: generate(), sample_ensemble(), DGMConfig, etc.
│       ├── config.py          # DGMConfig dataclass & parameter distribution engine
│       ├── default_config.yaml# Packaged default YAML configuration
│       ├── generator.py       # Facades: DeltaGenerator & EnsembleGenerator
│       ├── metrics.py         # 20-metric geomorphic battery (MetricsBattery)
│       ├── cli.py             # Command line interface (dgm generate, dgm ensemble)
│       ├── validation.py      # Statistical validation (Mahalanobis, KS-test, MMD, PERMANOVA)
│       ├── visualization.py   # Publication cartography, sinuous channel plots, stages
│       ├── core/
│       │   ├── domain.py      # PolarDomain (confinement, lateral valley bounds)
│       │   ├── geometry_utils.py # Jordan Curve planarity & 2D AABB fast rejection
│       │   ├── backbone.py    # Polar growth tree & shoreward spatial scaling
│       │   ├── loops.py       # Emergent loop engines (potential_field & ner_optimality)
│       │   ├── pruning.py     # Dead-end pruning & RivGraph-consistent reach simplification
│       │   ├── hydraulics.py  # Murray's law relaxation, flaring, conductance, Markov solver
│       │   └── channel_geometry.py # Curved polylines & proximity meander bounds
│       ├── benchmarks/
│       │   ├── empirical_validation.py      # 28-delta DIDNT global benchmark engine
│       │   ├── reality_check_diagnostics.py # Physical reality diagnostics
│       │   └── mega_delta_benchmark.py      # Planetary mega-scale benchmark (Lena, GBMD, Amazon)
│       ├── data/
│       │   └── canonical_empirical_deltas.json # 28 DIDNT delta geomorphic features
│       └── io/
│           ├── datasets.py    # DIDNT dataset loader & empirical delta table
│           └── export.py      # GraphML, GeoJSON (LineStrings), and CSV export
├── tests/                     # 212 unit & integration tests across 20 modules
├── examples/                  # 01 through 06 runnable example scripts
├── docs/                      # Markdown documentation, architecture, & parameter justifications
├── CITATION.cff               # Academic citation metadata
├── CONTRIBUTING.md            # Contributor guidelines & invariant specifications
├── CHANGELOG.md               # Version history and release notes
├── SECURITY.md                # Vulnerability reporting protocol
├── CODE_OF_CONDUCT.md         # Contributor Covenant v2.1
└── pyproject.toml             # Packaging specification & optional extras
```

---

## 3. Tooling, Environment & Commands

Always use **`uv`** for dependency resolution, virtual environment execution, and testing:

### Virtual Environment & Testing
```bash
# Sync all dev, test, and optional dependencies
uv sync --extra dev

# Run pytest across the complete test suite
uv run --extra dev pytest tests -v

# Run a specific test module
uv run --extra dev pytest tests/test_generator.py -v

# Run with coverage
uv run --extra dev pytest tests --cov=dgm

# Lint with Ruff
uv run ruff check .
```

### Running Examples & Benchmarks
```bash
# Run single realization example
uv run python examples/01_single_realization.py

# Run custom YAML stochastic ensemble example
uv run python examples/02_custom_yaml_ensemble.py

# Run river vs tide comparison
uv run python examples/03_compare_river_vs_tide.py

# Run 28-delta global empirical validation benchmark
uv run python examples/04_empirical_validation_benchmark.py --enable-pruning --simplify-chains

# Run planetary mega-delta benchmark (Lena, GBMD, Amazon, Niger)
uv run python examples/05_mega_delta_scale_benchmark.py --realizations 2

# Run 2D sinuous spatial visualizer
uv run python examples/06_visualize_sinuous_trees_and_networks.py

# Run physical module reality check diagnostics
uv run python -m dgm.benchmarks.reality_check_diagnostics

# Use DGM CLI (single realization or ensemble sampling)
uv run dgm generate -N 45 -R 0.5 -s 42 -o delta.graphml
uv run dgm ensemble -N 50 -R 0.5 -n 30 --sample-parameters -o ensemble.csv
```

---

## 4. Configuration & Parameter Conventions

- **YAML-First & Standalone Packaging**: Model configurations are driven by YAML. At runtime, `DGMConfig.load_default()` searches `src/dgm/default_config.yaml` (packaged in the wheel), `src/dgm/config/default_config.yaml`, and repo root `config/default_config.yaml`.
- **Modular Dependency Footprint**:
  - Core: `networkx`, `numpy`, `scipy`, `pyyaml`.
  - Extras: `[viz]` (`matplotlib`), `[benchmarks]` (`pandas`, `scikit-learn`, `matplotlib`), `[all]`, `[dev]`.
  - IO utilities (`metrics_to_csv`, `summary_to_csv`) include a standard library `csv` fallback if `pandas` is not installed.
- **Primary Forcing Boundary Conditions**:
  - `N`: Node budget ($N \ge 4$).
  - `R_rt`: River-to-Tidal forcing ratio ($R_{\text{rt}} \in [0.0, 1.0]$).
    - $R_{\text{rt}} \approx 0.05$: River-dominated (dendritic tree, zero/few loops).
    - $R_{\text{rt}} \approx 0.50$: Mixed fluvial-tidal (moderate loopiness, anastomosing channels).
    - $R_{\text{rt}} \ge 0.80$: Tide-dominated (dense anastomosing loops, shoreline flaring).
  - `loop_mechanism`: `'potential_field'` (Mechanism A) or `'ner_optimality'` (Mechanism B).
- **Distributions**:
  - When modifying or adding distributions in YAML or `config.py`, support `type: uniform | normal | lognormal | choice | fixed`.
  - Always maintain deterministic fallback values in `get_active_parameters(sample=False)`.

---

## 5. Coding Standards & Invariants

1. **Planarity Invariant**:
   Never bypass the Jordan Curve intersection filter (`segments_intersect` / `edge_intersects_graph`). Delta channel networks must maintain planar geometry (no intersecting edges without a junction). Use 2D Axis-Aligned Bounding Box (AABB) fast rejection for spatial operations.
2. **Topological Invariants**:
   - The graph must be weakly connected (`nx.is_weakly_connected(G) == True`).
   - Inflow apex must be unique and identifiable (`node_type == 'apex'`).
   - Junction degrees must respect geomorphic limits: $k_{\text{in}} \le 2$ (confluences), $k_{\text{out}} \le 3$ (bifurcations/trifurcations).
3. **Shoreline Delivery & Pruning Invariant**:
   - When pruning is enabled (`enable_pruning=True`), 100% of terminal outlet nodes must deliver discharge to the shoreline ($r \ge r_{\text{shore}} (1 - \epsilon)$). Zero unphysical inland dead-ends.
   - When reach simplification is enabled (`simplify_pass_through_nodes=True`), degree-2 pass-through chains ($d_{\text{in}}=1, d_{\text{out}}=1$) are contracted into direct links between morphological junctions, guaranteeing consistent physical reach sinuosity ($S_{\text{reach}} = \frac{\sum L_i}{D_{\text{straight}}} \ge 1.0$) matching RivGraph extraction.
4. **Hydraulic Invariants**:
   - Murray's Law exponent $p > 0$.
   - Channel widths must be strictly positive ($w \ge \text{min\_channel\_width}$).
   - Markov flux throughput vector $\Phi$ must sum to unit input at apex ($\Phi_0 = 1.0$) and satisfy non-negativity ($\Phi \ge 0$).
5. **Preserve Documentation & Comments**:
   Preserve existing docstrings, mathematical formulas, and scientific references when modifying code.
6. **Type Annotations**:
   All public functions and methods must have standard Python type hints (`Optional`, `Dict`, `List`, `Tuple`, `Union`).

---

## 6. How to Extend DGM

- **Adding a New Physical Module**:
  Add module parameters under the `extensions:` section in `config/default_config.yaml`. The `DGMConfig` object automatically stores and parses the `extensions` dictionary.
- **Adding a New Loop Engine**:
  Implement the algorithm in `src/dgm/core/loops.py`, register the string identifier in `DGMConfig.validate()`, and dispatch it in `apply_loop_mechanism()`.
- **Adding New Metrics**:
  Add metric calculation methods to `src/dgm/metrics.py` within `MetricsBattery`. Ensure the outputs are floats/ints and included in `evaluate_all()`. Add corresponding unit tests in `tests/test_metrics.py`.
