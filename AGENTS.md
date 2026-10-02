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
├── config/
│   └── default_config.yaml    # Master YAML configuration file
├── src/
│   └── dgm/
│       ├── __init__.py        # Clean public exports: generate(), sample_ensemble(), DGMConfig
│       ├── config.py          # DGMConfig dataclass & parameter distribution engine
│       ├── generator.py       # Facades: DeltaGenerator & EnsembleGenerator
│       ├── metrics.py         # 20-metric geomorphic battery (MetricsBattery)
│       ├── cli.py             # Command line interface (dgm generate, dgm ensemble)
│       ├── core/
│       │   ├── domain.py      # PolarDomain (r, theta, adaptive opening angle)
│       │   ├── geometry_utils.py # Jordan Curve Theorem planarity filter
│       │   ├── backbone.py    # Polar growth tree skeleton
│       │   ├── loops.py       # Emergent loop engines (potential_field & ner_optimality)
│       │   └── hydraulics.py  # Murray's law, flaring, conductance, Markov flux solver
│       └── io/
│           ├── datasets.py    # DIDNT dataset loader & empirical delta table
│           └── export.py      # GraphML, GeoJSON, and CSV export
├── tests/                     # Comprehensive pytest suite
├── examples/                  # Runnable example scripts
├── docs/                      # Markdown documentation & tutorials
└── pyproject.toml             # Packaging specification
```

---

## 3. Tooling, Environment & Commands

Always use **`uv`** for dependency resolution, virtual environment execution, and testing:

### Virtual Environment & Testing
```bash
# Run pytest with dev dependencies
uv run --extra dev pytest tests -v

# Run a specific test module
uv run --extra dev pytest tests/test_generator.py -v

# Run with coverage
uv run --extra dev pytest tests --cov=dgm
```

### Running Examples & Scripts
```bash
# Run single realization example
uv run python examples/01_single_realization.py

# Run river vs tide comparison
uv run python examples/03_compare_river_vs_tide.py

# Use DGM CLI
uv run dgm generate -N 45 -R 0.5 -s 42 -o delta.graphml
```

---

## 4. Configuration & Parameter Conventions

- **YAML-First**: Model configurations are driven by `config/default_config.yaml`.
- **Primary Forcing Boundary Conditions**:
  - `N`: Node budget ($N \ge 4$).
  - `R_rt`: River-to-Tidal forcing ratio ($R_{\text{rt}} \in [0.0, 1.0]$).
    - $R_{\text{rt}} \approx 0.05$: River-dominated (dendritic tree, zero/few loops).
    - $R_{\text{rt}} \approx 0.50$: Mixed fluvial-tidal (moderate loopiness, braided channels).
    - $R_{\text{rt}} \ge 0.80$: Tide-dominated (dense anastomosing loops, shoreline flaring).
  - `loop_mechanism`: `'potential_field'` (Mechanism A) or `'ner_optimality'` (Mechanism B).
- **Distributions**:
  - When modifying or adding distributions in YAML or `config.py`, support `type: uniform | normal | lognormal | choice | fixed`.
  - Always maintain deterministic fallback values in `get_active_parameters(sample=False)`.

---

## 5. Coding Standards & Invariants

1. **Planarity Invariant**:
   Never bypass the Jordan Curve intersection filter (`segments_intersect` / `edge_intersects_graph`). Delta channel networks must maintain planar geometry (no intersecting edges without a junction).
2. **Topological Invariants**:
   - The graph must be weakly connected (`nx.is_weakly_connected(G) == True`).
   - Inflow apex must be unique and identifiable (`node_type == 'apex'`).
   - Junction degrees must respect geomorphic limits: $k_{\text{in}} \le 2$ (confluences), $k_{\text{out}} \le 3$ (bifurcations/trifurcations).
3. **Hydraulic Invariants**:
   - Murray's Law exponent $p > 0$.
   - Channel widths must be strictly positive ($w \ge \text{min\_channel\_width}$).
   - Markov flux throughput vector $\Phi$ must sum to unit input at apex ($\Phi_0 = 1.0$) and satisfy non-negativity ($\Phi \ge 0$).
4. **Preserve Documentation & Comments**:
   Preserve existing docstrings, mathematical formulas, and scientific references when modifying code.
5. **Type Annotations**:
   All public functions and methods must have standard Python type hints (`Optional`, `Dict`, `List`, `Tuple`, `Union`).

---

## 6. How to Extend DGM

- **Adding a New Physical Module**:
  Add module parameters under the `extensions:` section in `config/default_config.yaml`. The `DGMConfig` object automatically stores and parses the `extensions` dictionary.
- **Adding a New Loop Engine**:
  Implement the algorithm in `src/dgm/core/loops.py`, register the string identifier in `DGMConfig.validate()`, and dispatch it in `apply_loop_mechanism()`.
- **Adding New Metrics**:
  Add metric calculation methods to `src/dgm/metrics.py` within `MetricsBattery`. Ensure the outputs are floats/ints and included in `evaluate_all()`. Add corresponding unit tests in `tests/test_metrics.py`.
