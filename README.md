# DeltaGraphModel (DGM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DeltaGraphModel (DGM)** is a modular, process-emergent generative graph framework for synthesizing realistic river delta channel networks (DCNs) and sampling large stochastic ensembles.

Parameterized by minimal physical boundary conditions—node budget ($N$) and relative river-to-tidal forcing weight ($R_{\text{rt}} \in [0, 1]$)—DGM generates realistic, topologically complex, and hydraulically consistent distributary networks in milliseconds.

---

## Key Features

- **Topological Planarity Invariant**: Enforces Jordan Curve planarity filtering to eliminate unphysical channel crossings.
- **Dual Emergent Loop Engines**:
  - **Mechanism A (Potential Field Reconnection)**: Dynamic spatial probability field governing cross-ties and floodplain anastomosis.
  - **Mechanism B (Thermodynamic nER* Optimality)**: Simulated Annealing link pruning maximizing Non-Local Shannon Entropy Rate ($nER^*$).
- **Hydraulics & Murray's Law Conservation**: Conserves cross-sectional channel capacity at bifurcations ($w_0^p = w_1^p + w_2^p$), models estuarine mouth flaring, and computes open-channel conductance.
- **Transient Submatrix Markov Flux Solver**: Direct, well-conditioned linear flux throughput solver $(I - Q^T)\Phi_T = S_{0,T}$.
- **YAML-Driven Configuration & Extensibility**: All parameters, solvers, and stochastic distributions are declared in YAML.
- **20-Metric Geomorphic Battery**: Quantitative topological, dynamic, vulnerability, and hydraulic form metrics benchmarked against 28 empirical real-world deltas (DIDNT dataset).

---

## Installation & Environment Setup

We recommend using [**`uv`**](https://github.com/astral-sh/uv) for fast, reproducible dependency resolution and local virtual environments.

### 1. Clone the repository
```bash
git clone <repo-url> dgm
cd dgm
```

### 2. Install dependencies with `uv`
```bash
# Create virtual environment and sync dependencies
uv sync --extra dev
```

---

## Quickstart

### Python API

```python
import dgm

# 1. Generate a single delta graph realization
G = dgm.generate(N=50, R_rt=0.45, seed=42)

# 2. Evaluate the 20-metric geomorphic battery
battery = dgm.MetricsBattery()
metrics = battery.evaluate_all(G)

print(f"Generated delta with {G.number_of_nodes()} nodes and {G.number_of_edges()} channels.")
print(f"Alpha Index (Loopiness): {metrics['alpha_index']:.4f}")
print(f"Normalized Entropy Rate (nER*): {metrics['nER_star']:.4f}")
print(f"Entropic Braiding Index (eBI): {metrics['eBI']:.4f}")

# 3. Export to GraphML (for Gephi/Cytoscape) or GeoJSON (for QGIS)
dgm.to_graphml(G, "outputs/my_delta.graphml")
dgm.to_geojson(G, "outputs/my_delta.geojson")
```

### Sampling Stochastic Ensemble Clouds

```python
import dgm

# Configure ensemble with stochastic parameter sampling enabled
cfg = dgm.DGMConfig.load_default()
cfg.N = 40
cfg.R_rt = 0.60
cfg.sample_parameters = True

# Sample 100 graph realizations
ensemble = dgm.EnsembleGenerator(
    N=cfg.N,
    R_rt=cfg.R_rt,
    num_realizations=100,
    config=cfg,
    base_seed=100
)

summary = ensemble.evaluate_ensemble()
print("Alpha mean:", summary['metrics']['alpha_index']['mean'])

# Export aggregated ensemble summary statistics (mean, std, percentiles)
dgm.summary_to_csv(summary, "outputs/ensemble_summary.csv")

# Or export raw realization metric cloud (one row per realization)
dgm.metrics_to_csv(summary, "outputs/ensemble_realizations.csv")
```

---

## Command Line Interface (CLI)

DGM includes a built-in CLI:

```bash
# Generate a single graph realization
uv run dgm generate -N 50 -R 0.5 -s 42 -o delta_50.graphml

# Sample an ensemble cloud and save metrics to CSV
uv run dgm ensemble -N 50 -R 0.5 -n 50 --sample-parameters -o ensemble_50.csv
```

---

## Configuration via YAML

DGM is configured via `config/default_config.yaml`. Users can supply custom YAML files or override individual parameters at runtime:

```python
import dgm

# Load custom config with runtime overrides
cfg = dgm.DGMConfig.from_yaml("my_config.yaml", N=70, R_rt=0.8)
G = dgm.generate(config=cfg)
```

See [docs/configuration.md](docs/configuration.md) for the complete schema and parameter bounds reference.

---

## Empirical Validation & Benchmarks

DGM includes rigorous benchmarking suites validating synthetic networks against real-world observations:

- **28-Delta Global Empirical Validation**: Exercises DGM across the complete DIDNT dataset ([Hariharan et al., 2021](https://doi.org/10.1029/2021WR030026)), verifying that 89.3% of empirical deltas fall within the 95% PCA confidence ellipse.
  ```bash
  uv run python examples/04_empirical_validation_benchmark.py --enable-pruning --simplify-chains
  ```
  Full Report: [outputs/VALIDATION_REPORT.md](outputs/VALIDATION_REPORT.md)

- **Planetary Mega-Delta Scale Benchmark**: Exercises DGM at full empirical scale ($N \ge 600 - 3,000$ nodes) across Earth's four largest distributary systems: Lena ($N=2,914$), GBMD ($N=2,488$), Amazon ($N=1,094$), and Niger ($N=651$).
  ```bash
  uv run python examples/05_mega_delta_scale_benchmark.py --realizations 2
  ```
  Full Report: [outputs/MEGA_DELTA_BENCHMARK_REPORT.md](outputs/MEGA_DELTA_BENCHMARK_REPORT.md)

---

## Running Tests

Run the full pytest suite with `uv`:

```bash
uv run --extra dev pytest tests -v
```

---

## Documentation & Interactive AI Assistant Portal

DGM includes a client-side **Interactive Documentation Portal & AI Assistant** (`site/`) styled with a high-precision, dark tactical HUD aesthetic inspired by [Kestrel Intelligence](https://kestrelintelligence.com/).

### Key Features
- **In-Browser Hybrid RAG**: Combines BM25 lexical search with precomputed 384-d semantic vector embeddings (`all-MiniLM-L6-v2`, float16) evaluated entirely client-side.
- **Local Neural Synthesis**: Runs open-weight coding models (e.g., `Qwen2.5-Coder-0.5B`, `SmolLM2-360M`) directly in your browser via [Transformers.js](https://huggingface.co/docs/transformers.js) with WebGPU acceleration and WASM CPU fallback (100% private, zero server API keys needed).
- **Tactical Tri-Mode HUD**: Adaptive layout supporting Dual View (`⌘1`), Full-Width Chat with 38px Collapsible Telemetry Spine (`⌘2` or `⌘B`), and Full-Width Retrieval (`⌘3`).
- **Complete HTML Guides**: Full narrative documentation rendered into styled HTML pages with two-way deep links to the AI assistant.

### Quick Commands via `Makefile`

```bash
# Preview the portal locally at http://localhost:8000
make site-serve

# Rebuild complete site (HTML docs + search index + vector embeddings)
make site-build

# Regenerate only the BM25 search index and 384-d vector embeddings
make site-index

# Run complete pytest suite
make test
```

### Automated GitHub Pages Deployment (CI/CD)

The documentation portal deploys automatically to GitHub Pages via `.github/workflows/deploy-pages.yml`:
- **Automated Embedding & Docs Generation**: On every push to `main`, GitHub Actions automatically installs `uv`, executes `scripts/build_docs_site.py` to recompute vector embeddings and build HTML docs in ~15 seconds, and publishes the static artifact.
- **Zero Drift**: Documentation updates and codebase docstrings are immediately reflected in the live RAG search index with zero manual intervention.

---

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory and the web portal:
- [Architecture & Mathematical Model](docs/architecture.md)
- [Configuration & YAML Schema](docs/configuration.md)
- [Python API Reference](docs/api_reference.md)
- [Tutorial 1: Quickstart](docs/tutorials/01_quickstart.md)
- [Tutorial 2: Ensemble Sampling](docs/tutorials/02_ensemble_clouds.md)
- [End-to-End Walkthrough](docs/walkthrough.md)
- [Developer & Agent Guide](AGENTS.md)
- [Global Empirical Validation Report](outputs/VALIDATION_REPORT.md)
- [Planetary Mega-Delta Benchmark Report](outputs/MEGA_DELTA_BENCHMARK_REPORT.md)


---

## License

MIT License. See [LICENSE](LICENSE) for details.
