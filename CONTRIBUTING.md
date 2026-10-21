# Contributing to DeltaGraphModel (DGM)

Thank you for your interest in contributing to **DeltaGraphModel (DGM)**! We welcome contributions from geomorphologists, hydrologists, network scientists, software engineers, and enthusiastic open-source developers.

Please review this guide to help make the contribution process smooth and effective.

---

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior following the guidelines in that document.

---

## Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- [**`uv`**](https://github.com/astral-sh/uv) (strongly recommended for fast, reproducible dependency resolution)
- Git

### 2. Fork and Clone
```bash
git clone https://github.com/<your-username>/dgm.git
cd dgm
```

### 3. Set Up Development Environment
With `uv`:
```bash
# Create virtual environment and install all dev and optional dependencies
uv sync --extra dev
```

Alternatively, with standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Development Workflow

### 1. Create a Topic Branch
Always branch from `main`:
```bash
git checkout -b feature/my-new-feature
# or: git checkout -b fix/issue-description
```

### 2. Physical & Topological Invariants
DGM models natural physical networks. When contributing algorithmic changes to `src/dgm/core/`, please uphold these core scientific invariants:

1. **Topological Planarity Invariant**: Channels must not cross without an explicit junction. Never bypass the Jordan Curve intersection test (`segments_intersect` / `edge_intersects_graph`). Use 2D Axis-Aligned Bounding Box (AABB) fast rejection for spatial queries.
2. **Weak Connectivity**: Generated networks must be weakly connected DAGs (`nx.is_weakly_connected(G) == True`).
3. **Unique Feeder Apex**: Each delta must have a single identifiable apex node (`node_type == 'apex'`) where total discharge $\Phi_0 = 1.0$ enters.
4. **Junction Degree Bounds**: Junctions must respect natural geomorphic limits ($k_{\text{in}} \le 2$ confluences, $k_{\text{out}} \le 3$ bifurcations/trifurcations).
5. **Shoreline Delivery**: When dead-end pruning is enabled (`enable_pruning=True`), 100% of terminal outlet nodes must deliver discharge to the shoreline envelope ($r \ge r_{\text{shore}}(1 - \epsilon_{\text{shore}})$).
6. **Hydraulic Conservation**: Bifurcation width allocation must satisfy Murray's Law ($w_0^p = w_1^p + w_2^p$) with strictly positive channel widths ($w \ge w_{\text{min}}$) and non-negative Markov flux throughput ($\Phi \ge 0$).

### 3. Code Style & Linting
DGM enforces modern Python standards using [Ruff](https://astral.sh/ruff):
```bash
# Run linter
uv run ruff check .

# Automatically apply safe formatting fixes
uv run ruff format .
uv run ruff check --fix .
```

### 4. Running the Test Suite
Ensure that all unit and integration tests pass before opening a Pull Request:
```bash
# Run pytest across the complete test suite
uv run --extra dev pytest tests -v

# Run with test coverage report
uv run --extra dev pytest tests --cov=dgm
```

### 5. Running Examples & Benchmarks
Test that existing examples execute cleanly:
```bash
uv run python examples/01_single_realization.py
uv run python examples/02_custom_yaml_ensemble.py
uv run python examples/03_compare_river_vs_tide.py
uv run python examples/06_visualize_sinuous_trees_and_networks.py
```

---

## Submitting Pull Requests

1. **Commit Messages**: Write clear, descriptive commit messages adhering to Conventional Commits:
   - `feat(...)`: New feature or physical module
   - `fix(...)`: Bug fix or numerical correction
   - `docs(...)`: Documentation updates
   - `test(...)`: Adding or modifying tests
   - `refactor(...)`: Code cleanup without behavior change
2. **Documentation**: If adding new parameters or algorithms, update:
   - `config/default_config.yaml` and `src/dgm/default_config.yaml`
   - `docs/parameter_justifications.md` (cite peer-reviewed literature for parameter defaults and bounds)
   - `docs/configuration.md`
3. **Tests**: Include unit tests covering new edge cases, boundary conditions, or configuration parameters.
4. **Open a PR**: Open a Pull Request against `main`. Fill in the PR template explaining the motivation, changes made, and verification steps.

---

## Reporting Bugs and Feature Requests

- **Bug Reports**: Open an issue on GitHub describing the bug, minimal reproducible example code, version of DGM and Python, and expected vs actual behavior.
- **Feature Requests**: Open an issue or discussion proposing the physical mechanism or capability, referencing relevant geomorphic or hydraulic literature where appropriate.
