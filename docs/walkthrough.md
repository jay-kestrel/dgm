# End-to-End Walkthrough: From Installation to Plotting

This walkthrough provides a comprehensive, step-by-step guide to using **DeltaGraphModel (DGM)**. You will learn how to:
1. [Install and configure your environment](#1-installation--environment-setup)
2. [Configure the model and customize physical parameters](#2-model-configuration--parameter-tuning)
3. [Run single realizations, multi-stage pipelines, and stochastic ensembles](#3-running-the-model)
4. [Generate geomorphically accurate 2D cartographic plots and figures](#4-plotting--visualizing-results)
5. [Run a complete, self-contained end-to-end Python script](#5-complete-runnable-script)

---

## 1. Installation & Environment Setup

DGM requires **Python 3.10+**. We recommend using [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management, but standard `pip` and virtual environments are fully supported.

### Option A: Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/delta-graphs/dgm.git
cd dgm

# Create a virtual environment and synchronize all dependencies (including dev tools)
uv sync --extra dev
```

### Option B: Using Standard Python `venv` and `pip`

```bash
# Clone the repository
git clone https://github.com/delta-graphs/dgm.git
cd dgm

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Core Dependencies

DGM relies on standard scientific Python libraries:
- **`networkx` (>= 3.0)**: Graph data structures, topological metrics, and traversals.
- **`numpy` (>= 1.24)**: Vectorized numerical computing and random number generation (`Generator` API).
- **`scipy` (>= 1.10)**: KD-Tree spatial spatial queries (`cKDTree`), sparse solvers, and statistical tests.
- **`matplotlib` (>= 3.7)**: Cartographic rendering, color mapping, and publication-ready figures.
- **`pyyaml` (>= 6.0)**: YAML configuration parsing and serialization.
- **`pandas` (>= 2.0)** & **`scikit-learn` (>= 1.2)**: Tabular metric export, distance matrices, and Mahalanobis validation.

### Verifying Installation

Verify that DGM and its CLI are installed correctly:

```bash
# Verify Python import and package version
python -c "import dgm; print(f'DGM version: {dgm.__version__}')"

# Verify CLI availability
dgm --help
```

---

## 2. Model Configuration & Parameter Tuning

DGM parameterizes river delta channel networks using physically meaningful boundary conditions and process parameters, organized into logical sections within [`DGMConfig`](api_reference.md#dgmconfig).

### Parameter Sections Overview

1. **`model`**: High-level simulation control
   - `N` (*int*, default `50`): Target node budget (junctions + outlets).
   - `R_rt` (*float*, default `0.5`): River-to-tidal marine forcing ratio in $[0, 1]$ ($0 = \text{pure fluvial}$, $1 = \text{pure tidal}$).
   - `loop_mechanism` (*str*, default `"potential_field"`): Reconnection engine (`"potential_field"` for spatial potential fields, `"ner_optimality"` for thermodynamic annealing).
   - `sample_parameters` (*bool*, default `false`): When `true`, stochastically draws parameters from specified distributions.
   - `seed` (*int*, default `42`): Base random seed.

2. **`domain`**: Polar basin geometry & lateral confinement
   - `r_shore` (*float*, default `10000.0` m): Shoreline radius.
   - `theta_base` (*float*, default `45.0`°): Fluvial opening angle.
   - `theta_delta` (*float*, default `60.0`°): Tidal fan expansion angle.
   - `confinement` (*float*, default `0.0`): Valley wall constriction factor in $[0.0, 0.95]$.
   - `lateral_valley_width` (*float*, optional): Fixed maximum corridor width in meters ($|x(r)| \le W/2$).

3. **`backbone`**: Branching tree growth
   - `branching_weights` (*tuple*, default `(0.50, 0.45, 0.05)`): Probabilities of $1$, $2$, or $3$ child branches.
   - `coastal_node_decay` (*float*, default `0.35`): Shoreward contraction rate of radial inter-junction spacing.
   - `coastal_sinuosity_decay` (*float*, default `0.30`): Shoreward straightening of distributary branches.

4. **`loops`**: Emergent loop formation & cross-ties
   - `reconnection_prob_base` ($c_0$, default `0.55`): Tidal/floodplain reconnection scale.
   - `anastomosis_prob_base` ($P_0$, default `0.0`): Baseline inland floodplain anastomosis probability.
   - `candidate_dist_scale` ($l_0 / r_{\text{shore}}$, default `0.25`): Spatial neighborhood search distance.
   - `max_in_degree` (default `2`), `max_out_degree` (default `3`): Geomorphic junction degree constraints.
   - `target_max_loops_slope` (default `0.22`), `target_max_loops_intercept` (default `0.05`): Scaling for maximum allowable loop closures.

5. **`pruning`**: Pre-hydraulic network simplification & spur removal
   - `enable_pruning` (*bool*, default `false`): Toggles pre-hydraulic simplification.
   - `prune_inland_dead_ends` (*bool*, default `true`): Prunes blind channels that do not reach the coast.
   - `outlet_tolerance` (*float*, default `0.05`): Coastal buffer envelope $(1 - \text{tol}) \cdot r_{\text{shore}}$ protecting near-shore outlets.
   - `min_spur_length` (*float*, default `0.0` m): Length threshold below which dead-end spurs are pruned.
   - `simplify_pass_through_nodes` (*bool*, default `false`): Contracts degree-2 chains ($d_{\text{in}}=1, d_{\text{out}}=1$) into single morphological reaches (RivGraph convention).
   - `recalculate_reach_sinuosity` (*bool*, default `true`): Recalculates unified reach sinuosity from the process model, preventing double-counting of geometric knees.

6. **`hydraulics`**: Murray's Law width conservation & discharge
   - `murray_p` (*float*, default `1.70`): Murray's Law exponent ($w_{\text{parent}}^p = w_1^p + w_2^p$).
   - `width_beta` (*float*, default `0.745`): Bifurcation width asymmetry factor.
   - `apex_width` (*float*, default `250.0` m): Feeder trunk channel width at the apex.
   - `tidal_flare_k` (*float*, default `1.40`): Seaward exponential width flaring rate.
   - `tidal_flare_onset` (*float*, default `0.30`): Normalized radius ($r / r_{\text{shore}}$) where tidal widening begins.
   - `loop_flow_penalty` (*float*, default `0.40`): Conductance penalty factor applied to secondary loop channels.
   - `manning_eta` (*float*, default `1.65`): Width exponent in Manning conductance ($C_e = w^\eta / L^\gamma$).
   - `length_gamma` (*float*, default `0.50`): Length friction damping exponent in conductance.

7. **`channel_geometry`**: Sinuous centerlines & proximity constraints
   - `enable_curved_centerlines` (*bool*, default `false`): Generates harmonic polyline coordinates with Dirichlet boundary conditions.
   - `min_channel_clearance` (*float*, default `0.40`): Safety clearance fraction preventing channel collisions.
   - `num_polyline_points` (*int*, default `21`): Discretization vertex count per reach polyline.
   - `meander_bends` (*int*, default `1`): Number of harmonic meander half-wavelengths per reach.

---

### 4 Ways to Configure DGM

#### Method 1: Loading Default Configuration

```python
import dgm

# Load standard default configuration dataclass
cfg = dgm.DGMConfig.load_default()
cfg.N = 60
cfg.R_rt = 0.40
cfg.enable_curved_centerlines = True
cfg.enable_pruning = True
```

#### Method 2: Loading from a YAML File

Create a YAML configuration file (`my_delta.yaml`):

```yaml
model:
  N: 55
  R_rt: 0.65
  seed: 123

domain:
  r_shore: 12000.0
  theta_base: 50.0

pruning:
  enable_pruning: true
  outlet_tolerance: 0.04
  simplify_pass_through_nodes: true

hydraulics:
  tidal_flare_onset: 0.25
  loop_flow_penalty: 0.35

channel_geometry:
  enable_curved_centerlines: true
  min_channel_clearance: 0.45
```

Load it directly in Python:

```python
cfg = dgm.DGMConfig.from_yaml("my_delta.yaml")
```

#### Method 3: Dynamic Parameter Tuning & Inline Distributions

You can adjust parameters at runtime or define stochastic distributions using `cfg.set_parameter()`:

```python
cfg = dgm.DGMConfig.load_default()

# 1. Set fixed value
cfg.set_parameter("tidal_flare_onset", 0.35)

# 2. Set uniform distribution [low, high]
cfg.set_parameter("loop_flow_penalty", (0.25, 0.55))

# 3. Set normal distribution with mean and std
cfg.set_parameter("apex_width", {"type": "normal", "mean": 300.0, "std": 30.0})
```

#### Method 4: Keyword Overrides in `dgm.generate()`

All configuration parameters can be passed directly to `dgm.generate()`:

```python
G = dgm.generate(
    N=45,
    R_rt=0.35,
    enable_curved_centerlines=True,
    min_channel_clearance=0.40,
    enable_pruning=True,
    tidal_flare_onset=0.28,
    seed=42
)
```

---

## 3. Running the Model

### Generating a Single Delta Network

```python
import dgm

# Generate a synthetic delta network
G = dgm.generate(
    N=50,
    R_rt=0.50,
    enable_curved_centerlines=True,
    enable_pruning=True,
    seed=101
)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

### Inspecting Node and Edge Attributes

Each generated graph is a standard `networkx.DiGraph` enriched with physical, geometric, and hydraulic attributes:

```python
# Inspect apex and outlet nodes
for node_id, data in G.nodes(data=True):
    node_type = data.get("node_type", "internal")
    x, y = data["x"], data["y"]
    r, theta = data["r"], data["theta"]
    if node_type in ("apex", "outlet"):
        print(f"Node {node_id:2d} ({node_type:6s}): x={x:7.1f} m, y={y:7.1f} m, r={r:7.1f} m")

# Inspect channel reach properties
for u, v, data in list(G.edges(data=True))[:5]:
    w = data.get("width", 0.0)
    L = data.get("length", 0.0)
    sin = data.get("sinuosity", 1.0)
    Q = data.get("flux", 0.0)
    coords = data.get("coordinates", [])
    print(f"Edge ({u} -> {v}): Width={w:5.1f} m, Length={L:6.1f} m, Sinuosity={sin:.3f}, Flux={Q:.4f}, Vertices={len(coords)}")
```

### Extracting Multi-Stage Generative Snapshots

To analyze how the delta emerges across the 6-stage generative assembly pipeline, set `return_stages=True`:

```python
G_final, stages = dgm.generate(
    N=50,
    R_rt=0.50,
    enable_curved_centerlines=True,
    enable_pruning=True,
    return_stages=True,
    seed=42
)

# stages is a dictionary containing graph snapshots:
print("Available generative stages:")
for stage_name, stage_graph in stages.items():
    print(f" - {stage_name:7s}: {stage_graph.number_of_nodes():2d} nodes, {stage_graph.number_of_edges():2d} edges")

# stage keys: 'tree', 'looped', 'pruned', 'final'
G_tree = stages["tree"]
G_looped = stages["looped"]
G_pruned = stages["pruned"]
```

### Evaluating Geomorphic Metrics (20-Metric Battery)

Evaluate the comprehensive battery of 20 quantitative geomorphic metrics benchmarked against the global DIDNT dataset:

```python
battery = dgm.MetricsBattery()
metrics = battery.evaluate_all(G_final)

print("Key Geomorphic Metrics:")
print(f"  Alpha Loopiness Index:           {metrics['alpha_index']:.4f}")
print(f"  Beta-1 Loops per Node:           {metrics['beta1_per_node']:.4f}")
print(f"  Normalized Entropy Rate (nER*):  {metrics['nER_star']:.4f}")
print(f"  Entropic Braiding Index (eBI):   {metrics['eBI']:.4f}")
print(f"  Global Vulnerability (V_net):    {metrics['global_vulnerability']:.4f}")
print(f"  Mean Channel Sinuosity:          {metrics['mean_sinuosity']:.3f}")
print(f"  Shoreline Delivery Ratio:        {metrics['shoreline_delivery_ratio']:.2%}")
```

### Sampling Stochastic Ensemble Clouds

Sample an ensemble of $M$ realizations across parameter distributions to quantify natural delta variability:

```python
cfg = dgm.DGMConfig.load_default()
cfg.N = 45
cfg.R_rt = 0.55
cfg.sample_parameters = True

# Instantiate ensemble generator
ensemble = dgm.EnsembleGenerator(
    N=cfg.N,
    R_rt=cfg.R_rt,
    num_realizations=50,
    config=cfg,
    base_seed=200
)

# Compute statistical distribution of metrics across the cloud
summary = ensemble.evaluate_ensemble()

for m in ["alpha_index", "nER_star", "eBI"]:
    mean = summary["metrics"][m]["mean"]
    std = summary["metrics"][m]["std"]
    print(f"{m:22s}: {mean:.4f} ± {std:.4f}")
```

### Exporting Realizations & Metrics

```python
# Export network topology and geometry for GIS & visualization
dgm.to_graphml(G_final, "outputs/my_delta.graphml")  # For Gephi / Cytoscape
dgm.to_geojson(G_final, "outputs/my_delta.geojson")  # For QGIS / ArcGIS / Mapbox

# Export ensemble summary to CSV
dgm.metrics_to_csv(summary, "outputs/ensemble_metrics.csv")
```

---

## 4. Plotting & Visualizing Results

DGM provides a high-level spatial visualization suite in `dgm.visualization` (accessible directly via `dgm.plot_network`, `dgm.plot_tree_vs_network`, and `dgm.plot_stages`).

### 1. Rendering a Single Delta Network (`dgm.plot_network`)

Render an publication-grade cartographic map displaying sinuous channel polylines, Murray's Law width hierarchies, geomorphic node classifications, and polar shoreline boundaries:

```python
import matplotlib.pyplot as plt
import dgm

G = dgm.generate(
    N=45,
    R_rt=0.35,
    enable_curved_centerlines=True,
    min_channel_clearance=0.40,
    enable_pruning=True,
    seed=101
)

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
dgm.plot_network(
    G,
    ax=ax,
    curved=True,                 # Render multi-vertex sinuous polylines
    edge_width="width",          # Scale channel thickness by physical width
    width_scaling="sqrt",        # Proportional to hydraulic cross-section
    min_linewidth=1.0,           # Minimum line thickness in points
    max_linewidth=6.5,           # Maximum line thickness in points
    show_nodes=True,             # Classify nodes (apex, outlets, bifurcations, confluences)
    show_domain=True,            # Draw shoreline arc and domain boundaries
    orientation="cartesian",     # "cartesian" (flow to right), "downward", or "upward"
    theme="light",               # "light", "dark", or "clean"
    title="Synthetic River Delta (R_rt = 0.35)",
    show_stats=True,             # Display summary statistics badge
    show_legend=True             # Display geomorphic taxonomy legend
)

fig.savefig("outputs/delta_map.png", bbox_inches="tight")
plt.close(fig)
```

#### Supported Themes and Orientations

- **Themes**:
  - `"light"`: Clean publication white background, classic river blue channels, dark text.
  - `"dark"`: Slate-blue dark mode background with luminous cyan/blue channels.
  - `"clean"`: Transparent background with minimal ornamentation.
- **Orientations**:
  - `"cartesian"`: Apex at $(0, 0)$, flow moves rightward towards $+x$.
  - `"downward"`: Apex at top $(0, 0)$, flow moves downward towards $-y$ (traditional map view).
  - `"upward"`: Apex at bottom $(0, 0)$, flow moves upward towards $+y$.

---

### 2. Side-by-Side Tree vs. Network Comparison (`dgm.plot_tree_vs_network`)

Compare the initial dendritic polar tree backbone against the final network with emergent loops, Murray widths, and sinuous meanders:

```python
G_final, stages = dgm.generate(
    N=50,
    R_rt=0.50,
    enable_curved_centerlines=True,
    enable_pruning=True,
    return_stages=True,
    seed=42
)

fig, (ax_tree, ax_net) = dgm.plot_tree_vs_network(
    G_tree=stages["tree"],
    G_final=G_final,
    curved=True,
    orientation="cartesian",
    theme="light",
    figsize=(18, 8),
    dpi=300,
    title="DGM Generative Transition: Tree Backbone vs. Final Hydraulic Network",
    show_stats=True,
    show_legend=True
)

fig.savefig("outputs/tree_vs_network.png", bbox_inches="tight")
plt.close(fig)
```

---

### 3. Progressive 4-Stage Assembly Evolution (`dgm.plot_stages`)

Visualize how the channel network forms across each stage of the generative pipeline:

```python
_, stages = dgm.generate(
    N=40,
    R_rt=0.55,
    enable_curved_centerlines=True,
    enable_pruning=True,
    return_stages=True,
    seed=77
)

fig, axes = dgm.plot_stages(
    stages=stages,
    curved=True,
    orientation="cartesian",
    theme="light",
    figsize=(24, 6.5),
    dpi=300,
    title="DGM Multi-Stage Assembly: Tree -> Loops -> Pruning -> Hydraulics & Meanders"
)

fig.savefig("outputs/progressive_stages.png", bbox_inches="tight")
plt.close(fig)
```

---

### 4. Hydrodynamic Forcing Spectrum Comparison

Demonstrate the morphological transition from river-dominated deltas (dendritic, low loopiness) to tide-dominated deltas (highly looped, flared channels):

```python
regimes = [
    ("Fluvial-Dominated (R_rt = 0.05)", 0.05, 101),
    ("Mixed Fluvial-Tidal (R_rt = 0.50)", 0.50, 102),
    ("Tide-Dominated (R_rt = 0.90)", 0.90, 103),
]

fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=300)

for idx, (label, r_val, s_val) in enumerate(regimes):
    g = dgm.generate(
        N=45,
        R_rt=r_val,
        enable_curved_centerlines=True,
        enable_pruning=True,
        seed=s_val
    )
    dgm.plot_network(
        g,
        ax=axes[idx],
        curved=True,
        theme="light",
        title=label,
        show_stats=True,
        show_legend=(idx == 0)
    )

fig.suptitle("DGM Hydrodynamic Forcing Spectrum (Fluvial to Tidal)", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig("outputs/forcing_spectrum.png", bbox_inches="tight")
plt.close(fig)
```

---

### 5. CLI Plotting Shortcut

You can generate and render delta plots directly from the command line by specifying an image output format (`.png`, `.pdf`, `.svg`):

```bash
# Generate and save a delta network map directly to PNG
dgm generate -N 55 -R 0.45 --curved --enable-pruning -o outputs/cli_delta.png
```

---

## 5. Complete Runnable Script

Save the script below as `run_walkthrough.py` and execute it with `python run_walkthrough.py` (or `uv run run_walkthrough.py`) to run the entire pipeline end-to-end:

```python
"""
End-to-End Walkthrough Script for DeltaGraphModel (DGM).
Demonstrates:
  1. Custom configuration with new parameters
  2. Single realization generation with stage tracking
  3. Metric evaluation with the 20-metric geomorphic battery
  4. Spatial plotting (single realization, stage comparison, and forcing spectrum)
  5. GraphML and GeoJSON export
"""

from pathlib import Path
import matplotlib.pyplot as plt
import dgm


def main():
    print("=" * 70)
    print("DeltaGraphModel (DGM): End-to-End Walkthrough Demonstration")
    print("=" * 70)

    # 1. Output directory setup
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Configuration setup
    print("\n[Step 1] Configuring model parameters...")
    cfg = dgm.DGMConfig.load_default()
    cfg.N = 48
    cfg.R_rt = 0.45
    cfg.enable_curved_centerlines = True
    cfg.min_channel_clearance = 0.40
    cfg.enable_pruning = True
    cfg.outlet_tolerance = 0.05
    cfg.tidal_flare_onset = 0.30
    cfg.loop_flow_penalty = 0.40
    cfg.seed = 42

    # 3. Model execution
    print("\n[Step 2] Synthesizing delta network with stage extraction...")
    G_final, stages = dgm.generate(
        config=cfg,
        return_stages=True
    )
    print(f"  -> Generated final network: {G_final.number_of_nodes()} nodes, {G_final.number_of_edges()} channels.")

    # 4. Metric evaluation
    print("\n[Step 3] Evaluating 20-metric geomorphic battery...")
    battery = dgm.MetricsBattery()
    metrics = battery.evaluate_all(G_final)
    print(f"  -> Alpha Loopiness:        {metrics['alpha_index']:.4f}")
    print(f"  -> Normalized Entropy Rate: {metrics['nER_star']:.4f}")
    print(f"  -> Entropic Braiding Index: {metrics['eBI']:.4f}")
    print(f"  -> Global Vulnerability:   {metrics['global_vulnerability']:.4f}")
    print(f"  -> Mean Sinuosity:         {metrics['mean_sinuosity']:.3f}")

    # 5. Spatial Visualization
    print("\n[Step 4] Plotting results...")

    # Plot 1: Detailed single network map
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    dgm.plot_network(
        G_final,
        ax=ax,
        curved=True,
        edge_width="width",
        width_scaling="sqrt",
        theme="light",
        title="DGM Synthetic River Delta (N=48, R_rt=0.45)",
        show_stats=True,
        show_legend=True
    )
    p1 = out_dir / "walkthrough_single_delta.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved single network plot to: {p1}")

    # Plot 2: Tree Backbone vs Final Network comparison
    fig_comp, _ = dgm.plot_tree_vs_network(
        G_tree=stages["tree"],
        G_final=G_final,
        curved=True,
        theme="light",
        figsize=(18, 8),
        dpi=300,
        title="DGM Transition: Dendritic Skeleton to Emergent Looped Network"
    )
    p2 = out_dir / "walkthrough_tree_vs_network.png"
    fig_comp.savefig(p2, bbox_inches="tight")
    plt.close(fig_comp)
    print(f"  -> Saved comparative tree vs network plot to: {p2}")

    # Plot 3: 4-Stage Progressive Evolution
    fig_stages, _ = dgm.plot_stages(
        stages=stages,
        curved=True,
        theme="light",
        figsize=(22, 6),
        dpi=300,
        title="DGM Multi-Stage Generative Evolution"
    )
    p3 = out_dir / "walkthrough_stages_evolution.png"
    fig_stages.savefig(p3, bbox_inches="tight")
    plt.close(fig_stages)
    print(f"  -> Saved progressive stages plot to: {p3}")

    # 6. Data Export
    print("\n[Step 5] Exporting graph data...")
    p_graphml = out_dir / "walkthrough_delta.graphml"
    p_geojson = out_dir / "walkthrough_delta.geojson"
    dgm.to_graphml(G_final, p_graphml)
    dgm.to_geojson(G_final, p_geojson)
    print(f"  -> Saved GraphML to: {p_graphml}")
    print(f"  -> Saved GeoJSON to: {p_geojson}")

    print("\n" + "=" * 70)
    print("Walkthrough completed successfully! Check the outputs/ directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## Next Steps

- Consult the [Configuration Guide](configuration.md) to explore all available YAML parameters and distribution specs.
- Learn about the physics and algorithmic invariants in [Architecture & Mathematical Formulation](architecture.md).
- View empirical literature justifications in [Parameter Justifications](parameter_justifications.md).
- Browse complete method and class signatures in the [API Reference](api_reference.md).
