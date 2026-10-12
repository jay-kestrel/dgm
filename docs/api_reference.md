# DGM Python API Reference

This reference documents the public Python API of **DeltaGraphModel (DGM)** (`import dgm`).

---

## Top-Level Functions

### `dgm.generate(...) -> Union[nx.DiGraph, Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]]`
Synthesizes a single synthetic delta channel network graph.

```python
G = dgm.generate(
    N=50,
    R_rt=0.5,
    confinement=None,
    lateral_valley_width=None,
    coastal_node_decay=None,
    coastal_sinuosity_decay=None,
    enable_curved_centerlines=None,
    min_channel_clearance=None,
    enable_pruning=None,
    prune_inland_dead_ends=None,
    min_spur_length=None,
    simplify_pass_through_nodes=None,
    return_stages=False,
    config=None,
    seed=None,
    **overrides
)
```

**Parameters**:
- `N` (*int*, default 50): Target node budget (junctions and outlets).
- `R_rt` (*float*, default 0.5): Relative river-to-tidal marine forcing ratio in $[0.0, 1.0]$.
- `confinement` (*float*, optional): Lateral valley confinement factor in $[0.0, 0.95]$.
- `lateral_valley_width` (*float*, optional): Maximum corridor width in meters ($|x(r)| \le W/2$).
- `coastal_node_decay` (*float*, optional): Shoreward radial node spacing contraction factor in $[0.0, 0.8]$.
- `coastal_sinuosity_decay` (*float*, optional): Shoreward reach straightening factor in $[0.0, 0.9]$.
- `enable_curved_centerlines` (*bool*, optional): If `True`, generates sinuous 2D polyline centerlines.
- `min_channel_clearance` (*float*, optional): Safety fraction bounding meander amplitude near neighboring channels in $[0.05, 1.0]$.
- `enable_pruning` (*bool*, optional): If `True`, prunes dead-end inland spurs and contracts chains.
- `prune_inland_dead_ends` (*bool*, optional): If `True`, eliminates inland dead ends (100% shoreline outlet delivery).
- `min_spur_length` (*float*, optional): Length threshold below which dead-end spurs are pruned.
- `simplify_pass_through_nodes` (*bool*, optional): If `True`, contracts degree-2 pass-through chains ($d_{\text{in}}=1, d_{\text{out}}=1$).
- `return_stages` (*bool*, default `False`): If `True`, returns `(G_final, stages)` where `stages` is a dictionary mapping stage names (`'tree'`, `'looped'`, `'pruned'`, `'final'`) to their respective graph snapshots.
- `config` (*Union[str, DGMConfig]*, optional): Master configuration object or path to YAML file.
- `seed` (*int*, optional): Random seed for reproducibility.
- `**overrides`: Direct parameter overrides applied to the active configuration (e.g. `tidal_flare_onset=0.35`, `loop_flow_penalty=0.50`, `outlet_tolerance=0.04`).

**Returns**:
- `nx.DiGraph` (if `return_stages=False`): Directed planar graph with node Cartesian coordinates (`x`, `y`), polar coordinates (`r`, `theta`), edge widths (`width`), lengths (`length`), sinuosities (`sinuosity`), conductances (`conductance`), and steady-state Markov flux (`flux`).
- `Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]` (if `return_stages=True`): Tuple of final graph and dictionary of intermediate stage graph snapshots.

---

### `dgm.sample_ensemble(...) -> List[nx.DiGraph]`
Generates an ensemble array of $M$ graph realizations across the parameter space.

```python
graphs = dgm.sample_ensemble(
    N=50,
    R_rt=0.5,
    num_realizations=50,
    config=None,
    base_seed=42,
    **overrides
)
```

**Parameters**:
- `N` (*int*, default 50): Target node budget.
- `R_rt` (*float*, default 0.5): River-to-tidal forcing ratio.
- `num_realizations` (*int*, default 50): Number of realizations to sample.
- `config` (*Union[str, DGMConfig]*, optional): Configuration object or path.
- `base_seed` (*int*, default 42): Base random seed (each realization uses `base_seed + i * 1000`).
- `**overrides`: Additional parameter overrides forwarded to each realization.

**Returns**:
- `List[nx.DiGraph]`: List of $M$ synthesized graph realizations.

---

## Core Classes

### `dgm.DGMConfig`
Unified configuration dataclass handling baseline parameters, solver thresholds, and bounded distributions.

```python
cfg = dgm.DGMConfig.load_default()
cfg.N = 60
cfg.set_parameter("tidal_flare_onset", 0.35)
cfg.set_parameter("loop_flow_penalty", (0.25, 0.55))
```

**Key Methods**:
- `DGMConfig.load_default() -> DGMConfig`: Loads baseline settings from package default configuration.
- `DGMConfig.from_yaml(path: Union[str, Path], **overrides) -> DGMConfig`: Loads custom YAML with runtime overrides and inline distribution parsing.
- `cfg.to_yaml(path: Union[str, Path]) -> None`: Serializes configuration to a YAML file.
- `cfg.set_parameter(name: str, value_or_spec: Union[float, int, str, bool, Tuple[float, float], Dict[str, Any]]) -> None`: Sets a parameter as a fixed value, a uniform distribution tuple `(low, high)`, or a distribution dictionary (`{'type': 'normal', ...}`).
- `cfg.get_active_parameters(rng=None, sample=None) -> Dict[str, Any]`: Returns active parameters, drawing from distributions if `sample_parameters=True`.
- `cfg.sample_distribution_parameters(rng=None) -> Dict[str, Any]`: Explicitly samples all parameters declared in the `distributions` block.
- `cfg.validate() -> None`: Validates all physical parameter bounds and domain consistency.

---

### `dgm.DeltaGenerator`
Single-realization generator facade.

```python
gen = dgm.DeltaGenerator(config=cfg)
G = gen.generate(seed_override=42, return_stages=False)
metrics = gen.evaluate_metrics(G)
```

- `gen.generate(seed_override=None, return_stages=False) -> Union[nx.DiGraph, Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]]`: Synthesizes graph realization, optionally capturing intermediate stages.
- `gen.evaluate_metrics(G: nx.DiGraph) -> Dict[str, Any]`: Evaluates the full 20-metric geomorphic battery.

---

### `dgm.EnsembleGenerator`
Ensemble generator facade for cloud sampling and statistical evaluation.

```python
ens = dgm.EnsembleGenerator(
    N=50,
    R_rt=0.6,
    num_realizations=100,
    sample_parameters=True,
    base_seed=100,
    config=cfg
)
graphs = ens.generate_ensemble()
summary = ens.evaluate_ensemble()
```

- `ens.generate_ensemble() -> List[nx.DiGraph]`: Generates list of graph realizations.
- `ens.evaluate_ensemble() -> Dict[str, Any]`: Generates graphs and computes cloud statistics (`mean`, `std`, `min`, `max`, `p10`, `p25`, `p50`, `p75`, `p90`) for all 20 metrics.

---

### `dgm.MetricsBattery`
Evaluates the 20-metric geomorphic battery across topological, entropic, energetic, and hydraulic properties.

- `battery.evaluate_all(G: nx.DiGraph, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`: Evaluates all 20 metrics.
- `battery.evaluate_topology(G) -> Dict[str, Any]`: $\alpha$, $\beta_1$, $V_b$, $V_c$, $V_c/V_b$, topological slope $\Gamma$.
- `battery.evaluate_dynamics(G, params) -> Dict[str, Any]`: $nER^*$, $eBI$, total dissipation $D_{\text{net}}$, variance dissipation.
- `battery.evaluate_vulnerability(G, params) -> Dict[str, Any]`: Global network vulnerability $V_{\text{net}}$, max channel vulnerability.
- `battery.evaluate_geomorphology(G) -> Dict[str, Any]`: Mean width, width CV, reach sinuosity, shoreline delivery ratio.

---

### `dgm.PolarDomain`
Encapsulates polar fan domain coordinates, lateral valley confinement, and angular opening boundaries.

```python
domain = dgm.PolarDomain(
    r_shore=10000.0,
    theta_base=45.0,
    theta_delta=60.0,
    confinement=0.3,
    lateral_valley_width=8000.0
)
```

- `domain.opening_angle(R_rt: float) -> float`: Computes effective opening angle $\theta_{\text{open}}(R_{\text{rt}}, C_{\text{conf}})$.
- `domain.is_in_bounds(r: float, theta: float, R_rt: float) -> bool`: Verifies polar and lateral Cartesian limits.
- `domain.clamp_coordinates(x: float, y: float) -> Tuple[float, float]`: Enforces lateral valley bounds $|x| \le W/2$.

---

## Spatial Network Visualization

### `dgm.plot_network(G: nx.DiGraph, ax=None, curved=True, ...) -> plt.Axes`
*(Alias: `dgm.plot_delta`)*

Renders a geomorphically accurate 2D spatial map of a DGM network or tree backbone preserving physical geometry.

```python
ax = dgm.plot_network(
    G,
    ax=None,
    curved=True,
    edge_width="width",
    min_linewidth=0.8,
    max_linewidth=5.5,
    width_scaling="sqrt",
    edge_color="type",
    edge_alpha=0.85,
    show_nodes=True,
    node_size=30.0,
    node_style="type",
    show_domain=True,
    orientation="cartesian",
    theme="light",
    title="Synthetic River Delta",
    show_legend=True,
    show_stats=False,
    domain=None
)
```

**Parameters**:
- `G` (*nx.DiGraph*): DGM network graph with node positions (`x`, `y`) and edge attributes (`width`, `sinuosity`, optional `coordinates`).
- `ax` (*plt.Axes*, optional): Target Matplotlib Axes. If `None`, creates a new figure and axes.
- `curved` (*bool*, default `True`): If `True`, renders multi-vertex sinuous polylines from edge `coordinates`.
- `edge_width` (*Union[str, float]*, default `'width'`): Channel thickness attribute name or constant float linewidth.
- `width_scaling` (*str*, default `'sqrt'`): Scaling mode for linewidths: `'sqrt'` ($W \propto \sqrt{w}$ cross-sectional diameter), `'linear'`, or `'fixed'`.
- `min_linewidth` / `max_linewidth` (*float*, default `0.8` / `5.5`): Minimum and maximum line thickness in points.
- `edge_color` (*Union[str, Dict[str, str]]*, default `'type'`): Colors channels by edge type (`'tree'`, `'loop'`, `'tidal_loop'`, `'crevasse_splay'`) or a specific color.
- `show_nodes` (*bool*, default `True`): Renders geomorphic node markers.
- `node_style` (*str*, default `'type'`): Node styling strategy: `'type'` (distinguishes apex, outlets, bifurcations, confluences, dead-ends) or `'uniform'`.
- `show_domain` (*bool*, default `True`): Draws polar opening rays, shoreline boundary arc, and valley walls.
- `orientation` (*str*, default `'cartesian'`): Viewing orientation: `'cartesian'` (apex origin, seaward $+x$), `'downward'` (apex top, seaward $-y$), or `'upward'` (apex bottom, seaward $+y$).
- `theme` (*str*, default `'light'`): Color palette theme (`'light'`, `'dark'`, `'clean'`).
- `show_stats` (*bool*, default `False`): Overlays summary statistics badge ($N, E, \beta_1$, mean width, sinuosity).

**Returns**:
- `plt.Axes`: The Matplotlib Axes containing the plot.

---

### `dgm.plot_tree_vs_network(G_tree, G_final, curved=True, ...) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]`
Renders a synchronized side-by-side comparative figure displaying the Stage 1 Polar Tree Backbone versus the Stage 6 Final Hydraulic Network with meanders and loops.

**Returns**:
- `Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]`: Matplotlib Figure and pair of `(ax_tree, ax_final)` Axes.

---

### `dgm.plot_stages(stages: Dict[str, nx.DiGraph], curved=True, ...) -> Tuple[plt.Figure, List[plt.Axes]]`
Renders a progressive multi-stage assembly figure (Tree Backbone $\to$ + Emergent Loops $\to$ Simplified & Pruned $\to$ Final Murray Hydraulics & Meanders).

**Returns**:
- `Tuple[plt.Figure, List[plt.Axes]]`: Matplotlib Figure and list of Axes for each stage.

---

## IO & Serialization

- `dgm.to_graphml(G: nx.DiGraph, path: Union[str, Path]) -> None`: Writes graph topology, node positions, and hydraulic attributes to GraphML format (compatible with Gephi, Cytoscape, NetworkX).
- `dgm.to_geojson(G: nx.DiGraph, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]`: Exports nodes (Points) and channel reaches (LineStrings) as a standard GeoJSON `FeatureCollection` (for QGIS, ArcGIS, Mapbox).
- `dgm.metrics_to_dataframe(metrics_data) -> pd.DataFrame`: Converts realization dictionary or ensemble summary to a `pandas.DataFrame`.
- `dgm.metrics_to_csv(metrics_data, path: Union[str, Path]) -> None`: Saves metrics table directly to CSV.
- `dgm.DIDNTLoader(dataset_dir: Optional[Union[str, Path]] = None)`: Loads canonical real deltas from the DIDNT global dataset.
- `dgm.get_empirical_deltas() -> List[EmpiricalDelta]`: Returns empirical delta descriptors for all 28 real-world systems.

---

## Command Line Interface (CLI)

The `dgm` CLI provides access to single generation and ensemble sampling directly from the shell:

```bash
# Generate a single graph realization and save as GraphML or GeoJSON
dgm generate -N 50 -R 0.45 --curved --enable-pruning -o outputs/delta.graphml
dgm generate -N 50 -R 0.45 --curved --enable-pruning -o outputs/delta.geojson

# Render directly to an image (.png, .pdf, .svg)
dgm generate -N 50 -R 0.45 --curved --enable-pruning -o outputs/delta.png

# Sample an ensemble cloud and export statistics to CSV
dgm ensemble -N 50 -R 0.65 -n 100 --sample-parameters -o outputs/ensemble.csv
```
