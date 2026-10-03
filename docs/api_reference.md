# DGM Python API Reference

This reference documents the public Python API of **DeltaGraphModel (DGM)** (`import dgm`).

---

## Top-Level Functions

### `dgm.generate(...) -> nx.DiGraph`
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
- `min_channel_clearance` (*float*, optional): Safety fraction bounding meander amplitude near neighboring channels.
- `enable_pruning` (*bool*, optional): If `True`, prunes dead-end inland spurs and contracts chains.
- `prune_inland_dead_ends` (*bool*, optional): If `True`, eliminates inland dead ends (100% shoreline outlet delivery).
- `simplify_pass_through_nodes` (*bool*, optional): If `True`, contracts degree-2 pass-through chains ($d_{\text{in}}=1, d_{\text{out}}=1$).
- `config` (*Union[str, DGMConfig]*, optional): Master configuration object or path to YAML file.
- `seed` (*int*, optional): Random seed for reproducibility.

**Returns**:
- `nx.DiGraph`: Directed planar graph with node Cartesian coordinates (`x`, `y`), polar coordinates (`r`, `theta`), edge widths (`width`), lengths (`length`), sinuosities (`sinuosity`), conductances (`conductance`), and steady-state Markov flux (`flux`).

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

**Returns**:
- `List[nx.DiGraph]`: List of $M$ synthesized graph realizations.

---

## Core Classes

### `dgm.DGMConfig`
Unified configuration dataclass handling baseline parameters, solver thresholds, and bounded distributions.

- `DGMConfig.load_default() -> DGMConfig`: Loads baseline settings from `config/default_config.yaml`.
- `DGMConfig.from_yaml(path: Union[str, Path], **overrides) -> DGMConfig`: Loads custom YAML with runtime overrides.
- `cfg.to_yaml(path: Union[str, Path]) -> None`: Serializes configuration to a YAML file.
- `cfg.get_active_parameters(rng=None, sample=None) -> Dict[str, Any]`: Returns active parameters, drawing from distributions if `sample_parameters=True`.
- `cfg.sample_distribution_parameters(rng=None) -> Dict[str, Any]`: Explicitly samples all parameters declared in the `distributions` block.

---

### `dgm.DeltaGenerator`
Single-realization generator facade.

```python
gen = dgm.DeltaGenerator(config=cfg)
G = gen.generate(seed_override=42)
metrics = gen.evaluate_metrics(G)
```

- `gen.generate(seed_override: Optional[int] = None) -> nx.DiGraph`: Synthesizes graph realization.
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

## IO & Serialization

- `dgm.to_graphml(G: nx.DiGraph, path: Union[str, Path]) -> None`: Writes graph topology, node positions, and hydraulic attributes to GraphML format (compatible with Gephi, Cytoscape, NetworkX).
- `dgm.to_geojson(G: nx.DiGraph, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]`: Exports nodes (Points) and channel reaches (LineStrings) as a standard GeoJSON `FeatureCollection` (for QGIS, ArcGIS, Mapbox).
- `dgm.metrics_to_dataframe(metrics_data) -> pd.DataFrame`: Converts realization dictionary or ensemble summary to a `pandas.DataFrame`.
- `dgm.metrics_to_csv(metrics_data, path: Union[str, Path]) -> None`: Saves metrics table directly to CSV.
- `dgm.DIDNTLoader(dataset_dir: Optional[Union[str, Path]] = None)`: Loads canonical real deltas from the DIDNT global dataset.
- `dgm.get_empirical_deltas() -> List[EmpiricalDelta]`: Returns empirical delta descriptors for all 28 real-world systems.

---

## Benchmark & Diagnostic Modules

### `dgm.benchmarks.empirical_validation.EmpiricalValidationBenchmark`
28-delta global empirical validation engine benchmarking synthetic ensembles against DIDNT field data.
- `suite.run(verbose=True) -> Dict[str, Any]`: Executes benchmark across all 28 deltas.
- `suite.save_csv(path) -> Path`: Exports detailed per-delta metrics to CSV.
- `suite.save_json(path) -> Path`: Exports summary statistics to JSON.
- `suite.plot_benchmark(path) -> Path`: Generates 8-panel publication comparison figure.
- `suite.generate_report(path) -> Path`: Generates comprehensive scientific Markdown report.

### `dgm.benchmarks.mega_delta_benchmark.MegaDeltaScaleBenchmark`
Planetary-scale benchmark engine exercising DGM at full scale ($N \ge 600 - 3,000$ nodes) on Lena, GBMD, Amazon, and Niger.
- `bench.run(verbose=True) -> Dict[str, Any]`: Executes mega-delta simulations.
- `bench.export_csv(path) -> Path`: Exports mega-delta tabular data.
- `bench.export_json(path) -> Path`: Exports JSON statistical summary.
- `bench.export_plot(path) -> Path`: Generates 4-panel comparison plot with flow entropy parity crossplot.
- `bench.export_report(path) -> Path`: Generates mega-delta benchmark report.

### `dgm.benchmarks.reality_check_diagnostics.run_reality_diagnostics`
Evaluates physical module realism: lateral valley confinement, shoreward spatial scaling, proximity sinuosity, and dead-end pruning.
