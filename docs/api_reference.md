# DGM Python API Reference

## Top-Level Functions

### `dgm.generate(N=50, R_rt=0.5, config=None, seed=None, **overrides) -> nx.DiGraph`
Synthesizes a single synthetic delta channel network graph.

### `dgm.sample_ensemble(N=50, R_rt=0.5, num_realizations=50, config=None, base_seed=42, **overrides) -> List[nx.DiGraph]`
Generates an ensemble array of $M$ graph realizations across the parameter space.

---

## Classes

### `dgm.DGMConfig`
Unified configuration dataclass.
- `DGMConfig.load_default() -> DGMConfig`
- `DGMConfig.from_yaml(path, **overrides) -> DGMConfig`
- `cfg.to_yaml(path) -> None`
- `cfg.get_active_parameters(rng=None, sample=None) -> Dict[str, Any]`
- `cfg.sample_distribution_parameters(rng=None) -> Dict[str, Any]`

### `dgm.DeltaGenerator(config=None, **overrides)`
Single-realization generator facade.
- `gen.generate(seed_override=None) -> nx.DiGraph`
- `gen.evaluate_metrics(G) -> Dict[str, Any]`

### `dgm.EnsembleGenerator(N=50, R_rt=0.5, num_realizations=100, ...)`
Ensemble generator facade for cloud sampling.
- `ens.generate_ensemble() -> List[nx.DiGraph]`
- `ens.evaluate_ensemble() -> Dict[str, Any]`

### `dgm.MetricsBattery(manning_eta=1.65, length_gamma=0.50)`
20-metric geomorphic battery.
- `battery.evaluate_all(G, params=None) -> Dict[str, Any]`
- `battery.evaluate_topology(G) -> Dict[str, Any]`
- `battery.evaluate_dynamics(G, params) -> Dict[str, Any]`
- `battery.evaluate_vulnerability(G, params) -> Dict[str, Any]`
- `battery.evaluate_geomorphology(G) -> Dict[str, Any]`

---

## IO & Export

- `dgm.to_graphml(G, path)`: Writes graph to GraphML format.
- `dgm.to_geojson(G, path=None)`: Exports nodes (points) and channels (linestrings) to GeoJSON.
- `dgm.metrics_to_dataframe(metrics)`: Converts metrics to a `pandas.DataFrame`.
- `dgm.metrics_to_csv(metrics, path)`: Saves metrics to a CSV file.
- `dgm.DIDNTLoader(dataset_dir=None)`: Parses real deltas from the DIDNT dataset.
- `dgm.get_empirical_deltas()`: Returns empirical delta summary structures.
