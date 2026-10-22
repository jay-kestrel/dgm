# DGM Python API Reference

This reference documents the complete public Python API of **DeltaGraphModel (DGM)** (`import dgm`).

---

## Top-Level Functions

<a id="generate"></a>
### `dgm.generate`

Synthesizes a single synthetic delta channel network graph according to boundary conditions and physical parameters.

```python
dgm.generate(
    N: int = 50,
    R_rt: float = 0.5,
    confinement: Optional[float] = None,
    lateral_valley_width: Optional[float] = None,
    coastal_node_decay: Optional[float] = None,
    coastal_sinuosity_decay: Optional[float] = None,
    enable_curved_centerlines: Optional[bool] = None,
    min_channel_clearance: Optional[float] = None,
    enable_pruning: Optional[bool] = None,
    prune_inland_dead_ends: Optional[bool] = None,
    min_spur_length: Optional[float] = None,
    simplify_pass_through_nodes: Optional[bool] = None,
    return_stages: bool = False,
    config: Optional[Union[str, Path, DGMConfig]] = None,
    seed: Optional[int] = None,
    **overrides: Any
) -> Union[nx.DiGraph, Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]]
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `N` | `int` | `50` | Target node budget (junctions and shoreline outlets). Must be $\ge 4$. |
| `R_rt` | `float` | `0.5` | Relative river-to-tidal marine forcing ratio in $[0.0, 1.0]$. $0.0$ generates dendritic fluvial trees; $1.0$ generates looped, flared estuarine networks. |
| `confinement` | `float` | `None` | Lateral valley confinement factor in $[0.0, 0.95]$. Compresses fan opening angle: $\theta_{\text{open}} = \theta_0 (1 - C_{\text{conf}})$. |
| `lateral_valley_width` | `float` | `None` | Maximum corridor width in meters ($\vert x(r) \vert \le W_{\text{valley}} / 2$). Clamps channel excursions to simulate bedrock valley walls. |
| `coastal_node_decay` | `float` | `None` | Shoreward radial node spacing contraction factor in $[0.0, 0.80]$. Modulates jet deceleration toward mouth-bars. |
| `coastal_sinuosity_decay` | `float` | `None` | Shoreward reach straightening factor in $[0.0, 0.90]$. Decreases channel sinuosity near shoreline. |
| `enable_curved_centerlines`| `bool` | `None` | If `True`, generates sinuous 2D polyline centerlines with proximity meander bounds. |
| `min_channel_clearance` | `float` | `None` | Safety fraction bounding meander amplitude near neighboring channels in $[0.05, 1.0]$. |
| `enable_pruning` | `bool` | `None` | If `True`, removes inland dead-end spurs and contracts pass-through nodes. |
| `prune_inland_dead_ends` | `bool` | `None` | If `True`, eliminates dead-ends terminating inland (guarantees 100% shoreline outlet delivery). |
| `min_spur_length` | `float` | `None` | Length threshold in meters below which dead-end channel spurs are pruned. |
| `simplify_pass_through_nodes` | `bool` | `None` | If `True`, contracts degree-2 pass-through chains ($d_{\text{in}}=1, d_{\text{out}}=1$) into single morphological links. |
| `return_stages` | `bool` | `False` | If `True`, returns `(G_final, stages)` capturing snapshots at key pipeline milestones. |
| `config` | `Union[str, Path, DGMConfig]` | `None` | Master configuration object or path to custom YAML file. |
| `seed` | `int` | `None` | Random seed for exact reproducible generation. |
| `**overrides` | `Any` | — | Direct overrides applied to active configuration (e.g. `tidal_flare_onset=0.35`, `loop_flow_penalty=0.50`). |

#### Returns

| Type | Description |
| :--- | :--- |
| `nx.DiGraph` | When `return_stages=False`: Directed planar graph with node coordinates (`x`, `y`, `r`, `theta`) and edge attributes (`width`, `length`, `sinuosity`, `conductance`, `flux`). |
| `Tuple[nx.DiGraph, Dict[str, nx.DiGraph]]` | When `return_stages=True`: Tuple containing the final graph and a dictionary of intermediate stage snapshots (`'tree'`, `'looped'`, `'pruned'`, `'final'`). |

#### Example

```python
import dgm

# Standard generation with sinuous centerlines and pruning
G = dgm.generate(
    N=60,
    R_rt=0.45,
    enable_curved_centerlines=True,
    enable_pruning=True,
    seed=42
)
print(f"Generated delta with {G.number_of_nodes()} nodes and {G.number_of_edges()} channels.")
```

---

<a id="sample_ensemble"></a>
### `dgm.sample_ensemble`

Samples an ensemble array of $M$ independent graph realizations conditioned on boundary conditions and stochastic parameter distributions.

```python
dgm.sample_ensemble(
    N: int = 50,
    R_rt: float = 0.5,
    num_realizations: int = 50,
    config: Optional[Union[str, Path, DGMConfig]] = None,
    base_seed: int = 42,
    **overrides: Any
) -> List[nx.DiGraph]
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `N` | `int` | `50` | Target node budget for each realization. |
| `R_rt` | `float` | `0.5` | River-to-tidal forcing ratio. |
| `num_realizations` | `int` | `50` | Total number of realizations to sample ($M \ge 1$). |
| `config` | `Union[str, Path, DGMConfig]` | `None` | Master configuration object or path to YAML file. |
| `base_seed` | `int` | `42` | Base random seed (each realization $i$ uses `base_seed + i * 1000`). |
| `**overrides` | `Any` | — | Parameter overrides forwarded to all realizations. |

#### Returns

| Type | Description |
| :--- | :--- |
| `List[nx.DiGraph]` | List of $M$ synthesized graph realizations. |

#### Example

```python
import dgm

# Sample 100 stochastic realizations
ensemble = dgm.sample_ensemble(N=40, R_rt=0.60, num_realizations=100, base_seed=42)
print(f"Sampled {len(ensemble)} realizations.")
```

---

## Core Configuration & Generation Classes

<a id="dgmconfig"></a>
### `dgm.DGMConfig`

Unified dataclass encapsulating all model hyperparameters, numerical solver tolerances, domain boundaries, and stochastic parameter distributions.

```python
from dgm import DGMConfig

cfg = DGMConfig.load_default()
```

#### Core Attributes

| Attribute | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `N` | `int` | `50` | Target node budget ($N \ge 4$). |
| `R_rt` | `float` | `0.50` | River-to-tidal forcing ratio ($0.0 \le R_{\text{rt}} \le 1.0$). |
| `loop_mechanism` | `str` | `"potential_field"` | Loop reconnection engine: `'potential_field'` or `'ner_optimality'`. |
| `r_shore` | `float` | `10000.0` | Radial distance from apex to receiving shoreline (meters). |
| `theta_base` | `float` | `65.0` | Baseline opening angle in degrees for river-dominated deltas. |
| `theta_delta` | `float` | `60.0` | Additional marine opening angle spreading capacity in degrees. |
| `confinement` | `float` | `0.0` | Valley confinement coefficient in $[0.0, 0.95]$. |
| `lateral_valley_width`| `float` | `None` | Optional valley corridor width limit (meters). |
| `murray_p` | `float` | `1.85` | Murray's Law width conservation exponent ($p > 0$). |
| `tidal_flare_k` | `float` | `1.40` | Exponential mouth flaring rate coefficient. |
| `tidal_flare_onset` | `float` | `0.30` | Relative radial distance where tidal mouth flaring initiates. |
| `loop_flow_penalty` | `float` | `0.40` | Conductance reduction factor for secondary anastomosing cross-ties. |
| `enable_pruning` | `bool` | `False` | Whether dead-end spur pruning is enabled. |
| `simplify_pass_through_nodes` | `bool` | `False` | Whether degree-2 pass-through chains are contracted. |
| `enable_curved_centerlines` | `bool` | `False` | Whether curved polyline meander coordinates are synthesized. |
| `distributions` | `Dict[str, Any]`| `{}` | Parameter distribution specifications for stochastic sampling. |

#### Methods

| Method | Return Type | Description |
| :--- | :---: | :--- |
| `load_default()` | `DGMConfig` | Loads default configuration packaged with DGM. |
| `from_yaml(path, **overrides)` | `DGMConfig` | Loads configuration from custom YAML file with keyword overrides. |
| `to_yaml(path)` | `None` | Serializes configuration to a YAML file. |
| `set_parameter(name, val_or_spec)` | `None` | Sets a fixed value, uniform range `(low, high)`, or distribution dict. |
| `get_active_parameters(rng, sample)` | `Dict[str, Any]` | Returns dictionary of active parameter values for a realization. |
| `validate()` | `None` | Validates all physical bounds and invariants, raising `ValueError` on violation. |

---

<a id="deltagenerator"></a>
### `dgm.DeltaGenerator`

Single-realization generator facade providing fine-grained control over execution and intermediate stage extraction.

```python
from dgm import DeltaGenerator, DGMConfig

cfg = DGMConfig.load_default()
gen = DeltaGenerator(config=cfg)
G = gen.generate(seed_override=42)
metrics = gen.evaluate_metrics(G)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `config` | `Optional[DGMConfig]` | `None` | Master configuration dataclass. Defaults to `DGMConfig.load_default()`. |
| `target_nodes` | `Optional[int]` | `None` | Explicit node budget override. |
| `**overrides` | `Any` | — | Runtime parameter overrides applied to the configuration. |

#### Methods

| Method | Parameters | Return Type | Description |
| :--- | :--- | :---: | :--- |
| `generate` | `seed_override=None, return_stages=False` | `nx.DiGraph` or `Tuple` | Synthesizes graph realization, optionally capturing intermediate stages. |
| `evaluate_metrics`| `G: nx.DiGraph` | `Dict[str, Any]` | Evaluates the full 20-metric geomorphic battery on graph `G`. |

---

<a id="ensemblegenerator"></a>
### `dgm.EnsembleGenerator`

Stochastic cloud generator for statistical ensemble sampling and Monte Carlo morphological analysis.

```python
from dgm import EnsembleGenerator

ens = EnsembleGenerator(
    N=50,
    R_rt=0.55,
    num_realizations=100,
    sample_parameters=True,
    base_seed=42
)
graphs = ens.generate_ensemble()
summary = ens.evaluate_ensemble()
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `N` | `int` | `50` | Node budget for realizations. |
| `R_rt` | `float` | `0.5` | River-to-tidal forcing ratio. |
| `num_realizations`| `int` | `100` | Number of graphs to sample ($M \ge 1$). |
| `loop_mechanism` | `str` | `"potential_field"` | Loop engine (`'potential_field'` or `'ner_optimality'`). |
| `sample_parameters`| `bool` | `True` | If `True`, draws active parameters from stochastic distributions. |
| `base_seed` | `int` | `42` | Seed offset for deterministic reproducibility. |
| `config` | `Optional[DGMConfig]` | `None` | Base configuration dataclass. |
| `**overrides` | `Any` | — | Runtime parameter overrides. |

#### Methods

| Method | Parameters | Return Type | Description |
| :--- | :--- | :---: | :--- |
| `generate_ensemble`| — | `List[nx.DiGraph]` | Synthesizes and returns list of $M$ graph realizations. |
| `evaluate_ensemble`| — | `Dict[str, Any]` | Generates ensemble and calculates statistical distributions (`mean`, `std`, `min`, `max`, `p25`, `p75`) across all 20 metrics. |

---

<a id="metricsbattery"></a>
### `dgm.MetricsBattery`

Comprehensive diagnostic metric battery computing 20 canonical geomorphic metrics across topological, dynamic, vulnerability, and hydraulic form tiers.

```python
from dgm import MetricsBattery

battery = MetricsBattery()
metrics = battery.evaluate_all(G)
```

#### Methods

| Method | Parameters | Return Type | Description |
| :--- | :--- | :---: | :--- |
| `evaluate_all` | `G: nx.DiGraph, params=None` | `Dict[str, Any]` | Computes complete 20-metric battery (returns 33 diagnostic fields). |
| `evaluate_topology` | `G: nx.DiGraph` | `Dict[str, Any]` | Computes $\alpha$, $\beta_1$, $V_b$, $V_c$, $V_c/V_b$, LSI, and resistance distance. |
| `evaluate_dynamics` | `G: nx.DiGraph, params, markov_sol=None` | `Dict[str, Any]` | Computes $nER^*$, $eBI$, energy dissipation $D_{\text{net}}$, and algebraic connectivity. |
| `evaluate_vulnerability` | `G: nx.DiGraph, params, markov_sol=None` | `Dict[str, Any]` | Computes global vulnerability $V_{\text{net}}$ and outlet flux entropy $H_{\text{out}}$. |
| `evaluate_geomorphology` | `G: nx.DiGraph` | `Dict[str, Any]` | Computes mean width, width CV, radial sinuosity, and angular dispersion. |

#### Canonical 20-Metric Summary Reference

| Category | Metric Key | Symbol | Empirical Meaning |
| :--- | :--- | :---: | :--- |
| **Topological Form** | `alpha_index` | $\alpha$ | Meshedness coefficient measuring planar loop redundancy. |
| | `betti_1_norm` | $\beta_1/N$ | Normalized loop density per node. |
| | `ratio_vc_vb` | $V_c/V_b$ | Ratio of confluences to bifurcations. |
| | `topological_slope` | $\Gamma$ | Rate of loop growth relative to recombination. |
| | `horton_inversion_index`| $H_{\text{inv}}$ | Balance of branching vs. confluence dominance. |
| | `link_sharing_index` | $\text{LSI}$ | Normalized path redundancy from apex to shoreline outlets. |
| | `resistance_distance` | $\text{RD}$ | Effective unweighted graph resistance from apex to sea. |
| **Dynamic & Entropic** | `nER_star` | $nER^*$ | Non-Local Shannon Entropy Rate normalized to $[0, 1]$. |
| | `eBI` | $\text{eBI}$ | Entropic Braiding Index (effective active channels across transects). |
| | `energy_dissipation` | $D_{\text{net}}$ | Total hydrodynamic energy dissipation: $\sum Q_e^2 / C_e$. |
| | `algebraic_connectivity`| $\lambda_2$ | Fiedler eigenvalue of normalized symmetric Laplacian. |
| | `energy_entropy_ratio` | $E_{\text{ratio}}$ | Ratio of normalized entropy rate to energy dissipation. |
| **Vulnerability** | `global_vulnerability`| $V_{\text{net}}$ | Average network sensitivity to single-channel disruptions. |
| | `outlet_flux_entropy` | $H_{\text{out}}$ | Shannon entropy in bits of discharge partition across outlets. |
| **Geomorphic Form** | `mean_width` | $\mu_w$ | Mean emergent channel width in meters. |
| | `width_cv` | $\sigma_w / \mu_w$ | Coefficient of variation of channel widths. |
| | `radial_sinuosity` | $\chi_{\text{rad}}$ | Path length to outlet divided by straight-line radial distance. |
| | `angular_dispersion` | $\theta_{\text{disp}}$ | Standard deviation of branch deflection angles at bifurcations. |
| | `loop_length_anisotropy`| $\Lambda_{\text{loop}}$| Ratio of mean tree link length to mean loop link length. |
| | `num_outlets` | $N_{\text{out}}$ | Number of terminal outlet channels reaching the sea. |

---

<a id="polardomain"></a>
### `dgm.PolarDomain`

Manages polar coordinate transformations, fan opening geometry, and lateral valley confinement boundaries.

```python
from dgm import PolarDomain

domain = PolarDomain(
    r_shore=10000.0,
    theta_base=65.0,
    theta_delta=60.0,
    confinement=0.40,
    lateral_valley_width=6000.0
)
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `r_shore` | `float` | `10000.0` | Radial distance from apex to shoreline (meters). |
| `theta_base` | `float` | `65.0` | Baseline opening angle in degrees. |
| `theta_delta` | `float` | `60.0` | Marine tidal spreading capacity in degrees. |
| `confinement` | `float` | `0.0` | Lateral valley confinement factor in $[0.0, 0.95]$. |
| `lateral_valley_width` | `Optional[float]` | `None` | Lateral corridor width limit in meters. |

#### Methods

| Method | Parameters | Return Type | Description |
| :--- | :--- | :---: | :--- |
| `opening_angle` | `R_rt: float` | `float` | Computes effective planform opening angle $\theta_{\text{open}}(R_{\text{rt}}, C_{\text{conf}})$. |
| `is_in_bounds` | `r: float, theta: float, R_rt: float` | `bool` | Verifies whether polar coordinates fall within fan and valley limits. |
| `clamp_coordinates` | `x: float, y: float` | `Tuple[float, float]` | Clamps Cartesian coordinates to valley width ($\vert x \vert \le W_{\text{valley}}/2$). |

---

## Spatial Network Cartography & Visualization (`[viz]`)

<a id="plot_network"></a>
### `dgm.plot_network`

*(Alias: `dgm.plot_delta`)*

Renders publication-quality 2D spatial maps of delta networks, supporting multi-vertex sinuous polylines, width-weighted channels, and geomorphic themes.

```python
dgm.plot_network(
    G: nx.DiGraph,
    ax: Optional[plt.Axes] = None,
    curved: bool = True,
    edge_width: Union[str, float] = "width",
    min_linewidth: float = 0.8,
    max_linewidth: float = 5.5,
    width_scaling: str = "sqrt",
    edge_color: Union[str, Dict[str, str]] = "type",
    edge_alpha: float = 0.85,
    show_nodes: bool = True,
    node_size: float = 30.0,
    node_style: str = "type",
    show_domain: bool = True,
    orientation: str = "cartesian",
    theme: str = "light",
    title: Optional[str] = "Synthetic River Delta",
    show_legend: bool = True,
    show_stats: bool = False,
    domain: Optional[PolarDomain] = None
) -> plt.Axes
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `G` | `nx.DiGraph` | *required* | Network graph with node positions (`x`, `y`) and edge attributes (`width`, `sinuosity`, optional `coordinates`). |
| `ax` | `Optional[plt.Axes]` | `None` | Target Matplotlib Axes. If `None`, creates a new figure. |
| `curved` | `bool` | `True` | If `True`, renders multi-vertex sinuous polylines from edge `coordinates`. |
| `edge_width` | `Union[str, float]` | `"width"` | Attribute key for linewidth scaling or fixed float width. |
| `min_linewidth` / `max_linewidth` | `float` | `0.8` / `5.5` | Minimum and maximum line thickness in points. |
| `width_scaling` | `str` | `"sqrt"` | Scaling function: `'sqrt'` ($W \propto \sqrt{w}$ diameter), `'linear'`, or `'fixed'`. |
| `edge_color` | `Union[str, Dict[str, str]]` | `"type"` | Channel coloring strategy: `'type'` (distinguishes tree, loop, tidal loop), or specific color. |
| `edge_alpha` | `float` | `0.85` | Alpha opacity for channel lines in $[0.0, 1.0]$. |
| `show_nodes` | `bool` | `True` | Whether node markers are rendered. |
| `node_style` | `str` | `"type"` | Node marker styling: `'type'` (apex, outlets, bifurcations, confluences) or `'uniform'`. |
| `node_size` | `float` | `30.0` | Node marker point size. |
| `show_domain` | `bool` | `True` | Draws polar fan boundary rays, shoreline arc, and valley walls. |
| `orientation` | `str` | `"cartesian"` | Viewing orientation: `'cartesian'` (apex origin, seaward $+x$), `'downward'` (apex top, seaward $-y$), or `'upward'`. |
| `theme` | `str` | `"light"` | Visual theme: `'light'`, `'dark'`, or `'clean'`. |
| `title` | `Optional[str]` | `"Synthetic River Delta"` | Title string displayed above plot. |
| `show_legend` | `bool` | `True` | Displays legend classifying channel reach types and junctions. |
| `show_stats` | `bool` | `False` | Overlays a summary badge with $N, E, \beta_1$, mean width, and sinuosity. |
| `domain` | `Optional[PolarDomain]` | `None` | Explicit domain instance defining boundary geometry. |

#### Returns

| Type | Description |
| :--- | :--- |
| `plt.Axes` | The Matplotlib Axes containing the rendered network. |

---

<a id="plot_tree_vs_network"></a>
### `dgm.plot_tree_vs_network`

Renders a synchronized side-by-side comparative figure displaying the initial Polar Tree Backbone alongside the final Hydraulic Network with emergent loops and sinuous channels.

```python
dgm.plot_tree_vs_network(
    G_tree: nx.DiGraph,
    G_final: nx.DiGraph,
    curved: bool = True,
    theme: str = "light",
    **kwargs: Any
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
```

#### Returns

| Type | Description |
| :--- | :--- |
| `Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]` | Figure object and pair of `(ax_tree, ax_final)` Axes. |

---

<a id="plot_stages"></a>
### `dgm.plot_stages`

Renders a progressive 4-panel assembly figure showing the generative evolution of the network across key milestones: `'tree'`, `'looped'`, `'pruned'`, and `'final'`.

```python
dgm.plot_stages(
    stages: Dict[str, nx.DiGraph],
    curved: bool = True,
    theme: str = "light",
    **kwargs: Any
) -> Tuple[plt.Figure, List[plt.Axes]]
```

#### Returns

| Type | Description |
| :--- | :--- |
| `Tuple[plt.Figure, List[plt.Axes]]` | Figure object and list of 4 panel Axes. |

---

## IO & Export Utilities

| Function | Parameters | Return Type | Description |
| :--- | :--- | :---: | :--- |
| `dgm.to_graphml` | `G, path` | `None` | Exports graph topology, node positions, and hydraulic attributes to standard GraphML (for Gephi, Cytoscape, NetworkX). |
| `dgm.to_geojson` | `G, path=None` | `Dict[str, Any]` | Exports nodes (`Point`) and channels (`LineString`) as a GeoJSON `FeatureCollection` (for QGIS, ArcGIS, Mapbox). |
| `dgm.metrics_to_dataframe` | `metrics_data` | `pd.DataFrame` | Converts realization metrics dict or ensemble raw list to a Pandas DataFrame. |
| `dgm.metrics_to_csv` | `metrics_data, path` | `None` | Exports realization metrics directly to CSV (with standard library fallback). |
| `dgm.summary_to_dataframe` | `summary` | `pd.DataFrame` | Converts ensemble aggregated summary statistics into a Pandas DataFrame. |
| `dgm.summary_to_csv` | `summary, path` | `None` | Exports ensemble aggregated summary statistics directly to CSV. |
| `dgm.get_empirical_deltas` | — | `List[EmpiricalDelta]` | Returns descriptors and feature vectors for all 28 real-world deltas in the DIDNT dataset. |
| `dgm.DIDNTLoader` | `dataset_dir=None` | `DIDNTLoader` | Loader utility for loading and parsing canonical empirical delta networks. |

---

## Command Line Interface (CLI)

The `dgm` CLI provides access to single generation and ensemble sampling directly from the terminal.

### `dgm generate`

```bash
dgm generate -N 50 -R 0.45 --curved --enable-pruning -s 42 -o delta.graphml
```

| Option | Flag | Type | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `--nodes` | `-N` | `int` | `50` | Target node budget ($N \ge 4$). |
| `--ratio` | `-R` | `float` | `0.5` | River-to-tidal forcing ratio ($0.0 \le R_{\text{rt}} \le 1.0$). |
| `--config` | `-c` | `path` | `None` | Path to custom YAML configuration. |
| `--seed` | `-s` | `int` | `None` | Random seed for exact reproducibility. |
| `--mechanism` | `-m` | `str` | `potential_field` | Loop engine: `'potential_field'` or `'ner_optimality'`. |
| `--curved` | — | `flag` | `False` | Enable sinuous polyline centerlines. |
| `--enable-pruning` | — | `flag` | `False` | Enable dead-end spur removal and reach simplification. |
| `--output` | `-o` | `path` | `None` | Output filepath (`.graphml`, `.geojson`, `.png`, `.pdf`, `.svg`). |

### `dgm ensemble`

```bash
dgm ensemble -N 50 -R 0.55 -n 100 --sample-parameters -o ensemble.csv
```

| Option | Flag | Type | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `--nodes` | `-N` | `int` | `50` | Node budget for realizations. |
| `--ratio` | `-R` | `float` | `0.5` | River-to-tidal forcing ratio. |
| `--num` | `-n` | `int` | `50` | Number of realizations to sample. |
| `--sample-parameters` | — | `flag` | `False` | Enable stochastic parameter sampling from distributions. |
| `--base-seed` | `-s` | `int` | `42` | Base random seed offset. |
| `--output` | `-o` | `path` | `None` | Output filepath for CSV metric summary. |
