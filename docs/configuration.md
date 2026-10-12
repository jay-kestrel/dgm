# DGM Configuration & YAML Schema Guide

All physical parameters, boundary conditions, solver settings, and stochastic distributions in DGM can be specified via YAML files or programmatically via [`DGMConfig`](api_reference.md#dgmconfig).

## Complete YAML Schema

```yaml
model:
  N: 50                          # Total node budget (junctions & outlets)
  R_rt: 0.5                      # Relative River-to-Tidal forcing ratio in [0, 1]
  loop_mechanism: "potential_field" # "potential_field" or "ner_optimality"
  sample_parameters: false       # If true, sample active parameters from 'distributions'
  seed: 42                       # Random seed for reproducibility

domain:
  r_shore: 10000.0               # Shoreline radius in meters
  theta_base: 45.0               # Fluvial opening angle in degrees
  theta_delta: 60.0              # Tidal spreading expansion in degrees
  confinement: 0.0               # Lateral valley confinement factor in [0.0, 0.95]
  lateral_valley_width: null     # Optional explicit corridor width (m, |x| <= W/2)

backbone:
  branching_weights: [0.50, 0.45, 0.05] # Probabilities for {1, 2, 3} child branches
  radial_step_scale_min: 0.65    # Minimum radial step scaling factor
  radial_step_scale_max: 1.35    # Maximum radial step scaling factor
  angular_spread_decay: 0.70     # Radial decay of angular perturbation width
  coastal_node_decay: 0.35       # Shoreward contraction rate of radial node spacing
  coastal_sinuosity_decay: 0.30  # Shoreward reach straightening factor

loops:
  candidate_dist_scale: 0.25     # Distance scale l0 = scale * r_shore
  reconnection_prob_base: 0.55   # c0 tidal reconnection factor
  anastomosis_prob_base: 0.0     # P0 inland anastomosis factor
  max_in_degree: 2               # Maximum incoming degree allowed per junction
  max_out_degree: 3              # Maximum outgoing degree allowed per junction
  target_max_loops_slope: 0.22   # Scaling slope for maximum loops vs R_rt
  target_max_loops_intercept: 0.05 # Scaling intercept for maximum loops vs R_rt
  # Simulated Annealing settings (for 'ner_optimality' loop mechanism)
  sa_initial_temp: 1.0
  sa_cooling_rate: 0.95
  sa_max_steps: 250

pruning:
  enable_pruning: false          # Toggle pre-hydraulic pruning & simplification
  prune_inland_dead_ends: true   # Remove blind inland channel fingers (100% ocean delivery)
  outlet_tolerance: 0.05         # Coastal buffer fraction: r >= r_shore * (1 - tol)
  min_spur_length: 0.0           # Length threshold (m) below which dead-end spurs are pruned
  simplify_pass_through_nodes: false # Contract degree-2 chains into direct reaches
  recalculate_reach_sinuosity: true  # Recalculate unified reach sinuosity from process model
  max_prune_iterations: 5        # Upper bound on cascading pruning passes

hydraulics:
  routing_mode: "width"          # Flow partition weighting: "width" or "conductance"
  murray_p: 1.70                 # Murray's Law exponent: w_parent^p = w_1^p + w_2^p
  width_beta: 0.745              # Bifurcation width asymmetry factor
  manning_eta: 1.65              # Width exponent in Manning conductance (Ce = w^eta / L^gamma)
  length_gamma: 0.50             # Length friction damping exponent in conductance
  apex_width: 250.0              # Trunk feeder channel width at apex (m)
  width_noise: 0.30              # Lognormal width variation sigma
  tidal_flare_k: 1.40            # Seaward exponential tidal flaring rate
  tidal_flare_onset: 0.30        # Normalized radius (r / r_shore) where tidal flaring initiates
  loop_flow_penalty: 0.40        # Conductance penalty factor on secondary loop channels
  min_channel_width: 0.1         # Minimum width cutoff (m)
  relaxation_max_iter: 15        # Iteration cap for Murray relaxation solver
  relaxation_tol: 1e-4           # Convergence tolerance for width relaxation

channel_geometry:
  enable_curved_centerlines: false # Generate harmonic sinuous polyline centerlines
  min_channel_clearance: 0.40    # Clearance safety fraction bounding meander amplitude
  num_polyline_points: 21        # Vertex count per reach polyline
  meander_bends: 1               # Number of harmonic meander half-wavelengths per reach

distributions:
  murray_p:
    type: "uniform"
    low: 1.20
    high: 2.20
  width_beta:
    type: "uniform"
    low: 0.50
    high: 0.99
  apex_width:
    type: "uniform"
    low: 50.0
    high: 500.0
  tidal_flare_onset:
    type: "uniform"
    low: 0.10
    high: 0.50
  loop_flow_penalty:
    type: "uniform"
    low: 0.20
    high: 0.70
  outlet_tolerance:
    type: "uniform"
    low: 0.02
    high: 0.10

extensions: {}                   # Custom plugin / hook parameters
```

---

## Supported Distribution Types

When `sample_parameters: true`, active parameters are sampled from the `distributions` block. Supported specifications include:

- **`uniform`**: Requires `low` and `high`.
  ```yaml
  tidal_flare_onset:
    type: "uniform"
    low: 0.15
    high: 0.45
  ```
- **`normal` / `gaussian`**: Requires `mean` and `std`, with optional `min`/`max` clamps.
  ```yaml
  apex_width:
    type: "normal"
    mean: 250.0
    std: 35.0
    min: 50.0
  ```
- **`lognormal`**: Requires `mean` and `sigma`, with optional `min`/`max`.
  ```yaml
  base_sinuosity:
    type: "lognormal"
    mean: 0.15
    sigma: 0.20
  ```
- **`choice` / `categorical`**: Requires `values` list and optional `weights` list.
  ```yaml
  murray_p:
    type: "choice"
    values: [1.50, 1.70, 2.00]
    weights: [0.2, 0.6, 0.2]
  ```
- **`fixed` / `constant`**: Requires `value`.

---

## Working with Configurations in Python

### 1. Loading & Overrides

```python
import dgm

# Load default settings
cfg = dgm.DGMConfig.load_default()

# Load custom YAML with runtime overrides
cfg = dgm.DGMConfig.from_yaml("my_run.yaml", N=75, R_rt=0.8)

# Sample active parameters
sampled = cfg.sample_distribution_parameters()
```

### 2. Setting Parameters & Dynamic Distributions (`set_parameter`)

The `cfg.set_parameter()` method provides a clean, unified interface to adjust any parameter as a constant, a uniform range tuple `(low, high)`, or a full distribution specification:

```python
# Fixed value (removes from stochastic sampling)
cfg.set_parameter("tidal_flare_onset", 0.35)

# Uniform distribution shorthand using a 2-tuple (low, high)
cfg.set_parameter("loop_flow_penalty", (0.25, 0.55))

# Full distribution dictionary
cfg.set_parameter("apex_width", {
    "type": "normal",
    "mean": 300.0,
    "std": 30.0,
    "min": 100.0
})
```

### 3. Inline Distribution Specifications in YAML

In YAML, parameters under any section can also be declared directly as distributions or ranges:

```yaml
hydraulics:
  tidal_flare_onset: [0.20, 0.40]      # Automatically parsed as uniform [0.20, 0.40]
  loop_flow_penalty:
    type: "uniform"
    low: 0.30
    high: 0.60
```

---

## Parameter Validation Rules

`DGMConfig.validate()` automatically validates all parameters before execution:

- **$N$**: Must be $\ge 3$.
- **$R_{\text{rt}}$**: Must be in $[0.0, 1.0]$.
- **$r_{\text{shore}}$**: Must be $> 0$.
- **$\theta_{\text{base}}$, $\theta_{\text{delta}}$**: Must be $> 0$.
- **`confinement`**: Must be in $[0.0, 0.95]$.
- **`outlet_tolerance`**: Must be in $[0.001, 0.50]$.
- **`tidal_flare_onset`**: Must be in $[0.0, 1.0)$.
- **`loop_flow_penalty`**: Must be in $(0.0, 1.0]$.
- **`min_channel_clearance`**: Must be in $[0.05, 1.0]$.
- **`num_polyline_points`**: Must be $\ge 3$.
- **`meander_bends`**: Must be $\ge 1$.
- **`max_prune_iterations`**: Must be $\ge 1$.

---

## Physical Assumptions & Literature Citations

For an in-depth reference table detailing every parameter, its mathematical formulation, default value, bounded distribution range, and peer-reviewed academic literature citations, see [Parameter Assumptions & Literature Justifications](parameter_justifications.md).
