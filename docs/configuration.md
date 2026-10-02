# DGM Configuration & YAML Schema Guide

All physical parameters, boundary conditions, solver settings, and stochastic distributions in DGM can be specified via YAML files.

## YAML Structure

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

backbone:
  branching_weights: [0.50, 0.45, 0.05] # Probabilities for {1, 2, 3} children
  radial_step_scale_min: 0.65
  radial_step_scale_max: 1.35
  angular_spread_decay: 0.70

loops:
  candidate_dist_scale: 0.25     # Distance scale l0 = scale * r_shore
  reconnection_prob_base: 0.55   # c0 reconnection factor
  anastomosis_prob_base: 0.0     # P0 inland anastomosis factor
  max_in_degree: 2
  max_out_degree: 3

hydraulics:
  murray_p: 1.70                 # Murray's Law exponent
  width_beta: 0.745              # Bifurcation width asymmetry
  manning_eta: 1.65              # Conductance width exponent
  length_gamma: 0.50             # Conductance length exponent
  apex_width: 250.0              # Inflow feeder channel width (m)
  width_noise: 0.30              # Width heterogeneity lognormal sigma
  tidal_flare_k: 1.40            # Tidal flaring exponent
  min_channel_width: 0.1         # Width cutoff (m)

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

extensions: {}                   # Custom plugin / hook parameters
```

## Distribution Types

The `distributions` section supports:
- **`uniform`**: Requires `low` and `high`.
- **`normal` / `gaussian`**: Requires `mean`, `std`, and optional `min`/`max` clamps.
- **`lognormal`**: Requires `mean`, `sigma`, and optional `min`/`max`.
- **`choice` / `categorical`**: Requires `values` list and optional `weights` list.
- **`fixed` / `constant`**: Requires `value`.

## Loading in Python

```python
import dgm

# Load default config
cfg = dgm.DGMConfig.load_default()

# Load custom config with parameter overrides
cfg = dgm.DGMConfig.from_yaml("my_run.yaml", N=75, R_rt=0.8)

# Sample active parameters
sampled = cfg.sample_distribution_parameters()
```

## Physical Assumptions & Literature Citations

For an in-depth table listing every parameter, its default value, underlying geomorphic assumptions, bounded distribution ranges, and formal academic literature citations justifying each parameter choice, see [Parameter Assumptions & Literature Justifications](parameter_justifications.md).

