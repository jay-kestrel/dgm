# DGM Mathematical Formulation & Architecture

The **DeltaGraphModel (DGM)** generative pipeline executes five sequential computational stages:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DGM GENERATIVE PIPELINE                         │
├────────────────────────────────────────────────────────────────────────┤
│ Inputs: N (Nodes), R_rt (River-to-Tidal Ratio ∈ [0, 1]), YAML Config   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Polar Domain Geometry & Lateral Valley Confinement             │
│ θ_open = (θ_base + Δθ · R_rt) · (1 - C_conf)  |  |x(r)| ≤ W_valley / 2   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Polar Tree Backbone Growth & Shoreward Spatial Scaling         │
│ Δr(r) = Δr_0 · (1 - λ_node · r/r_shore) | S(r) = 1 + (S_0 - 1)·(1-λ_sin)│
│ Jordan Curve Planarity Filter (0 edge crossings via 2D AABB)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Emergent Loop Engine (core.loops)                              │
│ ├─ Mechanism A: Spatial Potential-Field Reconnection (KDTree P_loop)   │
│ └─ Mechanism B: Thermodynamic nER* Annealing under Volume Limits       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Pre-Hydraulic Pruning & Reach Simplification (core.pruning)    │
│ ├─ Cascading inland dead-end spur removal (100% shoreline delivery)   │
│ └─ Degree-2 pass-through chain contraction (RivGraph consistency)      │
│    S_reach = (Σ L_i) / D_straight ≥ 1.0 | Polyline coordinate merging │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Emergent Hydraulics & Murray's Law (core.hydraulics)           │
│ w_0^p = w_1^p + w_2^p  |  Estuarine mouth flaring: w * exp(k · r/r_shore)│
│ Open-channel conductance: C_e = w_e^η / L_e^γ                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Curved Polyline Channel Geometry (core.channel_geometry)       │
│ Proximity-bounded amplitude: A_max = C_clear · d_clearance             │
│ Sinuous centerline coordinates & GeoJSON LineString synthesis         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Markov Flux Routing Solver & Metrics Battery                   │
│ Linear system: (I - Q^T) Φ_T = S_0,T                                   │
│ 20-metric geomorphic battery: α, β1/N, nER*, eBI, V_net, D_net         │
└────────────────────────────────────────────────────────────────────────┘
```

## Step 1: Polar Domain Geometry & Valley Confinement
The delta apex is positioned at origin $(r=0, \theta=0)$. Coastal shoreline boundary is located at radius $r = R_{\text{shore}}$ (default $10,000 \text{ m}$). Marine hydrodynamic energy expands active sectoral fan spreading, while valley confinement narrows it:
$$\theta_{\text{open}}(R_{\text{rt}}, C_{\text{conf}}) = (\theta_{\text{base}} + \Delta \theta \cdot R_{\text{rt}}) \cdot (1 - C_{\text{conf}})$$
Lateral valley corridor boundaries clamp Cartesian coordinate excursions: $|x(r)| \le W_{\text{valley}} / 2$.

## Step 2: Polar Tree Backbone Growth & Shoreward Spatial Scaling
A directed branching tree is grown outwards from apex. Branching order $k \in \{1, 2, 3\}$ is sampled stochastically according to `branching_weights`.
- **Shoreward Node Contraction**: Inter-junction distance contracts as channels approach the sea:
  $$\Delta r(r) = \Delta r_0 \cdot \max(0.2, 1 - \lambda_{\text{node}} \cdot r / r_{\text{shore}})$$
- **Coastal Sinuosity Straightening**: Distributaries straighten in the seaward reach:
  $$S(r) = \max(1.0, 1.0 + (S_0 - 1.0) \cdot \max(0, 1 - \lambda_{\text{sin}} \cdot r_{\text{mid}} / r_{\text{shore}}))$$
Planarity is strictly preserved using 2D Axis-Aligned Bounding Box (AABB) fast rejection combined with the Jordan Curve orientation predicate, ensuring 0 unphysical edge crossings.

## Step 3: Emergent Loop Engines
### Mechanism A (Spatial Potential-Field Reconnection)
Evaluates candidate pairs $(u, v)$ via KDTree nearest-neighbor queries:
$$P_{\text{loop}}(u, v \mid R_{\text{rt}}) = \min(0.95, (P_0 + c_0 R_{\text{rt}}) \cdot e^{-d(u,v)/l_0} \cdot \max(0, \cos(\Delta \theta_{uv})))$$
Links are added under planarity and degree bounds ($k_{\text{in}} \le 2, k_{\text{out}} \le 3$).

### Mechanism B (Thermodynamic nER* Optimality)
Constructs a candidate network and optimizes link configuration using Simulated Annealing to maximize Non-Local Entropy Rate ($nER^*$) under a channel count budget.

## Step 4: Pre-Hydraulic Network Pruning & Simplification
- **Inland Dead-End Pruning**: Cascades through leaf nodes terminating inland without reaching the shoreline ($r < r_{\text{shore}} (1 - \epsilon)$), guaranteeing 100% shoreline outlet delivery matching DIDNT observations.
- **Degree-2 Reach Simplification**: Contracts non-branching pass-through chains ($d_{\text{in}}=1, d_{\text{out}}=1$) into direct links between morphological junctions, accumulating path length into a consistent reach sinuosity ($S_{\text{reach}} = \frac{\sum L_i}{D_{\text{straight}}} \ge 1.0$) and concatenating polyline coordinates, matching RivGraph topology.

## Step 5: Emergent Hydraulics & Murray's Law Conservation
Channel widths are solved using an iterative topological relaxation solver conserving Murray's Law:
$$w_0^p = w_1^p + w_2^p + \dots$$
where exponent $p \in [1.2, 2.2]$ (default 1.70). Estuarine flaring scales widths toward the coast:
$$w_{\text{emergent}} = w_{\text{river}} \cdot ((1 - R_{\text{rt}}) + R_{\text{rt}} e^{k_{\text{flare}} (r/R_{\text{shore}} - 0.3)})$$
Conductance follows Manning scaling: $C_e = w_e^\eta / L_e^\gamma$.

## Step 6: Curved Polyline Channel Geometry & Proximity Constraints
Meander amplitude is bounded by proximity to neighboring channels:
$$A_{\text{max}} = C_{\text{clear}} \cdot d_{\text{clearance}}$$
$$S_{\text{max}} = 1.0 + \left(\frac{\pi A_{\text{max}}}{L}\right)^2$$
Sinuous meander trajectories are deflected away from adjacent channels and rendered as multi-vertex polylines for cartographic and GeoJSON export.

## Step 7: Markov Flux Routing Solver
Fluxes are solved via the transient submatrix formulation (Tejedor et al., 2017):
$$(I - Q^T) \Phi_T = S_{0,T}$$
$$\Phi_A = S_{0,A} + W[T, A]^T \Phi_T$$
Normalized branching entropy rate:
$$nER^* = \frac{-\sum_i \Phi_i \sum_j W_{ij} \log_2 W_{ij}}{\sum_i \Phi_i \log_2 k_{\text{out}}(i)}$$
