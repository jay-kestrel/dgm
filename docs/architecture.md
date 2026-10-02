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
│ STEP 1: Polar Coordinate Domain & Adaptive Spreading (core.domain)     │
│ θ_open(R_rt) = θ_base + Δθ · R_rt | x = r cos(θ), y = r sin(θ)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Polar Tree Backbone Growth (core.backbone)                     │
│ Asymmetric radial progression | Variable branching order {1, 2, 3}     │
│ Jordan Curve Planarity Filter (0 edge crossings)                       │
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
│ STEP 4: Emergent Hydraulics & Murray's Law Conservation (core.hydraulics)
│ w_0^p = w_1^p + w_2^p  |  Estuarine mouth flaring: w * exp(k · r/r_shore)│
│ Open-channel conductance: C_e = w_e^η / L_e^γ                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Markov Flux Routing Solver & Metrics Battery                   │
│ Linear system: (I - Q^T) Φ_T = S_0,T                                   │
│ 20-metric geomorphic battery: α, β1/N, nER*, eBI, V_net, D_net         │
└────────────────────────────────────────────────────────────────────────┘
```

## Step 1: Polar Domain Geometry
The delta apex is positioned at origin $(r=0, \theta=0)$. Coastal shoreline boundary is located at radius $r = R_{\text{shore}}$ (default $10,000 \text{ m}$). Marine hydrodynamic energy causes sectoral spreading:
$$\theta_{\text{open}}(R_{\text{rt}}) = \theta_{\text{base}} + \Delta \theta \cdot R_{\text{rt}}$$

## Step 2: Planar Polar Tree Backbone
A directed tree is grown from apex outwards. Branching order $k \in \{1, 2, 3\}$ is sampled stochastically according to `branching_weights`. Radial steps and branching angles narrow as depth increases:
$$\theta_{\text{spread}} = \frac{\theta_{\text{open}} / 2.5}{\max(1.0, \lambda_{\text{decay}} \cdot (\text{level} + 1))}$$
Planarity is enforced on every candidate segment using the Jordan Curve orientation predicate, guaranteeing 0 edge crossings.

## Step 3: Emergent Loop Engines
### Mechanism A (Spatial Potential-Field Reconnection)
Evaluates candidate pairs $(u, v)$ via KDTree nearest-neighbor queries:
$$P_{\text{loop}}(u, v \mid R_{\text{rt}}) = \min(0.95, (P_0 + c_0 R_{\text{rt}}) \cdot e^{-d(u,v)/l_0} \cdot \max(0, \cos(\Delta \theta_{uv})))$$
Links are added under planarity and degree bounds ($k_{\text{in}} \le 2, k_{\text{out}} \le 3$).

### Mechanism B (Thermodynamic nER* Optimality)
Constructs a candidate network and optimizes link configuration using Simulated Annealing to maximize Non-Local Entropy Rate ($nER^*$) under a channel count budget.

## Step 4: Hydraulics & Murray's Law Conservation
Channel widths are solved using an iterative topological relaxation solver conserving Murray's Law:
$$w_0^p = w_1^p + w_2^p + \dots$$
where exponent $p \in [1.2, 2.2]$ (default 1.70). Estuarine flaring scales widths toward the coast:
$$w_{\text{emergent}} = w_{\text{river}} \cdot ((1 - R_{\text{rt}}) + R_{\text{rt}} e^{k_{\text{flare}} (r/R_{\text{shore}} - 0.3)})$$
Conductance follows Manning scaling: $C_e = w_e^\eta / L_e^\gamma$.

## Step 5: Markov Flux Routing Solver
Fluxes are solved via the transient submatrix formulation (Tejedor et al., 2017):
$$(I - Q^T) \Phi_T = S_{0,T}$$
$$\Phi_A = S_{0,A} + W[T, A]^T \Phi_T$$
Normalized branching entropy rate:
$$nER^* = \frac{-\sum_i \Phi_i \sum_j W_{ij} \log_2 W_{ij}}{\sum_i \Phi_i \log_2 k_{\text{out}}(i)}$$
