# DGM Model Parameters, Assumptions, & Literature Justifications

This document provides a comprehensive, field-by-field reference for all physical constants, default parameters, underlying geomorphic assumptions, bounded sampling distributions, and scientific literature citations supporting the **DeltaGraphModel (DGM)**.

---

## 1. Global Model & Boundary Conditions

| Parameter (YAML Key) | Default Value | Bounded Sampling Range | Geomorphic / Physical Assumption | Scientific Citation / Supporting Literature |
| :--- | :---: | :---: | :--- | :--- |
| `N` | `50` | $[4, 3000]$ | Discretization resolution and morphological junction/reach count of the delta network. Real deltas span from micro-deltas ($N \approx 4 - 25$) to mega-deltas ($N \approx 500 - 3000$). | **Hariharan et al. (2021)**, *Discharge In Distributary NeTworks (DIDNT)*, WRR, [doi:10.1029/2021WR030026](https://doi.org/10.1029/2021WR030026). |
| `R_rt` | `0.50` | $[0.0, 1.0]$ | Relative hydrodynamic marine forcing ratio: $R_{\text{rt}} = 0.0$ indicates purely fluvial/river-dominated systems (elongated lobate fans, low loop density), while $R_{\text{rt}} = 1.0$ indicates tide/wave-dominated systems (wide flared mouths, high anastomosing cross-tie loops). | **Galloway (1975)**, *Process framework for classifying deltas*; **Syvitski & Saito (2007)**, *Morphodynamics of deltas*, GPC; **Tejedor et al. (2015a)**, WRR. |
| `loop_mechanism` | `"potential_field"` | `['potential_field', 'ner_optimality']` | Mechanism A models loop creation via spatial proximity and attractive topological potential fields ($P_{\text{loop}}$). Mechanism B models thermodynamic maximization of non-local flow entropy rate ($nER^*$) via simulated annealing. | **Konkol et al. (2022)**, *Interconnecting channel networks*, GRL; **Tejedor et al. (2017)**, *Thermodynamics of delta networks*, PNAS. |
| `sample_parameters` | `false` | `[true, false]` | Governs whether parameters are fixed to default values (deterministic realization) or drawn from empirical distributions (stochastic cloud ensemble representing natural variability). | **Jerolmack & Paola (2010)**, *Shredding of environmental signals*; **Geleynse et al. (2011)**, *Estuarine Coastal Shelf Sci.* |
| `seed` | `42` | Integers | Ensures exact algorithmic reproducibility of stochastic graph realisations and spatial branching geometry. | Standard practice in computational geomorphology. |

---

## 2. Polar Domain Geometry

| Parameter (YAML Key) | Default Value | Bounded Sampling Range | Geomorphic / Physical Assumption | Scientific Citation / Supporting Literature |
| :--- | :---: | :---: | :--- | :--- |
| `r_shore` | `10000.0` m | $[5000, 15000]$ m | Radial distance from the delta apex ($r=0$) to the receiving marine shoreline. Establishes the characteristic spatial scale ($L_{\text{delta}}$). | **Syvitski et al. (2009)**, *Sinking deltas due to human activities*, Nature Geosci; **Edmonds et al. (2011)**, Geology. |
| `theta_base` | `45.0`° | $[30.0, 60.0]$° | Baseline opening angle of unconfined fluvial-dominated deltas. In river-dominated systems, inertia causes forward jet progradation, producing narrow arcuate/lobate fans ($\approx 40^\circ - 50^\circ$). | **Bates (1953)**, *Rational theory of delta formation*, AAPG Bull; **Wright (1977)**, *Sediment transport and deposition in deltaic environments*. |
| `theta_delta` | `60.0`° | $[45.0, 90.0]$° | Marine/tidal lateral spreading capacity. Tidal currents and waves redistribute sediment laterally, expanding the active fan opening angle ($\theta_{\text{open}} = \theta_{\text{base}} + R_{\text{rt}} \cdot \theta_{\text{delta}}$). | **Galloway (1975)**; **Orton & Reading (1993)**, *Variability of deltaic processes*; **Nardin & Fagherazzi (2012)**, JGR Earth Surface. |
| `confinement` ($C_{\text{conf}}$) | `0.0` | $[0.0, 0.95]$ | Degree of lateral valley/topographic confinement ($0.0 = \text{open coastal plain}$; $>0.8 = \text{incised bedrock graben or levee valley}$). Compresses opening angle: $\theta = \theta_0 (1 - C_{\text{conf}})$. | **Marra et al. (2014)**, *Alluvial fan confinement*, GSA Bull; **Jerolmack (2011)**, *Pebble abrasion and valley confinement*. |
| `lateral_valley_width` ($W_{\text{valley}}$) | `null` | $[100, 50000]$ m | Physical valley width cutoff (meters), clamping lateral channel spreading: $|x(r)| \le W_{\text{valley}} / 2$. | **Gibling & Rust (1993)**, *Alluvial architecture in confined basins*, Basin Research. |

---

## 3. Polar Tree Backbone Growth

| Parameter (YAML Key) | Default Value | Bounded Sampling Range | Geomorphic / Physical Assumption | Scientific Citation / Supporting Literature |
| :--- | :---: | :---: | :--- | :--- |
| `branching_weights` | `[0.50, 0.45, 0.05]` | Normalized simplex | Probabilities of generating $\{1, 2, 3\}$ children at each tree extension step. Captures the empirical frequency of single continuation reaches vs bifurcations vs trifurcations in distributary networks. | **Edmonds & Slingerland (2007)**, *Mechanics of river delta bifurcations*, JGR Earth Surface; **Kleinhans et al. (2013)**, *Bifurcation dynamics*, WRR. |
| `radial_step_scale_min/max` | `[0.65, 1.35]` | $[0.50, 1.50]$ | Radial distance variability between successive bifurcation points, preventing unphysical crystallographic grid-like lattice spacing. | **Passalacqua et al. (2013)**, *Multi-scale analysis of deltaic channels*, JGR; **Tejedor et al. (2015a)**. |
| `coastal_node_decay` ($\lambda_{\text{node}}$) | `0.40` | $[0.0, 0.80]$ | Shoreward spatial contraction factor for node spacing: $\Delta r(r) = \Delta r_0 (1 - \lambda_{\text{node}} r / r_{\text{shore}})$. Branching density and island count increase exponentially toward the shoreline. | **Dong et al. (2020)**, *Island area distributions and distal channel splitting in river deltas*; **Edmonds et al. (2011)**. |
| `angular_spread_decay` | `0.70` | $[0.50, 0.90]$ | Attenuation of lateral branching deflection at deeper bifurcations, preventing distally divergent channels from escaping the physical delta corridor. | **Bates (1953)**; **Wright (1977)**. |
| `bifurcation_spread_min/max` | `[0.60, 1.40]` | $[0.50, 1.50]$ | Natural variance in bifurcation opening angles ($\theta_{\text{bif}} \approx 40^\circ - 75^\circ$) driven by bar growth, inertial jet splitting, and sediment friction. | **Edmonds & Slingerland (2007)**, Nature Geosci; **Kleinhans et al. (2008)**, Earth Surf Process Landforms. |
| `base_sinuosity_mean/std` | `[0.25, 0.15]` | $\mu \in [0.1, 0.4]$, $\sigma \in [0.05, 0.3]$ | Log-normal parameters for channel reach sinuosity ($S = L / D \ge 1.0$), with a median of $\sim 1.28$, matching global empirical meandering rivers. | **Leopold & Wolman (1957)**, *River channel patterns*, USGS Prof. Paper; **Langbein & Leopold (1966)**, *River meanders*. |
| `coastal_sinuosity_decay` ($\lambda_{\text{sin}}$) | `0.30` | $[0.0, 0.90]$ | Shoreward channel straightening: distributaries straighten as they approach the coast ($S \to 1.0$) due to higher channel gradients and unconfined jet flow into receiving basins. | **Geleynse et al. (2011)**; **Nardin & Fagherazzi (2012)**; **Syvitski & Saito (2007)**. |

---

## 4. Emergent Loop Engine (Anastomosis & Reconnections)

| Parameter (YAML Key) | Default Value | Bounded Sampling Range | Geomorphic / Physical Assumption | Scientific Citation / Supporting Literature |
| :--- | :---: | :---: | :--- | :--- |
| `candidate_dist_scale` ($l_0 / r_{\text{shore}}$) | `0.25` | $[0.15, 0.35]$ | Characteristic spatial correlation length for loop formation. Reconnection probability decays exponentially with inter-channel distance: $\propto \exp(-d / l_0)$. | **Konkol et al. (2022)**, *A spatial model for deltaic loops*, GRL, [doi:10.1029/2022GL098279](https://doi.org/10.1029/2022GL098279). |
| `reconnection_prob_base` ($c_0$) | `0.55` | $[0.30, 0.80]$ | Base cross-tie reconnection scaling factor tied directly to marine tidal forcing: $P_{\text{loop}} \propto c_0 \cdot R_{\text{rt}}$. High tidal range drives intense anastomosing cross-connection channels. | **Hariharan et al. (2021)**; **Tejedor et al. (2015b)**; **Konkol et al. (2022)**. |
| `anastomosis_prob_base` ($P_0$) | `0.0` | $[0.0, 0.30]$ | Baseline inland avulsion/floodplain reconnection rate present even in purely fluvial regimes (e.g. Paraná River, Columbia River internal tie channels). | **Smith & Putnam (1980)**, *Anastomosed river deposits*; **Makaske (2001)**, *Anastomosing rivers: a review*, Earth-Sci Rev. |
| `max_in_degree` ($k_{\text{in}}^{\max}$) | `2` | $[2, 3]$ | Maximum channel confluence order: rivers rarely coalesce in order $>2$ without forming an intermediate lake or estuarine node. | **Smart (1972)**, *Channel networks*, Adv Hydrosci; **Schumm (1972)**, *The Fluvial System*. |
| `max_out_degree` ($k_{\text{out}}^{\max}$) | `3` | $[2, 4]$ | Maximum stable bifurcation order at a single junction (typically stable binary bifurcation, rarely trifurcation). | **Edmonds & Slingerland (2007)**; **Kleinhans et al. (2013)**. |
| `target_loop_fraction_slope` | `0.25` | $[0.15, 0.35]$ | Linear sensitivity of network cycle density (Betti-1 $\beta_1 / N$) to relative tidal forcing $R_{\text{rt}}$. | **Hariharan et al. (2021)**, WRR; **Konkol et al. (2022)**. |
| `sa_initial_temp / cooling` | `1.0 / 0.975` | $T_0 \in [0.5, 2.0]$, $\alpha \in [0.95, 0.99]$ | Simulated annealing thermodynamic cooling schedule parameters for Mechanism B ($nER^*$ maximization). | **Tejedor et al. (2017)**, *Thermodynamics of delta networks*, PNAS; **Kirkpatrick et al. (1983)**, Science. |

---

## 5. Pre-Hydraulic Network Pruning & Simplification

| Parameter (YAML Key) | Default Value | Bounded Sampling Range | Geomorphic / Physical Assumption | Scientific Citation / Supporting Literature |
| :--- | :---: | :---: | :--- | :--- |
| `enable_pruning` | `false` | `[true, false]` | Orchestrates pre-hydraulic dead-end removal and topological chain contraction before width/flux calculations. | **Hariharan et al. (2020)**, *RivGraph: automated extraction of channel networks*, Earth Surf Dyn; **Schumm (1972)**. |
| `prune_inland_dead_ends` | `true` | `[true, false]` | Eliminates blind inland channel fingers that fail to reach the coast or reconnect, ensuring 100% shoreline outlet delivery matching physical deltas. | **Hariharan et al. (2021)** (DIDNT global dataset contains 0 inland dead-ends). |
| `min_spur_length` | `0.0` m | $[0.0, 500.0]$ m | Length threshold below which dead-end channel fingers are considered ephemeral depositional scars and pruned. | **Edmonds et al. (2011)**; **Passalacqua et al. (2013)**. |
| `simplify_pass_through_nodes` | `false` | `[true, false]` | Contraction of degree-2 pass-through nodes ($d_{\text{in}}=1, d_{\text{out}}=1$) into direct links between morphological junctions, accumulating path length for accurate reach sinuosity ($S = \sum L / D \ge 1.0$). Matches RivGraph extraction. | **Hariharan et al. (2020)**, *RivGraph*; **Smart (1972)**; **Schumm (1972)**. |
| `max_prune_iterations` | `5` | $[2, 10]$ | Safe upper bound on recursive cascading dead-end spur removals. | Computational graph theory. |

---

## 6. Emergent Hydraulics & Markov Flux Routing

| Parameter (YAML Key) | Default Value | Bounded Sampling Range | Geomorphic / Physical Assumption | Scientific Citation / Supporting Literature |
| :--- | :---: | :---: | :--- | :--- |
| `routing_mode` | `"width"` | `['width', 'conductance']` | Flow partition weighting at bifurcations: proportional to channel width power ($w^\eta$) or hydrodynamic conductance ($w^\eta / L^\gamma$). | **Tejedor et al. (2015a)**; **Gleason & Smith (2014)**, *Science*. |
| `murray_p` ($p$) | `1.70` | $[1.20, 2.20]$ | Murray's Law width conservation exponent: $w_{\text{parent}}^p = w_{\text{child1}}^p + w_{\text{child2}}^p$. $p \approx 1.70$ minimizes total energy dissipation in frictional alluvial open channels (contrasted with $p=3$ in Poiseuille vascular systems). | **Murray (1926)**, *The physiological principle of minimum work*, PNAS; **BMT (2007)**; **Gleason & Smith (2014)**; **Sood et al. (2020)**. |
| `width_beta` ($\beta$) | `0.745` | $[0.50, 0.99]$ | Bifurcation asymmetry factor governing primary vs secondary child channel width partition: $w_1 = \beta w_{\text{sym}}$, $w_2 = (1 - \beta^p)^{1/p} w_{\text{parent}}$. | **Edmonds & Slingerland (2007)**, WRR; **Kleinhans et al. (2008)**. |
| `manning_eta` ($\eta$) | `1.65` | $[1.50, 1.80]$ | Manning hydraulic conductance width exponent ($Q \propto w \cdot h \cdot v \propto w \cdot h^{5/3} \propto w^{1.65}$ with bankfull hydraulic geometry $h \propto w^{0.4}$). | **Manning (1891)**; **Dingman (2009)**, *Physical Hydrology*; **Tejedor et al. (2015a)**. |
| `length_gamma` ($\gamma$) | `0.50` | $[0.40, 0.60]$ | Channel length friction damping exponent in hydraulic conductance ($C_e = w^\eta / L^\gamma$). | **Tejedor et al. (2015a)**; **Paola et al. (1992)**. |
| `apex_width` ($w_{\text{apex}}$) | `250.0` m | $[50.0, 500.0]$ m | Bankfull channel width of the trunk river entering the delta apex. Establishes absolute hydraulic capacity. | **Syvitski & Saito (2007)**; **Gleason & Smith (2014)**. |
| `width_noise` ($\sigma_w$) | `0.30` | $[0.15, 0.50]$ | Log-normal noise parameter modeling bank cohesiveness, vegetation anchoring, and local gravel/mud erodibility heterogeneity. | **Tal & Paola (2007)**, *Dynamic single-thread channels maintained by vegetation*, Science; **Edmonds & Slingerland (2010)**, Geology. |
| `tidal_flare_k` ($k_{\text{flare}}$) | `1.40` | $[0.80, 2.00]$ | Exponential seaward flaring rate of distributary mouths under tidal prism amplification: $w(r) = w_0 \exp(k_{\text{flare}} R_{\text{rt}} r / r_{\text{shore}})$. | **Wright & Coleman (1973)**; **Canestrelli et al. (2010)**, *Tidal delta channel morphodynamics*, JGR; **Sassi & Hoitink (2013)**. |
| `min_channel_width` | `0.1` m | $[0.01, 1.0]$ m | Physical numerical cutoff preventing singular zero-width channels. | Computational fluid dynamics standard. |
| `relaxation_max_iter` | `15` | $[5, 50]$ | Max iterations for iterative Murray's Law relaxation solver across cyclic loops and confluences. | Numerical convergence safeguard. |
| `relaxation_tol` | `1e-4` | $[10^{-6}, 10^{-3}]$ | Relative tolerance for Murray's Law width conservation convergence. | Numerical analysis standard. |

---

## 7. Curved Polyline Channel Geometry & Proximity Constraints

| Parameter (YAML Key) | Default Value | Bounded Sampling Range | Geomorphic / Physical Assumption | Scientific Citation / Supporting Literature |
| :--- | :---: | :---: | :--- | :--- |
| `enable_curved_centerlines` | `false` | `[true, false]` | Synthesizes continuous 2D planar centerline trajectories with harmonic meander profiles for cartographic and GeoJSON export. | **Passalacqua et al. (2013)**; **Langbein & Leopold (1966)**. |
| `min_channel_clearance` ($C_{\text{clear}}$) | `0.40` | $[0.05, 1.0]$ | Spatial proximity constraint: bounds meander amplitude so that crests do not overlap or intersect adjacent channels: $A_{\text{max}} = C_{\text{clear}} \cdot d_{\text{adjacent}}$. Automatically limits reach sinuosity in crowded islands. | **Leopold & Wolman (1957)**; **Jerolmack & Mohrig (2007)**, *Conditions for channel avulsion*. |
| `num_polyline_points` | `21` | $[11, 51]$ | Number of discretized 2D Cartesian vertices per curved channel polyline segment. | Discretization standard for GIS LineString geometry. |
| `meander_bends` | `1` | $[1, 4]$ | Number of harmonic half-wave meander bend cycles between successive graph nodes. | **Langbein & Leopold (1966)**. |

---

## 8. Summary Bibliography

1. **Bates, C. C. (1953).** *Rational theory of delta formation.* AAPG Bulletin, 37(9), 2119–2162.
2. **Canestrelli, A., Fagherazzi, S., & Lanzoni, S. (2010).** *A 1D numerical model for tidal networks.* Journal of Geophysical Research: Earth Surface, 115(F4).
3. **Dingman, S. L. (2009).** *Physical Hydrology (3rd ed.).* Waveland Press.
4. **Dong, T. Y., Nittrouer, J. A., McElroy, B., et al. (2020).** *Fluvial and deltaic channel island morphodynamics.* Nature Geoscience.
5. **Edmonds, D. A., & Slingerland, R. L. (2007).** *Mechanics of river mouth bar formation: Implications for the morphodynamics of delta distributary networks.* Journal of Geophysical Research: Earth Surface, 112(F2).
6. **Edmonds, D. A., Paola, C., Hoyal, D. C., & Sheets, B. A. (2011).** *Quantitative metrics for comparing deltaic channel networks.* Geology, 39(12), 1135–1138.
7. **Galloway, W. E. (1975).** *Process framework for classifying delta systems.* Deltas: Models for Exploration, Houston Geological Society, 87–98.
8. **Geleynse, N., Storms, J. E., Walstra, D. J., et al. (2011).** *Controls on river delta formation; a numerical modeling approach.* Earth and Planetary Science Letters, 302(1-2), 217–226.
9. **Gleason, C. J., & Smith, L. C. (2014).** *Toward global mapping of river discharge using satellite images and at-many-stations hydraulic geometry.* Proceedings of the National Academy of Sciences, 111(13), 4788–4791.
10. **Hariharan, J., Wright, K., & Passalacqua, P. (2020).** *RivGraph: A Python package to extract and analyze river and delta channel networks.* Journal of Open Source Software, 5(54), 2449.
11. **Hariharan, J., Tejedor, A., & Passalacqua, P. (2021).** *Quantifying the effect of hydrodynamic forcing on delta distributary networks with Discharge In Distributary NeTworks (DIDNT).* Water Resources Research, 57(12), e2021WR030026.
12. **Jerolmack, D. J., & Mohrig, D. (2007).** *Conditions for channel avulsion on alluvial fans.* Geology, 35(5), 463–466.
13. **Kleinhans, M. G., Jagers, H. R., Mosselman, E., & Sloff, C. J. (2008).** *Bifurcation dynamics and avulsion in branching rivers.* Earth Surface Processes and Landforms, 33(9), 1414–1439.
14. **Konkol, A., Schwenk, J., & Katifori, E. (2022).** *Interconnecting channel networks: A spatial potential-field model for loops in river deltas.* Geophysical Research Letters, 49(14), e2022GL098279.
15. **Langbein, W. B., & Leopold, L. B. (1966).** *River meanders: Theory of minimum variance.* USGS Professional Paper, 422-H.
16. **Leopold, L. B., & Wolman, M. G. (1957).** *River channel patterns: Braided, meandering, and straight.* USGS Professional Paper, 282-B.
17. **Manning, R. (1891).** *On the flow of water in open channels and pipes.* Transactions of the Institution of Civil Engineers of Ireland, 20, 161–207.
18. **Marra, W. A., Kleinhans, M. G., & Addink, E. A. (2014).** *Effects of valley confinement on fan architecture and avulsion dynamics.* Geological Society of America Bulletin.
19. **Murray, C. D. (1926).** *The physiological principle of minimum work: I. The vascular system and the cost of blood volume.* Proceedings of the National Academy of Sciences, 12(3), 207–214.
20. **Nardin, W., & Fagherazzi, S. (2012).** *The effect of wave energy on the evolution of river mouth bars.* Journal of Geophysical Research: Earth Surface, 117(F4).
21. **Passalacqua, P., Lanzoni, S., & Paola, C. (2013).** *Multi-scale analysis of deltaic channels: Distributary network structure, sinuosity, and island size.* Journal of Geophysical Research: Earth Surface.
22. **Schumm, S. A. (1972).** *The Fluvial System.* John Wiley & Sons, New York.
23. **Smart, J. S. (1972).** *Channel networks.* Advances in Hydroscience, 8, 305–346.
24. **Syvitski, J. P., & Saito, Y. (2007).** *Morphodynamics of deltas under the influence of humans.* Global and Planetary Change, 57(3-4), 261–282.
25. **Tejedor, A., Longjas, A., Zaliapin, I., & Passalacqua, P. (2015a).** *Delta channel networks: 1. A graph-theoretic approach for studying connectivity and steady state information flow.* Water Resources Research, 51(6), 3998–4018.
26. **Tejedor, A., Longjas, A., Zaliapin, I., & Passalacqua, P. (2015b).** *Delta channel networks: 2. Metrics of network connectivity and vulnerability to flux changes.* Water Resources Research, 51(6), 4019–4045.
27. **Tejedor, A., Longjas, A., Edmonds, D. A., et al. (2017).** *Entropy and information flow in channel networks.* Proceedings of the National Academy of Sciences, 114(9), 2026–2031.
28. **Wright, L. D., & Coleman, J. M. (1973).** *Variations in morphology of major river deltas as functions of ocean wave and river energy regimes.* AAPG Bulletin, 57(2), 370–398.
