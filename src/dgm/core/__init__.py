"""
Core computational components of DeltaGraphModel (DGM).
"""

from .backbone import generate_polar_tree_backbone
from .channel_geometry import (
    apply_channel_proximity_constraints,
    compute_edge_clearance,
    generate_channel_centerlines,
    point_to_segment_distance,
)
from .domain import PolarDomain
from .geometry_utils import (
    edge_intersects_graph,
    extend_leaf_to_shoreline,
    segments_intersect,
)
from .hydraulics import (
    apply_emergent_hydraulics,
    solve_markov_flux,
    solve_markov_hydraulics,
)
from .loops import (
    apply_loop_mechanism,
    generate_loops_ner_optimality,
    generate_loops_potential_field,
)
from .pruning import (
    prune_and_simplify_network,
    prune_dead_end_spurs,
    refresh_node_types,
    simplify_degree_two_chains,
)

__all__ = [
    "PolarDomain",
    "apply_channel_proximity_constraints",
    "apply_emergent_hydraulics",
    "apply_loop_mechanism",
    "compute_edge_clearance",
    "edge_intersects_graph",
    "extend_leaf_to_shoreline",
    "generate_channel_centerlines",
    "generate_loops_ner_optimality",
    "generate_loops_potential_field",
    "generate_polar_tree_backbone",
    "point_to_segment_distance",
    "prune_and_simplify_network",
    "prune_dead_end_spurs",
    "refresh_node_types",
    "segments_intersect",
    "simplify_degree_two_chains",
    "solve_markov_flux",
    "solve_markov_hydraulics",
]
