"""
Core computational components of DeltaGraphModel (DGM).
"""

from .domain import PolarDomain
from .geometry_utils import segments_intersect, edge_intersects_graph, extend_leaf_to_shoreline
from .backbone import generate_polar_tree_backbone
from .loops import apply_loop_mechanism, generate_loops_potential_field, generate_loops_ner_optimality
from .hydraulics import apply_emergent_hydraulics, solve_markov_flux, solve_markov_hydraulics
from .channel_geometry import (
    point_to_segment_distance,
    compute_edge_clearance,
    apply_channel_proximity_constraints,
    generate_channel_centerlines,
)
from .pruning import (
    prune_and_simplify_network,
    prune_dead_end_spurs,
    simplify_degree_two_chains,
    refresh_node_types,
)

__all__ = [
    "PolarDomain",
    "segments_intersect",
    "edge_intersects_graph",
    "extend_leaf_to_shoreline",
    "generate_polar_tree_backbone",
    "apply_loop_mechanism",
    "generate_loops_potential_field",
    "generate_loops_ner_optimality",
    "prune_and_simplify_network",
    "prune_dead_end_spurs",
    "simplify_degree_two_chains",
    "refresh_node_types",
    "apply_emergent_hydraulics",
    "solve_markov_flux",
    "solve_markov_hydraulics",
    "point_to_segment_distance",
    "compute_edge_clearance",
    "apply_channel_proximity_constraints",
    "generate_channel_centerlines",
]
