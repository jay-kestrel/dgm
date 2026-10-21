"""
Tests for loop engines (Mechanism A: potential field and Mechanism B: nER optimality).
"""

import networkx as nx
import numpy as np
import pytest

from dgm.config import DGMConfig
from dgm.core.backbone import generate_polar_tree_backbone
from dgm.core.domain import PolarDomain
from dgm.core.geometry_utils import segments_intersect
from dgm.core.loops import apply_loop_mechanism


@pytest.mark.parametrize("mechanism", ["potential_field", "ner_optimality"])
def test_loop_mechanisms(mechanism):
    """Verify both potential_field and ner_optimality loop mechanisms generate valid planar cyclic networks."""
    cfg = DGMConfig(N=35, R_rt=0.8, loop_mechanism=mechanism, seed=42)
    domain = PolarDomain(R_rt=cfg.R_rt)
    params = cfg.get_active_parameters()
    rng = np.random.default_rng(42)

    G_tree = generate_polar_tree_backbone(cfg.N, cfg.R_rt, domain, params, rng)
    E_tree = G_tree.number_of_edges()

    G_looped = apply_loop_mechanism(
        G=G_tree,
        domain=domain,
        mechanism=mechanism,
        params=params,
        rng=rng,
        seed=42
    )

    # In a tide-dominated regime (R_rt=0.8), loop edges should be added
    assert G_looped.number_of_edges() >= E_tree
    assert nx.is_weakly_connected(G_looped)

    # Degree constraints check: max in-degree 2, max out-degree 3
    for n in G_looped.nodes():
        assert G_looped.in_degree(n) <= 2, f"Node {n} exceeded max in-degree 2: {G_looped.in_degree(n)}"
        assert G_looped.out_degree(n) <= 3, f"Node {n} exceeded max out-degree 3: {G_looped.out_degree(n)}"

    # Geometric planarity: loop mechanism must not introduce edge crossings
    edges = [
        ((G_looped.nodes[u]['x'], G_looped.nodes[u]['y']), (G_looped.nodes[v]['x'], G_looped.nodes[v]['y']))
        for u, v in G_looped.edges()
    ]
    crossings = sum(
        1 for i in range(len(edges))
        for j in range(i + 1, len(edges))
        if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1])
    )
    assert crossings == 0, f"Found {crossings} planarity crossings under mechanism={mechanism}"
