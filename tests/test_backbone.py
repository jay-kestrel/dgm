"""
Tests for backbone tree skeleton generation and planarity.
"""

import numpy as np
import networkx as nx

from dgm.core.domain import PolarDomain
from dgm.core.backbone import generate_polar_tree_backbone
from dgm.config import DGMConfig


def test_polar_tree_backbone_generation():
    cfg = DGMConfig(N=35, R_rt=0.3, seed=42)
    domain = PolarDomain(R_rt=cfg.R_rt)
    params = cfg.get_active_parameters()
    rng = np.random.default_rng(42)

    G = generate_polar_tree_backbone(
        N_target=cfg.N,
        R_rt=cfg.R_rt,
        domain=domain,
        params=params,
        rng=rng
    )

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() >= 20
    assert 0 in G.nodes()
    assert G.nodes[0]['node_type'] == 'apex'
    assert nx.is_weakly_connected(G)

    # In a directed tree from apex, all nodes except apex have in-degree 1
    for n in G.nodes():
        if n != 0:
            assert G.in_degree(n) == 1
