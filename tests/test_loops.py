"""
Tests for loop engines (Mechanism A: potential field and Mechanism B: nER optimality).
"""

import numpy as np
import networkx as nx

from dgm.core.domain import PolarDomain
from dgm.core.backbone import generate_polar_tree_backbone
from dgm.core.loops import apply_loop_mechanism
from dgm.config import DGMConfig


def test_potential_field_loop_mechanism():
    cfg = DGMConfig(N=35, R_rt=0.8, loop_mechanism="potential_field", seed=42)
    domain = PolarDomain(R_rt=cfg.R_rt)
    params = cfg.get_active_parameters()
    rng = np.random.default_rng(42)

    G_tree = generate_polar_tree_backbone(cfg.N, cfg.R_rt, domain, params, rng)
    E_tree = G_tree.number_of_edges()

    G_looped = apply_loop_mechanism(
        G=G_tree,
        domain=domain,
        mechanism="potential_field",
        params=params,
        rng=rng,
        seed=42
    )

    # In a tide-dominated regime (R_rt=0.8), loops should emerge
    assert G_looped.number_of_edges() >= E_tree
    assert nx.is_weakly_connected(G_looped)

    # Degree constraints check
    for n in G_looped.nodes():
        assert G_looped.in_degree(n) <= 2
        assert G_looped.out_degree(n) <= 3


def test_ner_optimality_loop_mechanism():
    cfg = DGMConfig(N=30, R_rt=0.7, loop_mechanism="ner_optimality", seed=101)
    domain = PolarDomain(R_rt=cfg.R_rt)
    params = cfg.get_active_parameters()
    rng = np.random.default_rng(101)

    G_tree = generate_polar_tree_backbone(cfg.N, cfg.R_rt, domain, params, rng)
    G_looped = apply_loop_mechanism(
        G=G_tree,
        domain=domain,
        mechanism="ner_optimality",
        params=params,
        rng=rng,
        seed=101
    )

    assert isinstance(G_looped, nx.DiGraph)
    assert nx.is_weakly_connected(G_looped)
    for n in G_looped.nodes():
        assert G_looped.in_degree(n) <= 2
        assert G_looped.out_degree(n) <= 3
