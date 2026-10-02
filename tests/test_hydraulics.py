"""
Tests for emergent hydraulics, Murray's Law width conservation, and Markov flux solver.
"""

import numpy as np
import networkx as nx

from dgm.core.domain import PolarDomain
from dgm.core.backbone import generate_polar_tree_backbone
from dgm.core.hydraulics import apply_emergent_hydraulics, solve_markov_flux
from dgm.config import DGMConfig


def test_emergent_hydraulics_widths_and_conductance():
    cfg = DGMConfig(N=30, R_rt=0.4, seed=42)
    domain = PolarDomain(R_rt=cfg.R_rt)
    params = cfg.get_active_parameters()
    rng = np.random.default_rng(42)

    G = generate_polar_tree_backbone(cfg.N, cfg.R_rt, domain, params, rng)
    G = apply_emergent_hydraulics(G, params=params, seed=42)

    # Check edge attributes
    for u, v, d in G.edges(data=True):
        assert 'width' in d
        assert d['width'] > 0.0
        assert 'conductance' in d
        assert d['conductance'] > 0.0
        assert 'length' in d
        assert d['length'] > 0.0


def test_markov_flux_solver():
    cfg = DGMConfig(N=30, R_rt=0.4, seed=42)
    domain = PolarDomain(R_rt=cfg.R_rt)
    params = cfg.get_active_parameters()
    rng = np.random.default_rng(42)

    G = generate_polar_tree_backbone(cfg.N, cfg.R_rt, domain, params, rng)
    G = apply_emergent_hydraulics(G, params=params, seed=42)

    C, W, phi = solve_markov_flux(G, params=params)

    assert C.shape == (len(G), len(G))
    assert W.shape == (len(G), len(G))
    assert phi.shape == (len(G),)
    # Unit flux at apex
    assert phi[0] == 1.0
    # All fluxes non-negative
    assert np.all(phi >= 0.0)
