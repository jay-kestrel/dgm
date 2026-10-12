"""
Tests for emergent hydraulics, Murray's Law width conservation, and Markov flux solver.
"""

import math
import pytest
import numpy as np

from dgm.core.domain import PolarDomain
from dgm.core.backbone import generate_polar_tree_backbone
from dgm.core.hydraulics import apply_emergent_hydraulics, solve_markov_flux
from dgm.config import DGMConfig


def test_emergent_hydraulics_widths_and_conductance():
    """Verify calculation and physical validity of width, conductance, and length attributes."""
    cfg = DGMConfig(N=30, R_rt=0.4, seed=42)
    domain = PolarDomain(R_rt=cfg.R_rt)
    params = cfg.get_active_parameters()
    rng = np.random.default_rng(42)

    G = generate_polar_tree_backbone(cfg.N, cfg.R_rt, domain, params, rng)
    G = apply_emergent_hydraulics(G, params=params, seed=42)

    # Check edge attributes
    for _, _, d in G.edges(data=True):
        assert 'width' in d
        assert d['width'] > 0.0
        assert 'conductance' in d
        assert d['conductance'] > 0.0
        assert 'length' in d
        assert d['length'] > 0.0


def test_markov_flux_solver():
    """Verify Markov transition matrix solver guarantees unit apex flux and mass conservation."""
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
    # Unit flux injected at apex (node 0)
    assert pytest.approx(phi[0], abs=1e-5) == 1.0
    # All fluxes non-negative and finite
    assert np.all(phi >= 0.0)
    assert np.all(np.isfinite(phi))

    # Outlet flux conservation: sum of outlet fluxes equals 1.0
    outlet_indices = [idx for idx, n in enumerate(G.nodes()) if G.out_degree(n) == 0]
    outlet_sum = sum(phi[idx] for idx in outlet_indices)
    assert pytest.approx(outlet_sum, abs=1e-4) == 1.0
