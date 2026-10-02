"""
Unit tests for the confinement parameter and lateral valley width constraint in DGM.
"""

import math
import numpy as np
import pytest
import networkx as nx

from dgm.core.domain import PolarDomain
from dgm.config import DGMConfig
import dgm


def test_polar_domain_confinement_opening_angle():
    domain_unconfined = PolarDomain(R_rt=0.5, confinement=0.0)
    domain_half = PolarDomain(R_rt=0.5, confinement=0.5)
    domain_strong = PolarDomain(R_rt=0.5, confinement=0.8)

    assert domain_unconfined.confinement == 0.0
    assert domain_half.confinement == 0.5
    assert domain_strong.confinement == 0.8

    # Opening angle must scale strictly as theta_open_unconfined * (1 - confinement)
    assert np.isclose(domain_half.theta_open, domain_unconfined.theta_open * 0.5)
    assert np.isclose(domain_strong.theta_open, domain_unconfined.theta_open * 0.2)
    assert domain_strong.theta_open < domain_half.theta_open < domain_unconfined.theta_open


def test_polar_domain_lateral_valley_width():
    # Valley width of 2000m implies max |y| <= 1000m
    domain = PolarDomain(r_shore=10000.0, confinement=0.0, lateral_valley_width=2000.0)

    # Point along centerline (y=0) is in domain
    assert domain.is_in_domain(5000.0, 0.0) is True

    # Point at r=5000 with y > 1000: theta = pi/4 gives y = 5000 * sin(pi/4) = 3535m > 1000m
    assert domain.is_in_domain(5000.0, math.pi / 4.0) is False

    # clamp_theta should bound the angle within arcsin(1000 / r)
    clamped = domain.clamp_theta(math.radians(30.0), r=5000.0)
    max_allowed = math.asin(1000.0 / 5000.0)
    assert clamped <= max_allowed + 1e-6


def test_dgm_config_confinement_validation():
    # Valid configurations
    cfg1 = DGMConfig(confinement=0.0)
    assert cfg1.confinement == 0.0

    cfg2 = DGMConfig(confinement=0.75, lateral_valley_width=3000.0)
    assert cfg2.confinement == 0.75
    assert cfg2.lateral_valley_width == 3000.0

    # Invalid confinement bounds
    with pytest.raises(ValueError, match="confinement parameter must be in"):
        DGMConfig(confinement=-0.1)

    with pytest.raises(ValueError, match="confinement parameter must be in"):
        DGMConfig(confinement=1.0)

    # Invalid lateral_valley_width
    with pytest.raises(ValueError, match="lateral_valley_width must be > 0"):
        DGMConfig(lateral_valley_width=0.0)


def test_generate_delta_with_confinement_narrows_network():
    # Unconfined baseline vs strongly confined
    G_unconf = dgm.generate(N=40, R_rt=0.5, confinement=0.0, seed=42)
    G_conf = dgm.generate(N=40, R_rt=0.5, confinement=0.8, seed=42)

    assert isinstance(G_unconf, nx.DiGraph)
    assert isinstance(G_conf, nx.DiGraph)
    assert nx.is_weakly_connected(G_unconf)
    assert nx.is_weakly_connected(G_conf)

    # Check metadata attachment
    assert G_conf.graph['confinement'] == 0.8

    # Confined delta must have significantly narrower lateral span (max |y|)
    max_y_unconf = max(abs(G_unconf.nodes[n]['y']) for n in G_unconf.nodes())
    max_y_conf = max(abs(G_conf.nodes[n]['y']) for n in G_conf.nodes())

    assert max_y_conf < max_y_unconf * 0.40


def test_cli_confinement_argument(tmp_path):
    from dgm.cli import main
    out_file = tmp_path / "confined_delta.graphml"

    main(["generate", "-N", "25", "-R", "0.4", "-C", "0.6", "-s", "42", "-o", str(out_file)])
    assert out_file.exists()
    assert out_file.stat().st_size > 0
