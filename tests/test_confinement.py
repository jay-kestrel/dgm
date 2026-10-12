"""
Unit tests for the confinement parameter and lateral valley width constraint in DGM.
"""

import math
import numpy as np
import pytest
import networkx as nx

from dgm.core.domain import PolarDomain
from dgm.core.geometry_utils import segments_intersect
from dgm.config import DGMConfig
import dgm


def test_polar_domain_confinement_opening_angle():
    """Verify that confinement factor monotonically contracts domain opening angle."""
    domain_unconfined = PolarDomain(R_rt=0.5, confinement=0.0)
    domain_half = PolarDomain(R_rt=0.5, confinement=0.5)
    domain_strong = PolarDomain(R_rt=0.5, confinement=0.8)

    # Opening angle must scale strictly as theta_open_unconfined * (1 - confinement)
    assert np.isclose(domain_half.theta_open, domain_unconfined.theta_open * 0.5)
    assert np.isclose(domain_strong.theta_open, domain_unconfined.theta_open * 0.2)
    assert domain_strong.theta_open < domain_half.theta_open < domain_unconfined.theta_open


def test_polar_domain_lateral_valley_width():
    """Verify lateral valley width bounds domain containment and angle clamping."""
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


def test_dgm_config_confinement_valid():
    """Verify valid confinement and lateral_valley_width assignments."""
    cfg = DGMConfig(confinement=0.75, lateral_valley_width=3000.0)
    assert cfg.confinement == 0.75
    assert cfg.lateral_valley_width == 3000.0


@pytest.mark.parametrize(
    "config_kwargs,match_pattern",
    [
        ({"confinement": -0.1}, "confinement parameter must be in"),
        ({"confinement": 1.0}, "confinement parameter must be in"),
        ({"lateral_valley_width": 0.0}, "lateral_valley_width must be > 0"),
        ({"lateral_valley_width": -100.0}, "lateral_valley_width must be > 0"),
    ],
)
def test_dgm_config_confinement_validation_errors(config_kwargs, match_pattern):
    """Verify invalid confinement bounds raise descriptive ValueError."""
    with pytest.raises(ValueError, match=match_pattern):
        DGMConfig(**config_kwargs)


def test_generate_delta_with_confinement_narrows_network():
    """Verify confined delta realization exhibits significantly narrower lateral span than unconfined baseline."""
    G_unconf = dgm.generate(N=40, R_rt=0.5, confinement=0.0, seed=42)
    G_conf = dgm.generate(N=40, R_rt=0.5, confinement=0.8, seed=42)

    assert nx.is_weakly_connected(G_unconf)
    assert nx.is_weakly_connected(G_conf)
    assert G_conf.graph['confinement'] == 0.8

    # Confined delta must have significantly narrower lateral span (max |y|)
    max_y_unconf = max(abs(G_unconf.nodes[n]['y']) for n in G_unconf.nodes())
    max_y_conf = max(abs(G_conf.nodes[n]['y']) for n in G_conf.nodes())

    assert max_y_conf < max_y_unconf * 0.40


def test_cli_confinement_argument(tmp_path):
    """Verify CLI accepts -C / --confinement flag."""
    from dgm.cli import main
    out_file = tmp_path / "confined_delta.graphml"

    main(["generate", "-N", "25", "-R", "0.4", "-C", "0.6", "-s", "42", "-o", str(out_file)])
    assert out_file.exists() and out_file.stat().st_size > 0


def test_cli_lateral_valley_width_argument(tmp_path):
    """Verify CLI accepts -W / --lateral-valley-width flag."""
    from dgm.cli import main
    out_file = tmp_path / "valley_delta.graphml"

    main(["generate", "-N", "25", "-R", "0.4", "-W", "2500.0", "-s", "42", "-o", str(out_file)])
    assert out_file.exists() and out_file.stat().st_size > 0


def test_polar_domain_boundary_checks():
    """Verify domain containment at apex and angular clamping across radial ranges."""
    domain = PolarDomain(r_shore=10000.0, confinement=0.5, lateral_valley_width=2000.0)

    # Apex boundary at r=0
    assert domain.is_in_domain(0.0, 0.0) is True
    clamped_apex = domain.clamp_theta(0.5, r=0.0)
    assert domain.min_theta <= clamped_apex <= domain.max_theta

    # Boundary floating-point inclusion
    for r in [500.0, 1000.0, 3000.0, 5000.0, 10000.0]:
        clamped = domain.clamp_theta(math.radians(45.0), r=r)
        assert domain.is_in_domain(r, clamped) is True

    # PolarDomain directly rejecting invalid lateral_valley_width
    with pytest.raises(ValueError, match="lateral_valley_width must be > 0"):
        PolarDomain(lateral_valley_width=-500.0)


def test_extreme_confinement_generation():
    """Verify generation at maximum allowed confinement boundary (0.95)."""
    G = dgm.generate(N=30, R_rt=0.5, confinement=0.95, seed=42)
    assert nx.is_weakly_connected(G)
    assert G.number_of_nodes() == 30
    assert G.graph['confinement'] == 0.95


def test_combined_confinement_and_valley_width():
    """Verify simultaneous confinement and lateral valley width corridor constraint."""
    G = dgm.generate(N=35, R_rt=0.5, confinement=0.5, lateral_valley_width=1500.0, seed=42)
    assert nx.is_weakly_connected(G)
    assert G.graph['confinement'] == 0.5
    assert G.graph['lateral_valley_width'] == 1500.0

    # Every node must satisfy |y| <= 750 + epsilon
    for n in G.nodes():
        assert abs(G.nodes[n]['y']) <= 750.0 + 1e-4, f"Node {n} y={G.nodes[n]['y']} exceeds corridor half-width 750"


@pytest.mark.parametrize(
    "conf,width",
    [
        (0.8, None),
        (0.0, 2000.0),
        (0.5, 1500.0),
    ],
)
def test_planarity_under_confinement_and_valley_corridor(conf, width):
    """Verify zero edge crossings under high confinement and narrow corridor geometries."""
    G = dgm.generate(N=25, R_rt=0.5, confinement=conf, lateral_valley_width=width, seed=42)
    edges = [((G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])) for u, v in G.edges()]
    crossings = 0
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                crossings += 1
    assert crossings == 0, f"Found {crossings} crossings for conf={conf}, width={width}"


def test_ensemble_with_confinement():
    """Verify ensemble generator attaches and preserves confinement settings across realizations."""
    ens = dgm.sample_ensemble(N=20, R_rt=0.5, confinement=0.7, lateral_valley_width=3000.0, num_realizations=3, base_seed=10)
    assert len(ens) == 3
    for G in ens:
        assert G.graph['confinement'] == 0.7
        assert G.graph['lateral_valley_width'] == 3000.0
        assert nx.is_weakly_connected(G)
