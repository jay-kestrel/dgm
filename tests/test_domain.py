"""
Tests for PolarDomain coordinate transformations and geometric metrics.
"""

import math
import pytest

from dgm.core.domain import PolarDomain


def test_polar_domain_opening_angle():
    """Verify calculation of opening angle across river-dominated and tide-dominated regimes."""
    domain_river = PolarDomain(R_rt=0.0, theta_base=45.0, theta_delta=60.0)
    domain_tide = PolarDomain(R_rt=1.0, theta_base=45.0, theta_delta=60.0)

    # 45 deg vs 105 deg
    assert math.isclose(domain_river.theta_open, math.radians(45.0), rel_tol=1e-5)
    assert math.isclose(domain_tide.theta_open, math.radians(105.0), rel_tol=1e-5)
    assert domain_tide.theta_open > domain_river.theta_open


@pytest.mark.parametrize(
    "r,theta",
    [
        (5000.0, 0.25),
        (1000.0, -0.30),
        (10000.0, 0.0),
        (0.0, 0.0),
    ],
)
def test_polar_cartesian_roundtrip(r, theta):
    """Verify bidirectional coordinate transformations between polar and cartesian frames."""
    domain = PolarDomain()
    x, y = domain.polar_to_cartesian(r, theta)
    r_back, theta_back = domain.cartesian_to_polar(x, y)

    assert math.isclose(r, r_back, abs_tol=1e-5)
    if r > 0.0:
        assert math.isclose(theta, theta_back, abs_tol=1e-5)


def test_law_of_cosines_distance():
    """Verify geodesic/law of cosines distance calculation between two polar points."""
    domain = PolarDomain()
    # Distance between (1000, 0) and (1000, pi/2) is sqrt(1000^2 + 1000^2) = 1414.21
    dist = domain.law_of_cosines_distance(1000.0, 0.0, 1000.0, math.pi / 2.0)
    assert math.isclose(dist, 1000.0 * math.sqrt(2), rel_tol=1e-5)

    # Zero distance between identical points
    dist_zero = domain.law_of_cosines_distance(500.0, 0.2, 500.0, 0.2)
    assert math.isclose(dist_zero, 0.0, abs_tol=1e-6)
