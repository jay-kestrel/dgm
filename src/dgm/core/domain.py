"""
Polar coordinate domain geometry and coordinate transformation utilities for DGM.
"""

from typing import Tuple, Optional
import math


class PolarDomain:
    """
    Encapsulates the polar coordinate domain (r, theta) for delta plain growth.

    Coordinate Conventions:
    - Apex / river feeder is at origin: r = 0, theta = 0.
    - Shoreline boundary is at r = r_shore.
    - Centerline corresponds to theta = 0.
    - Positive theta angles rotate counter-clockwise.
    - Adaptive opening angle: theta_open = (theta_base + theta_delta * R_rt) * (1 - confinement).
    - Optional lateral valley width constraint: |y| <= lateral_valley_width / 2.
    """

    def __init__(
        self,
        r_shore: float = 10000.0,
        theta_base: float = 45.0,
        theta_delta: float = 60.0,
        R_rt: float = 0.5,
        confinement: float = 0.0,
        lateral_valley_width: Optional[float] = None
    ):
        self.r_shore = float(r_shore)
        self.theta_base = math.radians(theta_base)
        self.theta_delta = math.radians(theta_delta)
        self.R_rt = max(0.0, min(1.0, float(R_rt)))
        self.confinement = max(0.0, min(0.95, float(confinement)))
        self.lateral_valley_width = float(lateral_valley_width) if lateral_valley_width is not None else None

        # Base unconfined opening angle
        self.theta_open_unconfined = self.theta_base + self.theta_delta * self.R_rt

        # Effective Opening Angle: contracts as lateral valley confinement increases
        self.theta_open = self.theta_open_unconfined * (1.0 - self.confinement)
        self.half_open = self.theta_open / 2.0

        self.min_theta = -self.half_open
        self.max_theta = self.half_open

    def polar_to_cartesian(self, r: float, theta: float) -> Tuple[float, float]:
        """Converts polar coordinates (r, theta in radians) to Cartesian (x, y)."""
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return (x, y)

    def cartesian_to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """Converts Cartesian (x, y) to polar coordinates (r, theta in radians)."""
        r = math.hypot(x, y)
        theta = math.atan2(y, x)
        return (r, theta)

    def is_in_domain(self, r: float, theta: float) -> bool:
        """Checks if a point (r, theta) lies within the domain sectoral wedge, shoreline radius, and lateral valley bounds."""
        in_wedge = (0.0 <= r <= self.r_shore) and (self.min_theta <= theta <= self.max_theta)
        if not in_wedge:
            return False
        if self.lateral_valley_width is not None:
            _, y = self.polar_to_cartesian(r, theta)
            if abs(y) > (self.lateral_valley_width / 2.0):
                return False
        return True

    def clamp_theta(self, theta: float, r: Optional[float] = None) -> float:
        """Clamps an angle theta to valid domain bounds, respecting both sectoral opening and lateral valley corridor."""
        min_t, max_t = self.min_theta, self.max_theta
        if self.lateral_valley_width is not None and r is not None and r > 1e-3:
            max_sin = min(1.0, (self.lateral_valley_width / 2.0) / r)
            lat_limit = math.asin(max_sin)
            min_t = max(min_t, -lat_limit)
            max_t = min(max_t, lat_limit)
        return max(min_t, min(max_t, theta))

    def law_of_cosines_distance(self, r1: float, theta1: float, r2: float, theta2: float) -> float:
        """Computes Euclidean distance between two polar points using the Law of Cosines."""
        val = r1**2 + r2**2 - 2.0 * r1 * r2 * math.cos(theta1 - theta2)
        return math.sqrt(max(0.0, val))

    def __repr__(self) -> str:
        return (
            f"PolarDomain(r_shore={self.r_shore:.1f}m, "
            f"theta_open={math.degrees(self.theta_open):.1f}°, "
            f"confinement={self.confinement:.2f}, R_rt={self.R_rt:.2f})"
        )
