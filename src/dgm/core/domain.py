"""
Polar coordinate domain geometry and coordinate transformation utilities for DGM.
"""

from typing import Tuple
import math


class PolarDomain:
    """
    Encapsulates the polar coordinate domain (r, theta) for delta plain growth.

    Coordinate Conventions:
    - Apex / river feeder is at origin: r = 0, theta = 0.
    - Shoreline boundary is at r = r_shore.
    - Centerline corresponds to theta = 0.
    - Positive theta angles rotate counter-clockwise.
    - Adaptive opening angle: theta_open = theta_base + theta_delta * R_rt.
    """

    def __init__(
        self,
        r_shore: float = 10000.0,
        theta_base: float = 45.0,
        theta_delta: float = 60.0,
        R_rt: float = 0.5
    ):
        self.r_shore = float(r_shore)
        self.theta_base = math.radians(theta_base)
        self.theta_delta = math.radians(theta_delta)
        self.R_rt = max(0.0, min(1.0, float(R_rt)))

        # Adaptive Opening Angle: expands with tidal/marine reworking energy
        self.theta_open = self.theta_base + self.theta_delta * self.R_rt
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
        """Checks if a point (r, theta) lies within the domain sectoral wedge and shoreline radius."""
        return (0.0 <= r <= self.r_shore) and (self.min_theta <= theta <= self.max_theta)

    def law_of_cosines_distance(self, r1: float, theta1: float, r2: float, theta2: float) -> float:
        """Computes Euclidean distance between two polar points using the Law of Cosines."""
        val = r1**2 + r2**2 - 2.0 * r1 * r2 * math.cos(theta1 - theta2)
        return math.sqrt(max(0.0, val))

    def __repr__(self) -> str:
        return (
            f"PolarDomain(r_shore={self.r_shore:.1f}m, "
            f"theta_open={math.degrees(self.theta_open):.1f}°, R_rt={self.R_rt:.2f})"
        )
