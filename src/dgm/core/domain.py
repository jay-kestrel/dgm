"""
Polar coordinate domain geometry and coordinate transformation utilities for DGM.
"""

from typing import Tuple, Optional
import math


class PolarDomain:
    """
    Encapsulates the polar coordinate domain (r, theta) for delta plain growth.

    Coordinate Conventions:
    - Apex / river feeder is at origin: r = 0 m, theta = 0 rad.
    - Shoreline boundary is at r = r_shore (meters).
    - Centerline corresponds to theta = 0 rad.
    - Positive theta angles rotate counter-clockwise (radians).
    - Adaptive opening angle: theta_open = (theta_base + theta_delta * R_rt) * (1 - confinement) [rad].
    - Optional lateral valley width constraint: |y| <= lateral_valley_width / 2 [meters].
    """

    def __init__(
        self,
        r_shore: float = 10000.0,
        theta_base: float = 65.0,
        theta_delta: float = 60.0,
        R_rt: float = 0.5,
        confinement: float = 0.0,
        lateral_valley_width: Optional[float] = None
    ) -> None:
        """
        Initializes the PolarDomain.

        Args:
            r_shore: Delta shoreline radius in meters (must be > 0).
            theta_base: Base opening angle at R_rt = 0 in degrees (> 0).
            theta_delta: Additional opening angle scaling across R_rt in degrees (>= 0).
            R_rt: River-to-tidal forcing ratio in [0.0, 1.0] (dimensionless).
            confinement: Lateral valley confinement factor in [0.0, 0.95] (dimensionless).
            lateral_valley_width: Optional physical corridor width in meters (must be > 0 if provided).
        """
        self.r_shore = float(r_shore)
        if self.r_shore <= 0.0:
            raise ValueError(f"r_shore must be > 0 (received {self.r_shore}).")

        self.theta_base = math.radians(max(1e-4, float(theta_base)))
        self.theta_delta = math.radians(max(0.0, float(theta_delta)))
        self.R_rt = max(0.0, min(1.0, float(R_rt)))
        self.confinement = max(0.0, min(0.95, float(confinement)))
        self.lateral_valley_width = float(lateral_valley_width) if lateral_valley_width is not None else None

        if self.lateral_valley_width is not None and self.lateral_valley_width <= 0.0:
            raise ValueError(f"lateral_valley_width must be > 0 (received {self.lateral_valley_width}).")

        # Base unconfined opening angle (radians)
        self.theta_open_unconfined = self.theta_base + self.theta_delta * self.R_rt

        # Effective Opening Angle: contracts as lateral valley confinement increases
        self.theta_open = max(1e-4, self.theta_open_unconfined * (1.0 - self.confinement))
        self.half_open = self.theta_open / 2.0

        self.min_theta = -self.half_open
        self.max_theta = self.half_open

    def polar_to_cartesian(self, r: float, theta: float) -> Tuple[float, float]:
        """
        Converts polar coordinates (r in meters, theta in radians) to Cartesian coordinates (x, y in meters).
        """
        r_safe = max(0.0, float(r))
        x = r_safe * math.cos(float(theta))
        y = r_safe * math.sin(float(theta))
        return (float(x), float(y))

    def cartesian_to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """
        Converts Cartesian coordinates (x, y in meters) to polar coordinates (r in meters, theta in radians).
        Apex origin (0, 0) maps to r = 0.0, theta = 0.0.
        """
        r = math.hypot(float(x), float(y))
        if r < 1e-12:
            return (0.0, 0.0)
        theta = math.atan2(float(y), float(x))
        return (float(r), float(theta))

    def is_in_domain(self, r: float, theta: float) -> bool:
        """
        Checks if a point (r in meters, theta in radians) lies within the domain sectoral wedge,
        shoreline radius, and lateral valley corridor.
        """
        r_val = float(r)
        theta_val = float(theta)
        in_wedge = (-1e-7 <= r_val <= self.r_shore + 1e-7) and (self.min_theta - 1e-7 <= theta_val <= self.max_theta + 1e-7)
        if not in_wedge:
            return False
        if self.lateral_valley_width is not None:
            _, y = self.polar_to_cartesian(r_val, theta_val)
            if abs(y) > (self.lateral_valley_width / 2.0) + 1e-6:
                return False
        return True

    def clamp_theta(self, theta: float, r: Optional[float] = None) -> float:
        """
        Clamps an angle theta (radians) to valid domain bounds, respecting both sectoral opening
        and lateral valley corridor limits.
        """
        min_t, max_t = self.min_theta, self.max_theta
        if self.lateral_valley_width is not None and r is not None:
            r_val = max(0.0, float(r))
            if r_val > 1e-3:
                max_sin = min(1.0, max(0.0, (self.lateral_valley_width / 2.0) / r_val))
                lat_limit = math.asin(max_sin)
                min_t = max(min_t, -lat_limit)
                max_t = min(max_t, lat_limit)
        return max(min_t, min(max_t, float(theta)))

    def law_of_cosines_distance(self, r1: float, theta1: float, r2: float, theta2: float) -> float:
        """
        Computes Euclidean distance in meters between two polar points (r1, theta1) and (r2, theta2)
        using the Law of Cosines with numerical underflow safeguarding.
        """
        r1_f = max(0.0, float(r1))
        r2_f = max(0.0, float(r2))
        val = r1_f**2 + r2_f**2 - 2.0 * r1_f * r2_f * math.cos(float(theta1) - float(theta2))
        return float(math.sqrt(max(0.0, val)))

    def __repr__(self) -> str:
        width_str = f", lateral_valley_width={self.lateral_valley_width:.1f}m" if self.lateral_valley_width is not None else ""
        return (
            f"PolarDomain(r_shore={self.r_shore:.1f}m, "
            f"theta_open={math.degrees(self.theta_open):.1f}°, "
            f"confinement={self.confinement:.2f}, R_rt={self.R_rt:.2f}{width_str})"
        )

