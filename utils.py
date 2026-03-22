"""
Small, pure-function utilities shared across modules.

All angles are in radians; all physical quantities in SI.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Vec2 = NDArray[np.float64]


def vec2(x: float = 0.0, y: float = 0.0) -> Vec2:
    """Create a 2-D vector as a NumPy array."""
    return np.array([x, y], dtype=np.float64)


def magnitude(v: Vec2) -> float:
    return float(np.linalg.norm(v))


def normalise(v: Vec2) -> Vec2:
    m = magnitude(v)
    if m < 1e-12:
        return vec2()
    return v / m


def angle_of(v: Vec2) -> float:
    """Angle of a 2-D vector measured CCW from +x axis (rad)."""
    return float(math.atan2(v[1], v[0]))


def direction_from_angle(theta: float) -> Vec2:
    """Unit vector pointing in direction *theta* (rad, CCW from +x)."""
    return vec2(math.cos(theta), math.sin(theta))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def rotate_vec(v: Vec2, angle: float) -> Vec2:
    """Rotate *v* by *angle* radians CCW."""
    c, s = math.cos(angle), math.sin(angle)
    return vec2(c * v[0] - s * v[1], s * v[0] + c * v[1])
