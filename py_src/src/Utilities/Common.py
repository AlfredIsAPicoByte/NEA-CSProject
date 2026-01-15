from __future__ import annotations
import numpy as np
import math
from typing import Optional

def clamp(value, min_value: float|int = 0.0, max_value: float|int = 1.0):
    """
    Constrains a value to lie between min_value and max_value.
    Useful for ensuring colors stay within [0, 1] or geometry stays in bounds.
    """
    return max(min_value, min(value, max_value))

def lerp(a, b, t: float):
    """
    Linear Interpolation. Blends between value 'a' and 'b' based on factor 't'.
    t=0 returns a, t=1 returns b, t=0.5 returns the average.
    """
    return a + (b - a) * t

def safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    """
    Calculates the length (magnitude) of a vector with a tiny safety buffer.
    Prevents DivisionByZero errors when normalizing zero-length vectors.
    """
    return float(np.linalg.norm(v) + eps)

def unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Returns the normalized direction vector (length of 1.0) of input v.
    """
    return v / safe_norm(v, eps)

def safe_asin(value: float) -> Optional[float]:
    """
    Calculates the arc sine but handles floating point errors where value is slightly > 1.0.
    Returns None if the value is mathematically impossible (outside -1 to 1).
    """
    if value > 1.0 or value < -1.0:
        return None
    return math.asin(value)

def orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs a local coordinate system (Tangent, Bitangent) based on a Normal vector.
    Used for scattering rays relative to a surface normal.
    """
    up = np.array([1, 0, 0]) if abs(n[1]) > 0.999 else np.array([0, 1, 0])
    t = np.cross(up, n)
    t = unit(t)
    b = np.cross(n, t)
    return t, b

def attenuate_distance_coefficents(distance: float, a: float = 0.0, b: float = 0.0, c: float = 1.0) -> float:
    """
    Calculates light intensity drop-off using the standard graphics quadratic formula:
    1 / (Quadratic + Linear + Constant).
    """
    if c == 0 and a == 0 and b == 0:
        raise ValueError("Attenuation coefficients cannot all be zero")
    factor = 1.0 / (a * (distance ** 2) + b * distance + c)
    return factor

def attenuate_distance_max(distance: float, max_distance: float) -> float:
    """
    Calculates linear falloff that reaches exactly zero at 'max_distance'.
    Useful for lights with a strict radius.
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be greater than zero")
    
    factor = max(0.0, 1.0 - (distance / max_distance))
    return factor

def attenuate_inv_sqr_distance(distance: float, bias: float = 1e-6) -> float:
    """
    Calculates physically accurate light falloff (Inverse Square Law).
    Light energy drops off with the square of the distance.
    """
    if distance <= 0:
        raise ValueError("distance and or max_distance must be greater than zero")
    
    factor = 1 / ((distance ** 2) + bias)
    return factor

def attenuate_distance_exponential(distance: float, decay_rate: float = 1.0) -> float:
    """
    Calculates exponential falloff (Beer's Law).
    Used for fog, volumetric density, or light passing through colored glass.
    """
    factor = np.exp(-decay_rate * distance)
    return factor