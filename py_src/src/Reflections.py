import numpy as np
import math
from typing import Tuple

from PrimaryStructures import Ray
from Sampling import Sampler

def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return np.linalg.norm(v) + eps

def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / _safe_norm(v, eps)

def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([1, 0, 0]) if abs(n[1]) > 0.999 else np.array([0, 1, 0])
    t = np.cross(up, n)
    t = _unit(t)
    b = np.cross(n, t)
    return t, b

def calculate_incident_angle(
        normalAngle: float,
        incomingAngle: float
    ) -> float:
    """
    Calculate the angle of incidence based on the normal and incoming angles.

    Attributes:
        normalAngle (float): The angle of the surface normal in degrees.
        incomingAngle (float): The angle of the incoming ray in degrees.
    """
    return abs(incomingAngle - normalAngle)

def calculate_reflection_angle(
        normalAngle: float,
        incomingAngle: float
    ) -> float:
    """
    Calculate the outgoing angle of the reflected ray based on the law of reflection.

    Attributes:
        normalAngle (float): The angle of the surface normal in degrees.
        incomingAngle (float): The angle of the incoming ray in degrees.
    """
    incidentAngle = calculate_incident_angle(normalAngle, incomingAngle)

    if incomingAngle > normalAngle:
        outgoingAngle = normalAngle + incidentAngle
    else:
        outgoingAngle = normalAngle - incidentAngle

    return outgoingAngle

def calculate_reflection_vector(
        normal: np.ndarray,
        direction: np.ndarray,
        bias: float = 1e-8
    ) -> np.ndarray:
    """
    Compute reflected ray direction using law of reflection.
    
    Args:
        normal: Surface normal (will be normalized)
        ray_direction: Incoming ray direction (will be normalized)
    
    Returns:
        Reflected ray direction (normalized)
    """
    unit_normal = _unit(normal, bias)
    unit_direction = _unit(direction, bias)

    reflected_direction = unit_direction - 2 * np.dot(unit_direction, unit_normal) * unit_normal
    return _unit(reflected_direction, bias)

def calculate_surface_reflection_ray(
        normal: np.ndarray,
        direction: np.ndarray,
        origin: np.ndarray,
        roughness: float,
        sampler: Sampler,
        bias: float = 1e-8
    ) -> Tuple[Ray, float]:
    """
        Generates a reflection ray using GGX Importance Sampling.
        Returns: (Ray, PDF)
    """
    # 1. Get Random Samples (u, v)
    # These determine "where" on the roughness hemisphere we pick a direction
    u, v = sampler.next_2d()

    # 2. Importance Sampling (GGX)
    # We map the random (u, v) to a 3D direction based on Roughness (alpha)
    # The rougher the surface, the wider the spread of possible directions.
    alpha = roughness * roughness

    phi = 2.0 * np.pi * u
    cos_theta = math.sqrt((1.0 - v) / (1.0 + (alpha*alpha - 1.0) * v))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    H_tangent = np.array([
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta
    ])

    tangent, bitangent = _orthonormal_basis(normal)

    H_world = (tangent * H_tangent[0]) + (bitangent * H_tangent[1]) + (normal * H_tangent[2])
    H_world = _unit(H_world)

    view_dir = -_unit(direction)

    dot_v_h = np.dot(view_dir, H_world)
    reflection_dir = (2.0 * dot_v_h * H_world) - view_dir
    reflection_dir = _unit(reflection_dir)

    # 5. Create the new Ray
    # Offset origin to prevent acne
    new_origin = origin + (normal * bias) # or hit_point + bias
    
    # Calculate Probability Density Function (PDF)
    # This is needed for the color math (throughput) to balance correctly.
    # (Simplified for demonstration)
    pdf = (2.0 * dot_v_h) / ((cos_theta * alpha * alpha) + bias) # Approximation

    new_ray = Ray(origin=new_origin, orientation=reflection_dir, name="reflection")
    
    return new_ray, pdf
"""
Reflection module: Provides functions to calculate reflection angles and reflected ray directions based on the law of reflection.
"""