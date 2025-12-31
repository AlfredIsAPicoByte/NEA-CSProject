import numpy as np
import math
from typing import Tuple

from CommonUtils import unit, orthonormal_basis
from PrimaryStructures import Ray
from Sampling import Sampler

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
    unit_normal = unit(normal, bias)
    unit_direction = unit(direction, bias)

    reflected_direction = unit_direction - 2 * np.dot(unit_direction, unit_normal) * unit_normal
    return unit(reflected_direction, bias)
"""
Reflection module: Provides functions to calculate reflection angles and reflected ray directions based on the law of reflection.
"""