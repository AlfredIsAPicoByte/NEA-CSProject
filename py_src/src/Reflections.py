from src.Basic import *
import numpy as np
import math

def calculate_reflection_angle(incidentAngle: float) -> float:
    """
    Calculate the reflection angle based on the law of reflection.
    The reflection angle is equal to the incident angle.

    Attributes:
        incidentincidentAngle (float): The angle of incidence in degrees.
    """
    return incidentAngle

def calculate_incident_angle(normalAngle: float, incomingAngle: float) -> float:
    """
    Calculate the angle of incidence based on the normal and incoming angles.

    Attributes:
        normalAngle (float): The angle of the surface normal in degrees.
        incomingAngle (float): The angle of the incoming ray in degrees.
    """
    return abs(incomingAngle - normalAngle)

def reflect_ray(normal: np.ndarray, incomingRay: Ray) -> Ray:
    """
    Calculate the outgoing angle of the reflected ray.

    Attributes:
        normal (float): The angle of the surface normal in degrees.
        incoming_ray (float): The angle of the incoming ray in degrees.
    """
    if normal.shape != incomingRay.direction.shape:
        raise ValueError("Input vector must match the ray's direction dimension")
    
    dot_product = np.dot(incomingRay.direction, normal)
    reflected_direction = incomingRay.direction - 2 * dot_product * normal
    return Ray(incomingRay.origin, reflected_direction)