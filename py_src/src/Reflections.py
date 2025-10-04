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

def reflect_angle(normalAngle: float, incomingAngle: float) -> float:
    """
    Calculate the outgoing angle of the reflected ray based on the law of reflection.

    Attributes:
        normalAngle (float): The angle of the surface normal in degrees.
        incomingAngle (float): The angle of the incoming ray in degrees.
    """
    incidentAngle = calculate_incident_angle(normalAngle, incomingAngle)
    reflectionAngle = calculate_reflection_angle(incidentAngle)

    if incomingAngle > normalAngle:
        outgoingAngle = normalAngle + reflectionAngle
    else:
        outgoingAngle = normalAngle - reflectionAngle

    return outgoingAngle

def reflect_ray(normal: np.ndarray, incomingRay: Ray) -> Ray:
    """
    Calculate the outgoing direction of the reflected ray.

    Attributes:
        normal (ndarray): the normal of the surface of interaction
        incoming_ray (Ray): the incoming ray
    """
    normal = normal / np.linalg.norm(normal)
    incomingDirection = incomingRay.direction / np.linalg.norm(incomingRay.direction)

    dot_product = np.dot(incomingDirection, normal)
    reflectedDirection = incomingDirection - 2 * dot_product * normal
    reflectedDirection = reflectedDirection / np.linalg.norm(reflectedDirection)

    return Ray(origin=incomingRay.origin, direction=reflectedDirection)