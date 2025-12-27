import numpy as np

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

def calculate_reflectance(incidentAngle: float, refractiveIndex1: float, refractiveIndex2: float) -> float:
    """
    Calculate the reflectance using Fresnel equations for unpolarized light.

    Attributes:
        incidentAngle (float): The angle of incidence in degrees.
        refractiveIndex1 (float): The refractive index of the first medium.
        refractiveIndex2 (float): The refractive index of the second medium.
    """
    import math

    # Convert angle to radians
    incidentAngleRad = math.radians(incidentAngle)

    # Calculate sine of transmission angle using Snell's law
    sinTransmissionAngle = (refractiveIndex1 / refractiveIndex2) * math.sin(incidentAngleRad)

    # Total internal reflection check
    if abs(sinTransmissionAngle) > 1.0:
        return 1.0  # Total internal reflection

    transmissionAngleRad = math.asin(sinTransmissionAngle)

    cosIncident = math.cos(incidentAngleRad)
    cosTransmission = math.cos(transmissionAngleRad)

    rs = ((refractiveIndex1 * cosIncident - refractiveIndex2 * cosTransmission) /
          (refractiveIndex1 * cosIncident + refractiveIndex2 * cosTransmission)) ** 2
    rp = ((refractiveIndex1 * cosTransmission - refractiveIndex2 * cosIncident) /
          (refractiveIndex1 * cosTransmission + refractiveIndex2 * cosIncident)) ** 2

    reflectance = (rs + rp) / 2.0
    return reflectance

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

def reflect_ray(normal: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
    """
    Compute reflected ray direction using law of reflection.
    
    Args:
        normal: Surface normal (will be normalized)
        ray_direction: Incoming ray direction (will be normalized)
    
    Returns:
        Reflected ray direction (normalized)
    """
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    direction = ray_direction / (np.linalg.norm(ray_direction) + 1e-12)
    
    dot_product = np.dot(direction, normal)
    reflected = direction - 2 * dot_product * normal
    return reflected / (np.linalg.norm(reflected) + 1e-12)

"""
Reflection module: Provides functions to calculate reflection angles and reflected ray directions based on the law of reflection.
"""