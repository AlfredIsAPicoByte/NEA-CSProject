import numpy as np

"""

"""

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