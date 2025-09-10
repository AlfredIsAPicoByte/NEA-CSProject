def calculate_reflection_angle(incidentAngle: float) -> float:
    """
    Calculate the reflection angle based on the law of reflection.
    The reflection angle is equal to the incident angle.

    Parameters:
        incidentincidentAngle (float): The angle of incidence in degrees.

    Returns:
        float: The angle of reflection in degrees.
    """
    return incidentAngle

def calculate_incident_angle(normalAngle: float, incomingAngle: float) -> float:
    """
    Calculate the angle of incidence based on the normal and incoming angles.

    Parameters:
        normalAngle (float): The angle of the surface normal in degrees.
        incomingAngle (float): The angle of the incoming ray in degrees.

    Returns:
        float: The angle of incidence in degrees.
    """
    return abs(incomingAngle - normalAngle)

def reflect_ray(normalAngle: float, incomingAngle: float) -> float:
    """
    Calculate the outgoing angle of the reflected ray.

    Parameters:
        normalAngle (float): The angle of the surface normal in degrees.
        incomingAngle (float): The angle of the incoming ray in degrees.

    Returns:
        float: The angle of the reflected ray in degrees.
    """
    incidentAngle = calculate_incident_angle(normalAngle, incomingAngle)
    reflectionAngle = calculate_reflection_angle(incidentAngle)
    return normalAngle + reflectionAngle

if __name__ == "__main__":
    normal = 90
    incoming = 45  

    reflected = reflect_ray(normal, incoming)
    print(f"Reflected ray angle: {reflected} degrees")