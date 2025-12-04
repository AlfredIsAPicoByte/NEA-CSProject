import numpy as np
import math

from PrimaryStructures import Ray

"""

"""

REFRACTIVE_INDICIES = {
    "air": 1.000293,
    "water": 1.333,
    "glass": 1.5,
    "diamond": 2.42,
    "silicon": 3.42,
    "germanium": 4.0,
    "sapphire": 1.76,
    "quartz": 1.46,
    "ice": 1.31,
    "carbon_dioxide": 1.00045,
    "ethanol": 1.36,
    "methanol": 1.33,
    "benzene": 1.5,
    "acetone": 1.36,
    "toluene": 1.496,
    "chloroform": 1.445,
    "glycerol": 1.473,
    "olive_oil": 1.47,
    "corn_oil": 1.47,
    "castor_oil": 1.475,
    "turpentine": 1.47,
}

SPEED_OF_LIGHT = 299792458 # m/s in a vacuum

def convert_speed_to_index(speed: float, speedOfLight: float = SPEED_OF_LIGHT) -> float:
    """
    Convert the speed of light in a medium to its refractive index.

    Attributes:
        speed (float): The speed of light in the medium in m/s.
        speedOfLight (float): The speed of light in vacuum in m/s. Default is about 3e8 m/s.
    """
    return speedOfLight / speed

def convert_index_to_speed(refractiveIndex: float, speedOfLight: float = SPEED_OF_LIGHT) -> float:
    """
    Convert the refractive index of a medium to the speed of light in that medium.

    Attributes:
        refractiveIndex (float): The refractive index of the medium.
        speedOfLight (float): The speed of light in vacuum in m/s. Default is about 3e8 m/s.
    """
    return speedOfLight / refractiveIndex

def calculate_angle_of_refraction(
    angleOfIncidence: float,
    refractiveIndexIncident: float,
    refractiveIndex: float):
    """
    Calculate the refraction angle based on the law of refraction.
    The angle of refraction is greater than the angle of incidence when the refractive index of the initial medium is greater than the refractive index of the new medium.

    Attributes:
        angleOfIncidence (float): The angle of incidence in degrees.
        refractiveIndexIncident (float): The refractive index of the first medium.
        refractiveIndex (float): The refractive index of the second medium.
    """

    angleOfIncidenceRad = math.radians(angleOfIncidence)
    
    sineAngleOfRefraction = (refractiveIndexIncident / refractiveIndex) * math.sin(angleOfIncidenceRad)
    
    if sineAngleOfRefraction > 1:
        raise ValueError("Total internal reflection occurs; no refraction.")
    
    angleOfRefractionRad = math.asin(sineAngleOfRefraction)
    angleOfRefraction: float = math.degrees(angleOfRefractionRad)
    
    return angleOfRefraction

def calculate_angle_of_incidence(
    angleOfRefraction: float,
    refractiveIndexIncident: float,
    refractiveIndex: float):
    """
    Calculate the refraction angle based on the law of refraction.
    The angle of refraction is greater than the angle of incidence when the refractive index of the initial medium is greater than the refractive index of the new medium.

    Attributes:
        angleOfRefraction (float): The angle of refaction in degrees.
        refractiveIndexIncident (float): The refractive index of the first medium.
        refractiveIndex (float): The refractive index of the second medium.
    """

    angleOfRefractionRad = math.radians(angleOfRefraction)
    
    sineAngleOfIncidence = (refractiveIndexIncident / refractiveIndex) * math.sin(angleOfRefractionRad)
    
    if sineAngleOfIncidence > 1:
        raise ValueError("Total internal reflection occurs; no refraction.")
    
    angleOfIncidenceRad = math.asin(sineAngleOfIncidence)
    angleOfIncidence: float = math.degrees(angleOfIncidenceRad)
    
    return angleOfIncidence

def calculate_refractive_index(
    angleOfIncidence: float,
    angleOfRefraction: float,
    refractiveIndexIncident: float):
    """
    Calculate the refractive index of the second medium based on the law of refraction.

    Attributes:
        angleOfIncidence (float): The angle of incidence in degrees.
        angleOfRefraction (float): The angle of refraction in degrees.
        refractiveIndexIncident (float): The refractive index of the first medium.
    """

    angleOfIncidenceRad = math.radians(angleOfIncidence)
    angleOfRefractionRad = math.radians(angleOfRefraction)
    
    refractiveIndexRefacted: float = (refractiveIndexIncident * math.sin(angleOfIncidenceRad)) / math.sin(angleOfRefractionRad)
    
    return refractiveIndexRefacted

def calculate_refractive_index_incident(
    angleOfIncidence: float,
    angleOfRefraction: float,
    refractiveIndex: float):
    """
    Calculate the refractive index of the first medium based on the law of refraction.

    Attributes:
        angleOfIncidence (float): The angle of incidence in degrees.
        angleOfRefraction (float): The angle of refraction in degrees.
        refractiveIndex (float): The refractive index of the second medium.
    """

    angleOfIncidenceRad = math.radians(angleOfIncidence)
    angleOfRefractionRad = math.radians(angleOfRefraction)
    
    refractiveIndexIncident: float = (refractiveIndex * math.sin(angleOfIncidenceRad)) / math.sin(angleOfRefractionRad)
    
    return refractiveIndexIncident

def calculate_critical_angle(
    refractiveIndexIncident: float,
    refractiveIndex: float):
    """
    Calculate the critical angle for total internal reflection.

    Attributes:
        refractiveIndexIncident (float): The refractive index of the first medium.
        refractiveIndex (float): The refractive index of the second medium.
    """
    
    if refractiveIndexIncident <= refractiveIndex:
        raise ValueError("Total internal reflection does not occur; no critical angle.")
    
    critical_angle_rad = math.asin(refractiveIndex / refractiveIndexIncident)
    critical_angle: float = math.degrees(critical_angle_rad)
    
    return critical_angle

def refract_angle(normalAngle: float, incomingAngle: float, refractiveIndexIncident: float, refractiveIndex: float) -> float:
    """
    Calculate the outgoing angle of the refracted ray based on the law of refraction.
    Attributes:
        normalAngle (float): The angle of the surface normal in degrees.
        incomingAngle (float): The angle of the incoming ray in degrees.
        refractiveIndexIncident (float): The refractive index of the first medium.
        refractiveIndex (float): The refractive index of the second medium.
    """
    incidentAngle = abs(incomingAngle - normalAngle)
    refractionAngle = calculate_angle_of_refraction(incidentAngle, refractiveIndexIncident, refractiveIndex)

    if incomingAngle > normalAngle:
        outgoingAngle = normalAngle + refractionAngle
    else:
        outgoingAngle = normalAngle - refractionAngle

    return outgoingAngle

def refract_ray(normal: np.ndarray, incomingRay: Ray, refractiveIndexIncident: float, refractiveIndex: float) -> Ray:
    """
    Calculate the outgoing angle of the refracted ray.

    Attributes:
        normal (ndarray): the normal of the surface of interaction
        incoming_ray (Ray): the incoming ray
    """
    normal = normal / np.linalg.norm(normal)
    incomingDirection = incomingRay.orientation / np.linalg.norm(incomingRay.orientation)

    cos_theta_i = -np.dot(normal, incomingDirection)
    sin_theta_i2 = 1.0 - cos_theta_i ** 2

    n_ratio = refractiveIndexIncident / refractiveIndex
    sin_theta_t2 = n_ratio ** 2 * sin_theta_i2

    if sin_theta_t2 > 1.0:
        raise ValueError("Total internal reflection occurs; no refraction.")

    cos_theta_t = math.sqrt(1.0 - sin_theta_t2)
    refractedDirection = n_ratio * incomingDirection + (n_ratio * cos_theta_i - cos_theta_t) * normal
    refractedDirection = refractedDirection / np.linalg.norm(refractedDirection)

    return Ray(origin=incomingRay.origin, direction=refractedDirection)
