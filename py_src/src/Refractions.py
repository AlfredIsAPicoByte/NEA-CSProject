import math

REFRACTIVE_INDICIES = {
    "air": 1.000293,
    "water": 1.333,
    "glass": 1.5,
    "diamond": 2.42,
    "silicon": 3.42,
    "germanium": 4.0,
    "sapphire": 1.76,
    "quartz": 1.46
}

def convert_speed_to_index(speed: float, speedOfLight: float = 3e8) -> float:
    """
    Convert the speed of light in a medium to its refractive index.

    Attributes:
        speed (float): The speed of light in the medium in m/s.
        speedOfLight (float): The speed of light in vacuum in m/s. Default is 3e8 m/s.
    """
    return speedOfLight / speed

def convert_index_to_speed(refractiveIndex: float, speedOfLight: float = 3e8) -> float:
    """
    Convert the refractive index of a medium to the speed of light in that medium.

    Attributes:
        refractiveIndex (float): The refractive index of the medium.
        speedOfLight (float): The speed of light in vacuum in m/s. Default is 3e8 m/s.
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
