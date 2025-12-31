import numpy as np
import math
from typing import Tuple, Optional
from PrimaryStructures import Ratio

REFRACTIVE_INDICES = {
    # --- Common Gases (at STP) ---
    "vacuum": 1.0000,
    "air": 1.000293,
    "helium": 1.000036,
    "hydrogen": 1.000132,
    "carbon_dioxide": 1.00045,

    # --- Common Liquids ---
    "water": 1.333,
    "water_ice": 1.31,
    "seawater": 1.339,
    "ethanol": 1.36,
    "methanol": 1.329,
    "acetone": 1.36,
    "benzene": 1.501,
    "glycerol": 1.473,
    "turpentine": 1.472,
    "chlorine_liquid": 1.385,
    "milk": 1.35,
    "vodka": 1.363,
    "whisky": 1.356,
    "honey": 1.484,      # Varies by water content (1.48 - 1.50)
    "bromine": 1.661,
    "shampoo": 1.362,
    "blood": 1.35,

    # --- Oils ---
    "olive_oil": 1.47,
    "corn_oil": 1.47,
    "castor_oil": 1.475,
    "vegetable_oil": 1.47,
    "mineral_oil": 1.46,
    "clove_oil": 1.535,
    "cinnamon_oil": 1.60,
    "orange_oil": 1.473,
    "cedarwood_oil": 1.51,
    "linseed_oil": 1.48,

    # --- Glass & Man-made Solids ---
    "glass": 1.5,            # Typical generic value
    "glass_crown": 1.52,     # Standard window/optical glass
    "glass_flint": 1.62,
    "glass_flint_heavy": 1.65,
    "glass_flint_dense": 1.66,
    "glass_pyrex": 1.474,
    "glass_arsenic_trisulfide": 2.04,
    "fused_silica": 1.458,
    "lucite": 1.495,
    "plexiglass": 1.49,
    "polycarbonate": 1.585,  # Eyeglass lenses
    "acrylic": 1.49,
    "pet_plastic": 1.575,    # Water bottles
    "polystyrene": 1.55,
    "teflon": 1.35,
    "nylon": 1.53,

    # --- Minerals & Crystals ---
    "quartz": 1.46,
    "rock_salt": 1.516,
    "fluorite": 1.433,
    "calcite": 1.486,        # Birefringent (average)
    "silicon": 3.42,
    "germanium": 4.0,
    "alumina": 1.77,
    "titanium_dioxide": 2.61,

    # --- Gemstones ---
    "diamond": 2.417,
    "emerald": 1.576,
    "ruby": 1.761,
    "sapphire": 1.762,
    "star_sapphire": 1.760,
    "amethyst": 1.544,
    "amber": 1.54,
    "opal": 1.45,
    "citrine": 1.55,
    "garnet": 1.78,          # Varies by composition
    "topaz": 1.62,
    "jade_nephrite": 1.61,
    "jade_jadeite": 1.665,
    "peridot": 1.654,
    "aquamarine": 1.577,
    "lapis_lazuli": 1.50,
    "zircon": 1.92,
    "moissanite": 2.65,
    "cubic_zirconia": 2.15,
    "pearl": 1.53,

    # --- Metals (Approximations) ---
    # Note: In PBR, metals rely on Complex IOR (n, k). 
    # These are scalar approximations for "Real" (n) part, 
    # but metal rendering usually relies on albedo + metallic = 1.0.
    "gold_approx": 0.47,
    "silver_approx": 0.135,
    "copper_approx": 1.1,
    "aluminum_approx": 1.44,
    "iron_approx": 2.9,
    "titanium_approx": 2.16,
}

SPEED_OF_LIGHT = 299792458 # m/s in a vacuum

def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return np.linalg.norm(v) + eps

def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / _safe_norm(v, eps)

def _deg_to_rad(deg: float) -> float:
    return math.radians(deg)

def _rad_to_deg(rad: float) -> float:
    return math.degrees(rad)

def _safe_asin(value: float) -> Optional[float]:
    if value > 1.0 or value < -1.0:
        return None
    return math.asin(value)

def convert_SI_speed_to_index(
        speed: float,
        speed_of_light: float = SPEED_OF_LIGHT
    ) -> float:
    """
    Convert the speed of light in a medium to its refractive index.

    Args:
        speed (float): The speed of light in the medium in m/s.
        speed_of_light (float): The speed of light in vacuum in m/s. Default is about 3e8 m/s.
    """
    return speed_of_light / speed

def convert_index_to_SI_speed(
        refractive_index: float,
        speed_of_light: float = SPEED_OF_LIGHT
    ) -> float:
    """
    Convert the refractive index of a medium to the speed of light in that medium.

    Args:
        refractive_index (float): The refractive index of the medium.
        speed_of_light (float): The speed of light in vacuum in m/s. Default is about 3e8 m/s.
    """
    return speed_of_light / refractive_index

def calculate_angle_of_refraction(
        angle_of_incidence: float,
        refractive_index_incident: float,
        refractive_index: float
    ) -> Optional[float]:
    """
    Calculate the refraction angle based on the law of refraction.
    The angle of refraction is greater than the angle of incidence when the refractive index of the initial medium is greater than the refractive index of the new medium.

    Args:
        angle_of_incidence (float): The angle of incidence in degrees.
        refractive_index_incident (float): The refractive index of the first medium.
        refractive_index (float): The refractive index of the second medium.
    """
    angle_of_incidence_rad = _deg_to_rad(angle_of_incidence)

    sin_theta_t = (refractive_index_incident / refractive_index) * math.sin(angle_of_incidence_rad)
    asin_result = _safe_asin(sin_theta_t)

    if asin_result is None:
        return None

    return _rad_to_deg(asin_result)

def calculate_reflectance(
        incident_angle: float,
        refractive_index_incident: float,
        refractive_index: float
    ) -> Optional[float]:
    """
    Calculate the reflectance using Fresnel equations for unpolarized light.

    Attributes:
        incident_angle (float): The angle of incidence in degrees.
        refractive_index_incident (float): The refractive index of the first medium.
        refractive_index (float): The refractive index of the second medium.
    """
    transmission_angle_deg = calculate_angle_of_refraction(incident_angle, refractive_index_incident, refractive_index)
    
    if transmission_angle_deg is None:
      # Total internal reflection occured
      return None
    
    incident_rad = _deg_to_rad(incident_angle)
    transmission_rad = _deg_to_rad(transmission_angle_deg)
    cos_incident = math.cos(incident_rad)
    cos_transmission = math.cos(transmission_rad)

    rs = ((refractive_index_incident * cos_incident - refractive_index * cos_transmission) /
        (refractive_index_incident * cos_incident + refractive_index * cos_transmission)) ** 2
    rp = ((refractive_index_incident * cos_transmission - refractive_index * cos_incident) /
        (refractive_index_incident * cos_transmission + refractive_index * cos_incident)) ** 2
    return (rs + rp) / 2.0

def calculate_angle_of_incidence(
        angle_of_refraction: float,
        refractive_index_incident: float,
        refractive_index: float
    ) -> Optional[float]:
    """
    Calculate the refraction angle based on the law of refraction.
    The angle of refraction is greater than the angle of incidence when the refractive index of the initial medium is greater than the refractive index of the new medium.

    Args:
        angle_of_refraction (float): The angle of refaction in degrees.
        refractive_index_incident (float): The refractive index of the first medium.
        refractive_index (float): The refractive index of the second medium.
    """
    # Use Snell's law in reverse: n1*sin(theta1) = n2*sin(theta2)
    # To compute theta1 given theta2, swap the indices and call calculate_angle_of_refraction
    return calculate_angle_of_refraction(angle_of_refraction, refractive_index, refractive_index_incident)

def calculate_refractive_index(
        angle_of_incidence: float,
        angle_of_refraction: float,
        refractive_index_incident: float,
        bias: float = 1e-8
    ) -> float:
    """
    Calculate the refractive index of the second medium based on the law of refraction.

    Args:
        angle_of_incidence (float): The angle of incidence in degrees.
        angle_of_refraction (float): The angle of refraction in degrees.
        refractive_index_incident (float): The refractive index of the first medium.
    """
    angle_inc_rad = _deg_to_rad(angle_of_incidence)
    angle_ref_rad = _deg_to_rad(angle_of_refraction)

    return (refractive_index_incident * math.sin(angle_inc_rad)) / (math.sin(angle_ref_rad) + bias)

def calculate_refractive_index_incident(
        angle_of_incidence: float,
        angle_of_refraction: float,
        refractive_index: float,
        bias: float = 1e-8
    ) -> float:
    """
    Calculate the refractive index of the first medium based on the law of refraction.

    Args:
        angle_of_incidence (float): The angle of incidence in degrees.
        angle_of_refraction (float): The angle of refraction in degrees.
        refractive_index (float): The refractive index of the second medium.
    """
    angle_inc_rad = _deg_to_rad(angle_of_incidence)
    angle_ref_rad = _deg_to_rad(angle_of_refraction)

    # From Snell: n1*sin(theta1) = n2*sin(theta2) => n1 = n2*sin(theta2)/sin(theta1)
    return refractive_index * math.sin(angle_ref_rad) / (math.sin(angle_inc_rad) + bias)

def calculate_critical_angle(
        refractive_index_incident: float,
        refractive_index: float
    ) -> float:
    """
    Calculate the critical angle for total internal reflection.

    Attributes:
        refractive_index_incident (float): The refractive index of the first medium.
        refractive_index (float): The refractive index of the second medium.
    """
    
    if refractive_index_incident <= refractive_index:
        return None

    critical_angle_rad = math.asin(refractive_index / refractive_index_incident)
    return math.degrees(critical_angle_rad)

def calculate_refraction_angle(
        surface_normal_angle: float,
        incoming_angle: float,
        refractive_index_incident: float,
        refractive_index: float
    ) -> float:
    """
    Calculate the outgoing direction of the refracted ray based on the law of refraction.
    
    Args:
        surface_normal_angle (float): The angle of the surface surface_normal in degrees.
        incoming_angle (float): The angle of the incoming ray in degrees.
        refractive_index_incident (float): The refractive index of the first medium.
        refractive_index (float): The refractive index of the second medium.
    """
    incident_angle = abs(incoming_angle - surface_normal_angle)
    refraction_angle = calculate_angle_of_refraction(incident_angle, refractive_index_incident, refractive_index)

    if refraction_angle is None:
        return None

    if incoming_angle > surface_normal_angle:
        return surface_normal_angle + refraction_angle
    else:
        return surface_normal_angle - refraction_angle

def calculate_refraction_vector(
        surface_normal: np.ndarray,
        direction: np.ndarray,
        refractive_index_incident: float,
        refractive_index: float,
        bias: float = 1e-8
    ) -> Optional[np.ndarray]:
    """
    Calculate the outgoing angle of the refracted ray using Snell's Law.
    Handles both entering and exiting cases by ensuring the normal opposes the ray.
    """
    # 1. Normalize inputs
    unit_direction = _unit(direction, bias)
    unit_normal = _unit(surface_normal, bias)

    # 2. Check orientation: Are we entering or exiting?
    # Dot product > 0 means the ray and normal point in the same direction (Exiting).
    dt = np.dot(unit_direction, unit_normal)
    
    # If we are exiting, the normal is pointing 'out' with the ray.
    # We need to flip it to point 'in' against the ray for the formula to work.
    if dt > 0:
        eff_normal = -unit_normal
        cos_theta_i = dt # dot(I, N) is already positive here
    else:
        eff_normal = unit_normal
        cos_theta_i = -dt # We want positive cosine (angle < 90)

    # 3. Calculate Ratio (eta)
    # The caller is responsible for swapping n1/n2 based on 'is_inside',
    # so we just divide the incident by the transmission index.
    eta = refractive_index_incident / refractive_index

    # 4. Check for Total Internal Reflection (TIR)
    # sin^2(theta_t) = eta^2 * sin^2(theta_i)
    # sin^2(theta_i) = 1 - cos^2(theta_i)
    sin_theta_t2 = (eta * eta) * (1.0 - cos_theta_i * cos_theta_i)

    if sin_theta_t2 > 1.0:
        # TIR: The ray cannot escape; it reflects entirely inside.
        return None

    # 5. Calculate Refraction Vector
    # T = eta * I + (eta * cos_i - sqrt(1 - sin_t^2)) * N
    cos_theta_t = math.sqrt(1.0 - sin_theta_t2)
    
    refracted_vector = (eta * unit_direction) + \
                       ((eta * cos_theta_i) - cos_theta_t) * eff_normal

    return _unit(refracted_vector, bias)

"""
Refraction module: Provides functions to calculate refraction angles, refractive indices, and refracted ray directions based on Snell's Law.
"""