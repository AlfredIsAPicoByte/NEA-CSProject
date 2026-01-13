import numpy as np
import math
from typing import Optional

from src.Utilities.Common import unit, safe_asin

def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Calculates the reflection vector R given an incoming vector I and surface normal N.
    Formula: R = I - 2(N · I)N
    
    Args:
        direction: Incoming ray direction (should be normalized).
        normal: Surface normal (should be normalized).
    """
    
    # The dot product projects the direction onto the normal
    dn = np.dot(direction, normal)
    
    # If the ray is coming from inside the object (dot > 0), 
    # the normal is pointing the same way. We usually don't need to flip 
    # for reflection math, but it's good to be aware of.
    
    return direction - (2 * dn * normal)

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
    angle_of_incidence_rad = np.deg2rad(incident_angle)

    sin_theta_t = (refractive_index_incident / refractive_index) * math.sin(angle_of_incidence_rad)
    asin_result = safe_asin(sin_theta_t)

    if asin_result is None:
        return None
    
    transmission_angle_deg = np.rad2deg(asin_result)
    
    if transmission_angle_deg is None:
      # Total internal reflection occured
      return None
    
    incident_rad = np.deg2rad(incident_angle)
    transmission_rad = np.deg2rad(transmission_angle_deg)
    cos_incident = math.cos(incident_rad)
    cos_transmission = math.cos(transmission_rad)

    rs = ((refractive_index_incident * cos_incident - refractive_index * cos_transmission) /
        (refractive_index_incident * cos_incident + refractive_index * cos_transmission)) ** 2
    rp = ((refractive_index_incident * cos_transmission - refractive_index * cos_incident) /
        (refractive_index_incident * cos_transmission + refractive_index * cos_incident)) ** 2
    return (rs + rp) / 2.0

def refract(direction: np.ndarray, normal: np.ndarray, ior_ratio: float) -> Optional[np.ndarray]:
    """
    Calculates the refraction vector using Snell's Law.
    Returns None if Total Internal Reflection (TIR) occurs.

    Args:
        direction: Incoming ray direction (normalized).
        normal: Surface normal (normalized).
        ior_ratio: Ratio of refractive indices (n1 / n2). 
                   e.g., Air->Glass = 1.0/1.5
    """
    cos_theta = min(np.dot(-direction, normal), 1.0)
    
    # Snell's Law Vector Form
    perp = ior_ratio * (direction + cos_theta * normal)
    perp_len_sq = np.dot(perp, perp)
    
    # Check for Total Internal Reflection
    if perp_len_sq > 1.0:
        return None  # TIR: No light enters, it all reflects
    
    parallel = -np.sqrt(abs(1.0 - perp_len_sq)) * normal
    
    return perp + parallel

def schlick_fresnel_refactive(cos_theta: float, ior_incident: float, ior_transmitted: float) -> float:
    """
    Approximates the ratio of light that Reflects vs Refracts.
    Returns a value 0.0 (All Refract) to 1.0 (All Reflect).
    """
    # R0 = Reflection at normal incidence (looking straight on)
    r0 = (ior_incident - ior_transmitted) / (ior_incident + ior_transmitted)
    r0 = r0 * r0
    
    # Schlick approximation
    # R(theta) = R0 + (1 - R0)(1 - cos(theta))^5
    return r0 + (1.0 - r0) * ((1.0 - cos_theta) ** 5)

def schlick_fresnel_metalic(cos_theta: float, f0: np.ndarray) -> np.ndarray:
    """
    Calculates the portion of light that is reflected (Specular) vs. absorbed/refracted (Diffuse).
    """
    return f0 + (1.0 - f0) * ((1.0 - cos_theta) ** 5)

def get_reflection_ratio(direction: np.ndarray, normal: np.ndarray, ior_incident: float, ior_transmitted: float) -> float:
    """
    High-level helper. Returns the probability (0.0-1.0) that a ray reflects.
    Handles TIR automatically (returns 1.0).
    """
    # 1. Determine Cosine
    cos_i = np.dot(direction, normal)
    
    # 2. Handle Orientation for IOR
    etai = ior_incident
    etat = ior_transmitted
    
    if cos_i > 0:
        # Exiting: Swap indices
        etai, etat = etat, etai
    
    # 3. Compute Snell's ratio (sin_t) to check for TIR
    # If we are going from dense -> rare, we might have TIR.
    sin_t = etai / etat * np.sqrt(max(0.0, 1.0 - cos_i * cos_i))
    
    if sin_t >= 1.0:
        # Total Internal Reflection: 100% Reflect
        return 1.0
        
    # 4. Otherwise, use Schlick to approximate reflection amount
    cos_theta = abs(cos_i)
    return schlick_fresnel_refactive(cos_theta, ior_incident, ior_transmitted)

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
