import math
import numpy as np

from src.Data.Ray import TracingRay
from src.Utilities.Sampling import Sampler
from src.Lighting.Optics import reflect, refract, schlick_fresnel_refactive
from src.Utilities.Common import unit, orthonormal_basis

def ggx_distribution(normal: np.ndarray, half_vector: np.ndarray, roughness: float) -> float:
    """Calculates the GGX/Trowbridge-Reitz Normal Distribution Function (D)."""
    alpha = max(roughness ** 2, 1e-4)
    dot_n_h = np.dot(normal, half_vector)
    
    # D is 0 if the half vector is below the geometric surface
    if dot_n_h <= 0:
        return 0.0

    denominator = (dot_n_h ** 2) * (alpha ** 2 - 1.0) + 1.0
    return (alpha ** 2) / (np.pi * denominator * denominator)

def smith_geometry(n: np.ndarray, v: np.ndarray, l: np.ndarray, roughness: float) -> float:
    """
    Smith Geometry Shadowing-Masking function.
    Determines what percentage of microfacets are blocked by other microfacets.
    """
    # Using the Schlick-GGX approximation for Smith G
    # k = (alpha + 1)^2 / 8  (for direct lighting / analytic)
    # k = alpha^2 / 2        (for IBL / path tracing) -> We use this usually for consistency
    alpha = roughness ** 2
    k = (alpha) / 2.0 

    dot_n_v = np.abs(np.dot(n, v))
    dot_n_l = np.abs(np.dot(n, l))

    g1_v = dot_n_v / (dot_n_v * (1.0 - k) + k)
    g1_l = dot_n_l / (dot_n_l * (1.0 - k) + k)

    return g1_v * g1_l

def calculate_microfacet_pdf(
    incident_dir: np.ndarray,   # Direction light is coming FROM (World space)
    outgoing_dir: np.ndarray,   # The sampled direction (Reflection or Refraction)
    surface_normal: np.ndarray,
    roughness: float,
    ior_incident: float,
    ior_transmitted: float,
    fresnel_probability: float  # The 'F' value calculated during sampling
) -> float:
    """
    Calculates the PDF for a specific reflection or refraction event.
    """
    # 1. Normalize Vectors
    V = -unit(incident_dir) # View Vector (pointing to viewer)
    L = unit(outgoing_dir)  # Light/Sample Vector
    N = unit(surface_normal)

    # 2. Determine if this is Reflection or Refraction
    # We check if L and N are in the same hemisphere
    is_reflection = np.dot(L, N) > 0

    if is_reflection:
        # --- REFLECTION PDF ---
        
        # Calculate Half Vector (H) for Reflection
        # H = Normalize(V + L)
        H = unit(V + L)
        
        # Calculate Dot Products
        dot_n_h = np.abs(np.dot(N, H))
        dot_v_h = np.abs(np.dot(V, H))
        
        # Calculate D term
        D = ggx_distribution(N, H, roughness)
        
        # Jacobian for Reflection: 1 / (4 * (V.H))
        pdf_geometry = (D * dot_n_h) / (4.0 * dot_v_h + 1e-8)
        
        # Combine with Selection Probability (Fresnel)
        return pdf_geometry * fresnel_probability

    else:
        # --- REFRACTION PDF ---
        
        # Calculate Half Vector (H) for Refraction
        # Standard microfacet H for refraction: -(eta_i * V + eta_t * L)
        # Note: We must be careful with signs. 
        # Usually H is constructed to point into the simpler medium or averaged.
        # Robust method:
        eta_i = ior_incident
        eta_t = ior_transmitted
        
        H_unstand = -(eta_i * V + eta_t * L)
        H = unit(H_unstand)
        
        # D term
        dot_n_h = np.abs(np.dot(N, H))
        D = ggx_distribution(N, H, roughness)
        
        # Calculate Terms for Jacobian
        dot_v_h = np.dot(V, H)
        dot_l_h = np.dot(L, H)
        
        # Denominator part: (eta_i * (V.H) + eta_t * (L.H))^2
        sqrt_denom = (eta_i * dot_v_h + eta_t * dot_l_h)
        denom = sqrt_denom * sqrt_denom
        
        # Jacobian for Refraction
        # J = (eta_t^2 * |L.H|) / (eta_i * (V.H) + eta_t * (L.H))^2
        # Note: The 'D(h) * dot_n_h' part comes from the sampling of H itself.
        jacobian = (eta_t ** 2 * np.abs(dot_l_h)) / (denom + 1e-8)
        
        # PDF in solid angle measure
        pdf_geometry = D * dot_n_h * jacobian 
        
        # Combine with Selection Probability (1 - F)
        # Note: We multiply by derivative of H wrt solid angle
        # Most implementations simplify the weight calculation directly, 
        # but this is the raw PDF value.
        return pdf_geometry * (1.0 - fresnel_probability)
    
def sample_microfacet_surface(
        incident_dir: np.ndarray,
        surface_normal: np.ndarray,
        new_origin: np.ndarray,
        sampler: Sampler,
        roughness: float,
        ior_1: float,
        ior_2: float,
        bias: float = 1e-4
    ) -> TracingRay:
    """
    Unified Microfacet BSDF (Glass/Dielectric).
    Probabilistically samples Reflection or Refraction based on Fresnel term.
    """
    # 1. Setup & IOR
    # -------------------------
    # Direction vectors should be normalized
    V = -unit(incident_dir) # Pointing towards viewer/light source
    N = unit(surface_normal)
    
    # Determine if entering or exiting to set IOR
    dot_vn = np.dot(V, N)
    if dot_vn < 0:
        # Exiting (Inside -> Outside)
        N = -N
        eta_i, eta_t = ior_1, ior_2
    else:
        # Entering (Outside -> Inside)
        eta_i, eta_t = ior_2, ior_1

    # 2. Sample Microfacet Normal (H) (GGX)
    # -------------------------
    # Both reflection and refraction rely on the same microfacet distribution.
    # We sample H once.
    u, v = sampler.sample_bsdf()
    alpha = max(roughness ** 2, bias) # Prevent div by zero

    phi = 2.0 * np.pi * u
    cos_theta = np.sqrt((1.0 - v) / (1.0 + ((alpha ** 2) - 1.0) * v))
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta ** 2))

    h_local = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

    # Transform to World Space
    tangent, bitangent = orthonormal_basis(N)
    H = unit(tangent * h_local[0] + bitangent * h_local[1] + N * h_local[2])

    # Ensure H points into the same hemisphere as the view direction
    dot_vh = max(0.0, np.dot(V, H))
    F = schlick_fresnel_refactive(dot_vh, eta_i, eta_t)

    # 3. Calculate Fresnel Term (The Selector)
    # -------------------------
    dot_v_h = np.dot(V, H)
    dot_v_h = np.clip(dot_v_h, 0.0, 1.0)
    incident_angle_deg = np.degrees(math.acos(dot_v_h))

    if sampler.sample_roulette() < F:
        # --- REFLECTION ---
        final_dir = reflect(incident_dir, H)
        
        # Push away from geometric normal
        final_origin = new_origin + (surface_normal * bias)
    else:
        # --- REFRACTION ---
        final_dir = refract(incident_dir, H, eta_i / eta_t)
        
        # Safety: If your module detects TIR (returning None), fallback to reflection.
        # (Though our F check above should catch this, floating point errors happen).
        if final_dir is None:
            final_dir = reflect(incident_dir, H)

            # Push away from geometric normal
            final_origin = new_origin + (surface_normal * bias)
        else:
            # Origin Offset: Push into geometric normal
            final_origin = new_origin - (surface_normal * bias)

    return TracingRay(origin=final_origin, orientation=final_dir)

def calculate_throughput_weight(
        light_dir: np.ndarray,
        surface_normal: np.ndarray,
        bsdf_value: np.ndarray,
        pdf: float,
        bias: float = 1e-6
    ) -> np.ndarray: 
    """
    Calculates the weight (color contribution) of a specific ray sample 
    using the Monte Carlo estimator: (BSDF * CosTheta) / PDF.
    
    Args:
        light_dir: The direction of the OUTGOING (sampled) ray.
        surface_normal: The geometric surface normal.
        bsdf_value: The RGB color returned by evaluate_bsdf().
        pdf: The probability density calculated by calculate_pdf().
    """
    
    # 1. Safety Check: Avoid division by zero
    if pdf < bias:
        return np.array([0.0, 0.0, 0.0])

    # 2. Geometry Term (Cosine Law / Foreshortening)
    # We take the Absolute value (|N.L|) because:
    # - Reflection: L is on the same side as N (positive).
    # - Refraction: L is on the opposite side of N (negative).
    # Both attenuate light based on the projected area.
    cos_theta = np.abs(np.dot(surface_normal, light_dir))

    # 3. Calculate Weight
    # Weight = (BSDF * CosTheta) / PDF
    weight = (bsdf_value * cos_theta) / pdf

    return weight