import math
import numpy as np

from src.Data.Color import Color
from src.Data.Ray import TracingRay
from src.Data.Sampling.Core import Sampler
from src.Lighting.Optics import reflect, refract, schlick_fresnel_refactive, schlick_fresnel_metalic
from src.Utilities.Common import unit, orthonormal_basis

def ggx_distribution(roughness: float, N: np.ndarray, H: np.ndarray) -> float:
    """
    Calculates the GGX/Trowbridge-Reitz Normal Distribution Function (D).

    :param roughness: The roughness of the surface.
    :param N: Surface normal.
    :param H: Half vector.
    :return: The D value.
    """
    alpha = max(roughness ** 2, 1e-4)
    dot_n_h = np.dot(N, H)
    
    # D is 0 if the half vector is below the geometric surface
    if dot_n_h <= 0:
        return 0.0

    denominator = (dot_n_h ** 2) * (alpha ** 2 - 1.0) + 1.0
    return (alpha ** 2) / (np.pi * denominator * denominator)

def smith_geometry(roughness: float, N: np.ndarray, V: np.ndarray, L: np.ndarray) -> float:
    """
    Smith Geometry Shadowing-Masking function.
    Determines what percentage of microfacets are blocked by other microfacets.
    """
    # Using the Schlick-GGX approximation for Smith G
    # k = (alpha + 1)^2 / 8  (for direct lighting / analytic)
    # k = alpha^2 / 2        (for IBL / path tracing) -> We use this usually for consistency
    alpha = roughness ** 2
    k = (alpha) / 2.0 

    dot_n_v = np.abs(np.dot(N, V))
    dot_n_l = np.abs(np.dot(N, L))

    g1_v = dot_n_v / (dot_n_v * (1.0 - k) + k)
    g1_l = dot_n_l / (dot_n_l * (1.0 - k) + k)

    return g1_v * g1_l

def calculate_microfacet_pdf(
    roughness: float,
    I: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    ior_incident: float,
    ior_transmitted: float,
    fresnel_probability: float
) -> float:
    """
    Calculates the PDF for a specific reflection or refraction event.

    :param roughness: Surface roughness [0,1]
    :type roughness: float
    :param I: Incident direction (pointing TOWARDS surface)
    :type I: np.ndarray
    :param L: Outgoing direction (pointing AWAY from surface)
    :type L: np.ndarray
    :param N: Surface normal
    :type N: np.ndarray
    :param ior_incident: Index of Refraction of the incident medium
    :type ior_incident: float
    :param ior_transmitted: Index of Refraction of the transmitted medium
    :type ior_transmitted: float
    :param fresnel_probability: Probability of reflection (Fresnel term)
    :type fresnel_probability: float
    :return: The PDF value for the given event
    :rtype: float
    """
    # 1. Normalize Vectors
    V = -unit(I) # View Vector (pointing to viewer)
    L = unit(L)  # Light/Sample Vector
    N = unit(N)

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
        D = ggx_distribution(roughness, N, H)
        
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
        D = ggx_distribution(roughness, N, H)
        
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
        roughness: float,
        I: np.ndarray,
        N: np.ndarray,
        new_origin: np.ndarray,
        sampler: Sampler,
        ior_1: float,
        ior_2: float,
        bias: float = 1e-4
    ) -> TracingRay:
    """
    Unified Microfacet BSDF (Glass/Dielectric).
    Probabilistically samples Reflection or Refraction based on Fresnel term.

    :param roughness: Surface roughness [0,1]
    :type roughness: float
    :param I: Incident direction (pointing TOWARDS surface)
    :type I: np.ndarray
    :param N: Surface normal
    :type N: np.ndarray
    :param new_origin: The origin point for the new ray
    :type new_origin: np.ndarray
    :param sampler: Sampler instance for random sampling
    :type sampler: Sampler
    :param ior_1: Index of Refraction of the incident medium
    :type ior_1: float
    :param ior_2: Index of Refraction of the transmitted medium
    :type ior_2: float
    :param bias: Small bias to offset ray origin to avoid self-intersection
    :type bias: float
    :return: The sampled TracingRay (either reflected or refracted)
    :rtype: TracingRay
    """
    # 1. Setup & IOR
    # -------------------------
    # Direction vectors should be normalized
    V = -unit(I) # Pointing towards viewer/light source
    N = unit(N)
    
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
        final_dir = reflect(I, H)
        
        # Push away from geometric normal
        final_origin = new_origin + (N * bias)
    else:
        # --- REFRACTION ---
        final_dir = refract(I, H, eta_i / eta_t)
        
        # Safety: If your module detects TIR (returning None), fallback to reflection.
        # (Though our F check above should catch this, floating point errors happen).
        if final_dir is None:
            final_dir = reflect(I, H)

            # Push away from geometric normal
            final_origin = new_origin + (N * bias)
        else:
            # Origin Offset: Push into geometric normal
            final_origin = new_origin - (N * bias)

    return TracingRay(origin=final_origin, orientation=final_dir)

def calculate_throughput_weight(
        L: np.ndarray,
        N: np.ndarray,
        bsdf_value: np.ndarray,
        pdf: float,
        bias: float = 1e-6
    ) -> np.ndarray: 
    """
    Calculates the weight (color contribution) of a specific ray sample 
    using the Monte Carlo estimator: (BSDF * CosTheta) / PDF.

    :param L: Light direction (pointing TO light)
    :type L: np.ndarray
    :param N: Surface normal
    :type N: np.ndarray
    :param bsdf_value: The RGB color returned by evaluate_bsdf().
    :type bsdf_value: np.ndarray
    :param pdf: The probability density calculated by calculate_pdf().
    :type pdf: float
    :param bias: Small bias to avoid division by zero.
    :type bias: float
    :return: The throughput weight (RGB color)
    :rtype: np.ndarray
    """

    # 1. Safety Check: Avoid division by zero
    if pdf < bias:
        return np.array([0.0, 0.0, 0.0])

    # 2. Geometry Term (Cosine Law / Foreshortening)
    # We take the Absolute value (|N.L|) because:
    # - Reflection: L is on the same side as N (positive).
    # - Refraction: L is on the opposite side of N (negative).
    # Both attenuate light based on the projected area.
    cos_theta = np.abs(np.dot(N, L))

    # 3. Calculate Weight
    # Weight = (BSDF * CosTheta) / PDF
    weight = (bsdf_value * cos_theta) / pdf

    return weight

def calculate_microfacet_brdf(roughness: float, intensity: float, L: np.ndarray, V: np.ndarray, N: np.ndarray, F0: np.ndarray) -> Color:
    """
    Calculate the Microfacet BRDF using GGX NDF, Smith GSF, and Schlick Fresnel.

    :param roughness: Surface roughness [0,1]
    :type roughness: float
    :param intensity: Intensity multiplier for the BRDF
    :type roughness: float
    :param L: Light direction (pointing TO light)
    :type L: np.ndarray
    :param V: View direction (pointing TO viewer)
    :type V: np.ndarray
    :param N: Surface normal
    :type N: np.ndarray
    :param F0: Base reflectivity at normal incidence (RGB)
    :type F0: np.ndarray
    :return: Evaluated BRDF color
    :rtype: Color
    """
    safe_roughness = max(roughness, 1e-2)
    alpha = safe_roughness ** 2
    alpha_sq = alpha ** 2
    
    H = unit(L + V)
    
    NdotH = max(0.0, np.dot(N, H))
    NdotL = max(0.0, np.dot(N, L))
    NdotV = max(0.0, np.dot(N, V))
    VdotH = max(0.0, np.dot(V, H))
    
    # NDF (GGX)
    denom_ndf = (NdotH * NdotH * (alpha_sq - 1.0) + 1.0)
    NDF = alpha_sq / (np.pi * denom_ndf * denom_ndf)
    
    # GSF (Schlick-GGX)
    k = ((alpha + 1.0) ** 2) / 8.0
    GS_Schlick = lambda n_dot_k: n_dot_k / (n_dot_k * (1.0 - k) + k)
    GSF: float = GS_Schlick(NdotL) * GS_Schlick(NdotV)
    
    # Fresnel (Schlick's Approximation)
    FF = schlick_fresnel_metalic(VdotH, F0)
    
    # BRDF (without cosine term)
    denom_fs = 4.0 * NdotL * NdotV
    
    if denom_fs > 1e-6:
        brdf = Color(*((NDF * GSF * FF) / denom_fs * intensity))
    else:
        brdf = Color(0.0, 0.0, 0.0)
    
    return brdf

def evaluate_glass_bsdf(roughness: float, ior: float, L: np.ndarray, V: np.ndarray, N: np.ndarray) -> Color:
    """
    Evaluate glass BSDF (both reflection and refraction lobes).
    This is tricky because we need to know which lobe the light direction is in.

    :param roughness: Surface roughness [0,1]
    :type roughness: float
    :param ior: Index of Refraction of the material
    :type ior: float
    :param L: Light direction (pointing TO light)
    :type L: np.ndarray
    :param V: View direction (pointing TO viewer)
    :type V: np.ndarray
    :param N: Surface normal
    :type N: np.ndarray
    :return: Evaluated BSDF color
    :rtype: Color
    """
    # Determine if L is a reflection or refraction of V
    is_reflection = np.dot(L, N) * np.dot(V, N) > 0  # Same hemisphere
    
    if is_reflection:
        # Evaluate reflection lobe using microfacet BRDF
        H = unit(L + V)
        D = ggx_distribution(roughness, N, H)
        G = smith_geometry(roughness, N, V, L)
        
        VdotH = max(0.0, np.dot(V, H))
        F = schlick_fresnel_refactive(VdotH, 1.0, ior)
        
        denom = 4.0 * abs(np.dot(N, V)) * abs(np.dot(N, L))
        if denom > 1e-6:
            return Color(1,1,1) * (D * G * F) / denom
        return Color(0,0,0)
    
    else:
        # Evaluate refraction lobe (BTDF)
        # This requires calculating the half-vector for refraction
        eta = 1.0 / ior  # Assuming air->glass
        H = -unit(eta * V + L)  # Refraction half-vector
        
        D = ggx_distribution(roughness, N, H)
        G = smith_geometry(roughness, N, V, L)
        
        VdotH = np.dot(V, H)
        F = schlick_fresnel_refactive(abs(VdotH), 1.0, ior)
        
        # BTDF formula (see Walter et al. 2007)
        LdotH = np.dot(L, H)
        denom = (eta * VdotH + LdotH) ** 2
        
        if abs(denom) > 1e-6:
            btdf = (1 - F) * D * G * abs(VdotH * LdotH) / (abs(np.dot(N, V)) * abs(np.dot(N, L)) * denom)
            # Glass is typically uncolored, but absorption can tint it
            return Color(1.0, 1.0, 1.0) * btdf
        return Color(0.0, 0.0, 0.0)