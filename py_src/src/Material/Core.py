import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

from src.Data.Hit import HitInfo
from src.Data.Color import Color
from .BSDF import *
from src.Lighting.Core import LightSource
from src.Lighting.Optics import reflect, schlick_fresnel_metalic
from src.Utilities.Common import lerp, unit, attenuate_inv_sqr_distance, attenuate_distance_exponential

class MaterialType(Enum):
    DIFFUSE = 1 # Only diffuse reflections
    SPECULAR = 2 # Includes specular reflections and metallic properties
    GLASS = 3 # Uses refraction and reflection based on IOR
    TRANSPARENT = 4 # Just an albeodo tint
    EMISSIVE = 5 # Self-illuminating material, no light calculations

@dataclass
class MaterialData:
    name: str = "DefaultMat"
    
    # --- PBR Parameters ---
    albedo: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))
    roughness: float = 0.5
    metallic: float = 0.0
    specular_intensity: float = 0.5
    specular_tint: float = 0.0
    
    # --- Transparency/Volume Parameters ---
    ior: float = 1.45
    transmission: float = 0.0      # 0=Opaque, 1=Glass
    absorption_color: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))
    absorption_density: float = 0.0
    
    # --- Emissive ---
    emission_color: Color = field(default_factory=lambda: Color(0.0, 0.0, 0.0))
    emission_intensity: float = 0.0

    # --- Flags ----
    type: MaterialType = MaterialType.DIFFUSE

class PBRMaterial:
    """
    
    """
    def __init__(self, data: MaterialData):
        """
        Docstring for __init__
        
        :param self: Description
        :param data: Description
        :type data: MaterialData
        """
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)

    def evaluate_direct_light(
        self,
        scene_lights: List[LightSource],
        hit_info: HitInfo,
        view_dir: np.ndarray,
        visibility_function: Callable[[np.ndarray, LightSource], float],
        bias: float = 1e-4
    ) -> Color:
        """
        Calculates Direct Lighting contribution.
        Assumes hit_info.point and scene_lights are ALL in World Space.
        
        :param scene_lights: Description
        :type scene_lights: List[LightSource]
        :param hit_info: Description
        :type hit_info: HitInfo
        :param view_dir: Description
        :type view_dir: np.ndarray
        :param visibility_function: Description
        :type visibility_function: Callable[[np.ndarray, LightSource], float]
        :param bias: Description
        :type bias: float
        :return: Description
        :rtype: Color
        """
        # Use world-space hit info directly for lighting calculations.
        # Mixing object-space and world-space (e.g., transforming hit point to local
        # space while using world-space light positions) caused incorrect shading.
        V = unit(view_dir)
        N = getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0]))
        hit_point = getattr(hit_info, "point", None)
        accumulated_light = Color(0.0, 0.0, 0.0)
        
        if hit_point is None:
            return accumulated_light
        
        if self.data.type == MaterialType.EMISSIVE:
            return self.get_emissive_component()

        for light in scene_lights:
            # --- Light Setup (World Space) ---
            L, dist = light.get_direction_and_dist(hit_point)
            if dist <= bias: continue
            
            # --- Visibility Check (Shadows) ---
            # Note: If checking visibility through glass, this simple function 
            # usually returns 0 (blocked) unless you implement "Transparent Shadows"
            visibility = visibility_function(hit_point, light)
            if visibility <= 0.0:
                continue
            
            # Pre-calculate attenuation (distance-based).
            attenuation = attenuate_inv_sqr_distance(dist)

            # Final radiance
            incoming_radiance = light.get_radiance(hit_point) * attenuation * visibility
            if incoming_radiance.r + incoming_radiance.g + incoming_radiance.b <= 0.0:
                continue

            # --- Material Handling ---
            if self.type == MaterialType.DIFFUSE:
                diffuse = self.get_diffuse_component(L, N)
                accumulated_light += diffuse * incoming_radiance

            elif self.type == MaterialType.SPECULAR:
                diffuse = self.get_diffuse_component(L, N)
                specular = self.get_specular_component(L, N, V)
                accumulated_light += (diffuse + specular) * incoming_radiance

        return accumulated_light
    
    def sample_indirect_contribution(self, incident_dir: np.ndarray, surface_normal: np.ndarray, sampler: Sampler) -> Tuple[np.ndarray, Color, float]:
        """
        Generates a new ray direction based on the material properties.
        Returns: (New Direction, Throughput Color, PDF)
        
        :param incident_dir: Description
        :type incident_dir: np.ndarray
        :param hit_info: Description
        :type hit_info: HitInfo
        :param sampler: Description
        :type sampler: Sampler
        :return: The new direction for tracing the next ray, the throughput color without the attenuation factor and the probability density function to normalize the brighness of indirect light
        :rtype: Tuple[ndarray[_AnyShape, dtype[Any]], Color, float]
        """
        N = unit(surface_normal)
        I = unit(incident_dir)
        
        # --- A. EMISSIVE ---
        if self.type == MaterialType.EMISSIVE:
            return reflect(I, N), self.evaluate_emissive_component(), 1.0

        # --- B. TRANSPARENT ---
        if self.type == MaterialType.TRANSPARENT:
            return I, self.data.albedo, 1.0
        
        # --- C. DIFFUSE (Lambertian) ---
        if self.type == MaterialType.DIFFUSE:
            # 1. Sample direction from Cosine-Weighted Hemisphere, already biased towards the normal
            new_dir = sampler.sample_cosine_hemisphere(N)
            
            # 2. PDF calculation
            cos_theta = max(0.0, np.dot(new_dir, N))
            pdf = cos_theta / np.pi
            
            # 3. Throughput (The attenuation of light)
            return new_dir, self.data.albedo, pdf
        
        # --- D. SPECULAR (Metal/Mirror) ---
        if self.type == MaterialType.SPECULAR:
            # 1. Perfect Reflection Vector
            reflected = reflect(I, N)
            
            # 2. Roughness (Fuzzy Reflection)
            if self.data.roughness > 0.0:
                fuzz = sampler.sample_unit_sphere() * self.data.roughness
                reflected = reflected + fuzz
                # Renormalize to ensure consistent ray speed/length
                reflected = unit(reflected)
                
                # If we reflected INTO the object (due to fuzz), absorb the ray
                if np.dot(reflected, N) < 0:
                    return N, Color(0,0,0), 0.0

            # 3. Throughput
            # Metals tint the reflection with their Albedo
            return reflected, self.evaluate_metallic_component(), 1.0
        
        # --- E. GLASS (Dielectric) ---
        # skipped due to complexity, instead delegated to a seperate function sample_glass_contribution
                
        return N, Color(0,0,0), 0.0
    
    def sample_glass_contribution(self, incident_dir: np.ndarray, surface_normal: np.ndarray, sampler: Sampler, other_ior: float):
        """
        Docstring for sample_glass_contribution
        
        :param incident_dir: Description
        :type incident_dir: np.ndarray
        :param surface_normal: Description
        :type surface_normal: np.ndarray
        :param sampler: Description
        :type sampler: Sampler
        :param other_ior: Description
        :type other_ior: float
        :return: The new direction for tracing the next ray and the throughput color without the attenuation factor
        :rtype: Tuple[ndarray[_AnyShape, dtype[Any]], Color, float]
        """
        I = unit(incident_dir)
        N = unit(surface_normal)

        dt = np.dot(I, N)
        
        if dt > 0:
            # Inside going out
            n1, n2 = self.data.ior, other_ior
        else:
            # Outside going in
            n1, n2 = other_ior, self.data.ior
        
        cos_theta = abs(dt)
        
        # 1. Calculate Fresnel (Reflection Probability)
        reflect_prob = schlick_fresnel_refactive(cos_theta, n1, n2)
        
        # 2. Decide: Reflect or Refract?
        # We treat Glass as a singular event (PDF=1.0) chosen randomly
        # Pre-compute reflection for later
        reflected = reflect(I, N)

        if sampler.sample_roulette() < reflect_prob: # Reflect
            return reflected, Color(1.0, 1.0, 1.0)
        else: # Refract
            refracted = refract(I, N, n1, n2)
            
            # Check for Total Internal Reflection (TIR)
            if refracted is None:
                return reflected, Color(1.0, 1.0, 1.0), 1.0
            
            return refracted, Color(1.0, 1.0, 1.0)

    def evaluate_bsdf(self, incident_dir: np.ndarray, view_dir: np.ndarray, normal: np.ndarray) -> Color:
        """
        Returns the BSDF value (f_r) for a given set of vectors.
        Used for Direct Light Sampling (Next Event Estimation).
        
        :param incident_dir: Description
        :type incident_dir: np.ndarray
        :param view_dir: Description
        :type view_dir: np.ndarray
        :param normal: Description
        :type normal: np.ndarray
        :return: Description
        :rtype: Color
        """
        if self.type == MaterialType.DIFFUSE:
            return self.data.albedo / np.pi
        
        # Microfacet BRDF (GGX with roughness)
        elif self.type == MaterialType.SPECULAR:
            # Only evaluate if roughness > 0 (otherwise it's a delta distribution)
            if self.data.roughness > 0.01:
                # Use your existing get_specular_component logic
                L = unit(incident_dir)
                V = unit(view_dir)
                N = unit(normal)
                
                # Calculate the microfacet BRDF
                specular_brdf = calculate_microfacet_brdf(L, V, N)
                
                # Add diffuse component (scaled by metallic)
                diffuse_brdf = (self.data.albedo / np.pi) * (1.0 - self.data.metallic)
                
                return diffuse_brdf + specular_brdf
            else:
                # Perfect mirror - delta distribution
                return Color(0.0, 0.0, 0.0)
        
        # Glass/Dielectric with microfacets
        elif self.type == MaterialType.GLASS:
            if self.data.roughness > 0.01:
                # Evaluate both reflection and refraction lobes
                # This is complex - see below
                return evaluate_glass_bsdf(incident_dir, view_dir, normal)
            else:
                # Perfect glass - delta distribution
                return Color(0.0, 0.0, 0.0)

        return Color(0.0, 0.0, 0.0)

    def evaluate_diffuse_component(self, light_dir: np.ndarray, surface_normal: np.ndarray) -> Color:
        """
        Docstring for evaluate_diffuse_component
        
        :param light_dir: Description
        :type light_dir: np.ndarray
        :param surface_normal: Description
        :type surface_normal: np.ndarray
        :return: Description
        :rtype: Color
        """
        NdotL = max(0.0, np.dot(surface_normal, light_dir))
        
        # 1. Standard Lambert Diffuse (Simple but physically consistent)
        diffuse =  self.data.albedo * NdotL
        
        # 2. ENERGY CONSERVATION: Metals have NO diffuse.
        # As metallic approaches 1.0, diffuse must approach 0.0.
        diffuse = diffuse * (1.0 - self.data.metallic)
        
        return diffuse

    def evaluate_specular_component(self, light_dir: np.ndarray, surface_normal: np.ndarray, view_dir: np.ndarray) -> Color:
        """
        Docstring for evaluate_specular_component
        
        :param self: Description
        :param light_dir: Description
        :type light_dir: np.ndarray
        :param surface_normal: Description
        :type surface_normal: np.ndarray
        :param view_dir: Description
        :type view_dir: np.ndarray
        :return: Description
        :rtype: Color
        """
        # --- 0. Pre-Calculations and Constants ---
        safe_roughness = max(self.data.roughness, 1e-2)
        alpha = safe_roughness ** 2
        alpha_sq = alpha ** 2
        
        # Halfway Vector (H)
        H = (light_dir + view_dir)
        H = unit(H) 
        
        # Dot Products (must be clamped to avoid negative light/view angles)
        NdotH = max(0.0, np.dot(surface_normal, H))
        NdotL = max(0.0, np.dot(surface_normal, light_dir))
        NdotV = max(0.0, np.dot(surface_normal, view_dir))
        VdotH = max(0.0, np.dot(view_dir, H))

        # --- 1. Normal Distribution Function (NDF - GGX) ---
        denom_ndf = (NdotH * NdotH * (alpha_sq - 1.0) + 1.0)
        NDF = alpha_sq / (np.pi * denom_ndf * denom_ndf)
        
        # --- 2. Geometric Shadowing Function (GSF - Schlick-GGX Approximation) ---
        k = ((alpha + 1.0) ** 2) / 8.0 
        
        GS_Schlick = lambda n_dot_k: n_dot_k / (n_dot_k * (1.0 - k) + k)
        GSF = GS_Schlick(NdotL) * GS_Schlick(NdotV)
        
        # --- 3. Fresnel Function (FF - Schlick Approximation) ---
        F0 = self.get_metallic_component()
        
        # F_schlick calculation (Color operations are handled correctly)
        term_pow5 = (1.0 - VdotH) ** 5 
        FF = F0 + (Color(1.0, 1.0, 1.0) - F0) * term_pow5

        # --- 4. Final BRDF Term (Fs) and Specular Contribution ---
        # Denominator of the BRDF term
        denom_fs = 4.0 * NdotL * NdotV 
        
        if denom_fs > 0:
            # Fs is the specular BRDF (Fs = D * G * F / denominator)
            Fs = (NDF * GSF * FF) * (1.0 / denom_fs) 
        else:
            Fs = Color(0.0, 0.0, 0.0) 

        # Final Specular Color: Light Intensity * BRDF * Cosine Term
        specular =  Fs * NdotL * self.data.specular_intensity
        
        return specular

    def evaluate_metallic_component(self, bias: float = 1e-4) -> Color:
        """
        Calculates the F0 (Base Reflectivity) for Fresnel calculations.
        
        :param bias: Description
        :type bias: float
        :return: Description
        :rtype: Color
        """
        # --- 1. Dielectric (Non-Metal) F0 ---
        # Base reflectivity for plastic/glass/water is approx 0.04 (Linear Grey)
        F0_dielectric = Color(0.04, 0.04, 0.04)
        
        # If the material is very dark, we don't want the F0 to drop to 0.
        # So we ensure the tinted version maintains some brightness if needed, 
        # but usually just using Albedo is fine for the tint *color*.
        # For correct magnitude, we just lerp the *color* of the 0.04 base.
        
        # Lerp between Grey (0.04, 0.04, 0.04) and Tinted (0.04 * AlbedoColor)
        # Note: We normalize albedo to preserve the 0.04 intensity magnitude.
        max_val = max(self.data.albedo.r, self.data.albedo.g, self.data.albedo.b, bias)
        albedo_normalized = self.data.albedo / max_val
        
        F0_dielectric_tinted = F0_dielectric * albedo_normalized
        
        # Apply the Specular Tint Slider
        F0_final_dielectric = lerp(F0_dielectric, F0_dielectric_tinted, self.data.specular_tint)

        # --- 2. Metallic F0 ---
        # Metals use their Albedo directly as F0
        F0_metal = self.data.albedo

        # --- 3. Final Blend ---
        # Lerp between Dielectric (0.04) and Metal (Albedo)
        final_F0: Color = lerp(F0_final_dielectric, F0_metal, self.data.metallic)
        
        return final_F0

    def evaluate_emissive_component(self) -> Color:
        """
        Docstring for evaluate_emissive_component
        
        :return: Description
        :rtype: Color
        """
        return self.data.emission_color * self.data.emission_intensity

    def evaluate_volumetric_component(self, distance: float) -> Color:
        """
        Calculates the volumetric attenuation of light passing through the material.
        
        :param distance: Description
        :type distance: float
        :return: Description
        :rtype: Color
        """
        sigma = Color(1.0, 1.0, 1.0) - self.data.absorption_color
        
        # Calculate attenuation
        attenuation = sigma * attenuate_distance_exponential(distance, self.data.absorption_density)
        
        return attenuation

    def evaluate_ambient_color(self, ambient_color: Color, ambient_intensity: float) -> Color:
        """
        Docstring for evaluate_ambient_color
        
        :param ambient_color: Description
        :type ambient_color: Color
        :param ambient_intensity: Description
        :type ambient_intensity: float
        :return: Description
        :rtype: Color
        """
        ambient = ambient_color * ambient_intensity * self.data.albedo
        return ambient

    def __repr__(self):
        return (
            f"Material(albedo={self.data.albedo}, other={self.data})"
        )