import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

from src.Data.Hit import HitInfo
from src.Data.Color import Color
from .BSDF import *
from src.Lighting.Core import LightSource
from src.Lighting.Optics import schlick_fresnel_metalic
from src.Utilities.Common import lerp, unit, attenuate_sqr_distance, attenuate_distance_exponential

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
    ior: float = 1.45
    
    # --- Transparency/Volume Parameters ---
    transmission: float = 0.0      # 0=Opaque, 1=Glass
    absorption_color: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))
    absorption_density: float = 0.0
    
    # --- Emissive ---
    emission_color: Color = field(default_factory=lambda: Color(0.0, 0.0, 0.0))
    emission_intensity: float = 0.0

    # --- Flags ----
    type: MaterialType = MaterialType.DIFFUSE

class PBRMaterial:
    def __init__(
            self,
            data: MaterialData
        ):
        self.data = data

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying `data` container for
        convenience (so callers can use `material.type` or `material.albedo`).
        """
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
            attenuation = attenuate_sqr_distance(dist)

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

            # Material Response (BSDF * CosTheta)
            # Note: evaluate_bsdf returns f_r. We must multiply by cos(theta) for the rendering equation.
            bsdf_val = self.evaluate_bsdf(L, V, N)
            cos_theta = max(0.0, np.dot(N, L))
            
            accumulated_light += (light.color * incoming_radiance) * bsdf_val * cos_theta

        return accumulated_light
    
    def sample(self, incident_dir: np.ndarray, hit_info: HitInfo, sampler: Sampler) -> Tuple[np.ndarray, Color, float]:
        """
        Generates a new ray direction based on the material properties.
        Returns: (New Direction, Throughput Color, PDF)
        """
        N = getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0]))
        I = unit(incident_dir)
        
        # --- A. EMISSIVE ---
        if self.type == MaterialType.EMISSIVE:
            return N, self.data.emission_color * self.data.emission_intensity, 1.0

        # --- B. TRANSPARENT ---
        if self.type == MaterialType.TRANSPARENT:
            return N, self.data.albedo, 1.0
        
        # --- C. DIFFUSE (Lambertian) ---
        if self.type == MaterialType.DIFFUSE:
            # 1. Sample direction from Cosine-Weighted Hemisphere
            # This automatically prefers directions close to the normal.
            new_dir = sampler.sample_cosine_hemisphere(N)
            
            # 2. PDF calculation
            # PDF of Cosine sample = Cos(theta) / PI
            # For robustness, we calculate it, though it often cancels out.
            cos_theta = max(0.0, np.dot(new_dir, N))
            pdf = cos_theta / np.pi
            
            # 3. Throughput (The attenuation of light)
            # Probability cancelation simplification:
            # Weight = (BRDF * Cos) / PDF
            # Weight = (Albedo/PI * Cos) / (Cos/PI) = Albedo
            return new_dir, self.data.albedo, pdf
        
        # --- D. SPECULAR (Metal/Mirror) ---
        if self.type == MaterialType.SPECULAR:
            # 1. Perfect Reflection Vector
            reflected = I - 2.0 * np.dot(I, N) * N
            
            # 2. Roughness (Fuzzy Reflection)
            if self.data.roughness > 0.0:
                fuzz = sampler.sample_unit_sphere() * self.data.roughness
                reflected = reflected + fuzz
                # Renormalize to ensure consistent ray speed/length
                reflected = reflected / np.linalg.norm(reflected)
                
                # If we reflected INTO the object (due to fuzz), absorb the ray
                if np.dot(reflected, N) < 0:
                    return N, Color(0,0,0), 0.0

            # 3. Throughput
            # Metals tint the reflection with their Albedo
            return reflected, self.data.albedo, 1.0
        
        # --- E. GLASS (Dielectric) ---
        if self.type == MaterialType.GLASS:
            ior = self.data.ior
            dt = np.dot(I, N)
            
            # Determine Entering vs Exiting
            if dt > 0:
                # Inside going out
                outward_normal = -N
                ni_over_nt = ior # Assuming air ior = 1.0
                cosine = ior * dt 
            else:
                # Outside going in
                outward_normal = N
                ni_over_nt = 1.0 / ior
                cosine = -dt

            # 1. Calculate Fresnel (Reflection Probability)
            # Schlick's approximation
            r0 = (1 - ior) / (1 + ior)
            r0 = r0**2
            reflect_prob = r0 + (1 - r0) * ((1 - cosine) ** 5)
            
            # 2. Decide: Reflect or Refract?
            # We treat Glass as a singular event (PDF=1.0) chosen randomly
            if sampler.sample_roulette() < reflect_prob:
                # Reflect
                reflected = I - 2.0 * np.dot(I, N) * N
                return reflected, Color(1,1,1), 1.0
            else:
                # Refract (Snell's Law)
                # Inline implementation of refract logic for dependencies
                uv = I
                n = outward_normal
                
                cos_theta_i = min(np.dot(-uv, n), 1.0)
                perp = ni_over_nt * (uv + cos_theta_i * n)
                parallel = -np.sqrt(abs(1.0 - np.dot(perp, perp))) * n
                refracted = perp + parallel
                
                # Check for Total Internal Reflection (TIR)
                # If discriminant was negative, we reflect instead
                if np.isnan(refracted).any(): 
                    reflected = I - 2.0 * np.dot(I, N) * N
                    return reflected, Color(1,1,1), 1.0

                return refracted, Color(1,1,1), 1.0
                
        return N, Color(0,0,0), 0.0

    def evaluate_bsdf(self, incident_dir: np.ndarray, view_dir: np.ndarray, normal: np.ndarray) -> Color:
        """
        Returns the BSDF value (f_r) for a given set of vectors.
        Used for Direct Light Sampling (Next Event Estimation).
        
        incident_dir: Direction TO the light
        view_dir: Direction TO the camera
        """
        # Glass and Specular are "Delta Distributions" (singularities).
        # The probability of hitting the exact perfect reflection angle 
        # when sampling a random point on a light is 0.
        # Therefore, we return Black. Caustics must be handled by the 'sample' method (indirect rays).

        # Lambertian BRDF = Albedo / PI
        if self.type == MaterialType.DIFFUSE:
            return self.data.albedo / np.pi

        return Color(0.0, 0.0, 0.0)

    def get_diffuse_component(self, light_dir: np.ndarray, surface_normal: np.ndarray) -> Color:
        """
        Get the diffuse component (Lambertian). 
        Corrected to respect the Metallic workflow energy conservation.
        """
        NdotL = max(0.0, np.dot(surface_normal, light_dir))
        
        # 1. Standard Lambert Diffuse (Simple but physically consistent)
        diffuse =  self.data.albedo * NdotL
        
        # 2. ENERGY CONSERVATION: Metals have NO diffuse.
        # As metallic approaches 1.0, diffuse must approach 0.0.
        diffuse = diffuse * (1.0 - self.data.metallic)
        
        return diffuse

    def get_specular_component(self, light_dir: np.ndarray, surface_normal: np.ndarray, view_dir: np.ndarray, bias: float = 1e-4) -> Color:
        """Get the specular component of the material response using the Micro-Facet BRDF."""
        
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

    def get_metallic_component(self, bias: float = 1e-4) -> Color:
        """
        Calculates the F0 (Base Reflectivity) for Fresnel calculations.
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

    def get_fresnel_component(self, cos_theta: float) -> np.ndarray:
        """
        Calculates the portion of light that is reflected (Specular) vs. absorbed/refracted (Diffuse).
        """
        f0 = self.get_metallic_component().to_np_array()
        return schlick_fresnel_metalic(cos_theta, f0)

    def get_emissive_component(self) -> Color:
        """Get the emissive color of the material."""
        return self.data.emission_color * self.data.emission_intensity

    def get_volumetric_component(self, distance: float) -> Color:
        """
        Calculates the volumetric attenuation of light passing through the material
        """
        sigma = Color(1.0, 1.0, 1.0) - self.data.absorption_color
        
        # Calculate attenuation per channel
        # e.g. If Red has high sigma, exp(-high * dist) -> 0.0 (Red is blocked)
        attenuation = sigma * attenuate_distance_exponential(distance, self.data.absorption_density)
        
        return attenuation

    def get_transparency_component(self) -> Color:
        """
        Calculates the transparency color contribution of the material.
        """
        return self.data.albedo

    def get_ambient_color(self, ambient_color: Color, ambient_intensity: float) -> Color:
        """
        Calculates the ambient color contribution of the material.
        """
        ambient = ambient_color * ambient_intensity * self.data.albedo
        return ambient

    def __repr__(self):
        return (
            f"Material(albedo={self.data.albedo}, other={self.data})"
        )