from __future__ import annotations
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, cast

from src.Data.Hit import HitInfo
from src.Data.Color import Color
from src.Data.Sampling.Core import Sampler
from src.Data.Scene import SceneNode
from .BSDF import *
from src.Lighting.Core import Light
from src.Lighting.Optics import reflect, schlick_fresnel_metalic, refract, schlick_fresnel_refactive
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

@dataclass
class PBRMaterial:
    """
    Physically-Based Rendering (PBR) Material for ray tracing.
    Supports multiple material types: Diffuse, Specular, Glass, Transparent, and Emissive.
    Handles direct lighting, indirect sampling, and BSDF evaluation.
    """
    data: MaterialData = field(default_factory=MaterialData)

    def evaluate_direct_light(
        self,
        light_nodes: List[SceneNode],
        hit_info: HitInfo,
        view_dir: np.ndarray,
        visibility_function: Callable[[np.ndarray, SceneNode], float],
        bias: float = 1e-4
    ) -> Color:
        """
        Calculates direct lighting contribution from all light sources in the scene.
        Evaluates diffuse and specular components based on material type.
        Uses visibility function for shadow calculations (world-space).
        
        :param scene_lights: List of light sources in the scene
        :type scene_lights: List[SceneNode]
        :param hit_info: Surface intersection information (normal, point, etc.)
        :type hit_info: HitInfo
        :param view_dir: Direction from surface point to camera/viewer
        :type view_dir: np.ndarray
        :param visibility_function: Function to determine light visibility (shadow testing)
        :type visibility_function: Callable[[np.ndarray, SceneNode], float]
        :param bias: Small offset to prevent self-intersection in shadow rays
        :type bias: float
        :return: Accumulated color contribution from all direct lights
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
            return self.evaluate_emissive_component()

        for light_node in light_nodes:
            # --- Light Setup (World Space) ---
            light = light_node.context

            light = cast(Light, light)

            L = light.get_direction(light_node.world_transform.position, hit_point)
            dist = light.get_distance(light_node.world_transform.position, hit_point)
            if dist <= bias: continue
            
            # --- Visibility Check (Shadows) ---
            # Note: If checking visibility through glass, this simple function 
            # usually returns 0 (blocked) unless you implement "Transparent Shadows"
            visibility = visibility_function(hit_point, light_node)
            if visibility <= 0.0:
                continue
            
            # Pre-calculate attenuation (distance-based).
            attenuation = attenuate_inv_sqr_distance(dist)

            # Final radiance
            incoming_radiance = light.get_radiance(light_node.world_transform.position, hit_point) * attenuation * visibility
            if incoming_radiance.r + incoming_radiance.g + incoming_radiance.b <= 0.0:
                continue

            # --- Material Handling ---
            if self.data.type == MaterialType.DIFFUSE:
                diffuse = self.evaluate_diffuse_component(L, N)
                accumulated_light += diffuse * incoming_radiance

            elif self.data.type == MaterialType.SPECULAR:
                diffuse = self.evaluate_diffuse_component(L, N)
                specular = self.evaluate_specular_component(L, N, V)
                accumulated_light += (diffuse + specular) * incoming_radiance

        return accumulated_light
    
    def sample_indirect_contribution(
            self,
            incident_dir: np.ndarray,
            surface_normal: np.ndarray,
            sampler: Sampler
        ) -> Tuple[np.ndarray, Color, float]:
        """
        Generates a new ray direction for indirect (global) illumination based on material type.
        Samples from appropriate distribution: Cosine-Weighted (Diffuse), Delta (Specular/Glass), or Pass-through (Transparent).
        
        :param incident_dir: Incoming ray direction
        :type incident_dir: np.ndarray
        :param surface_normal: Surface normal at intersection point
        :type surface_normal: np.ndarray
        :param sampler: Random number sampler for stochastic sampling
        :type sampler: Sampler
        :return: Tuple of (new ray direction, throughput color, PDF probability density)
        :rtype: Tuple[np.ndarray, Color, float]
        """
        N = unit(surface_normal)
        I = unit(incident_dir)
        
        # --- A. EMISSIVE ---
        if self.data.type == MaterialType.EMISSIVE:
            return N, self.evaluate_emissive_component(), 0.0

        # --- B. TRANSPARENT ---
        if self.data.type == MaterialType.TRANSPARENT:
            return I, Color(0.0, 0.0, 0.0), 0.0
        
        # --- C. DIFFUSE (Lambertian) ---
        if self.data.type == MaterialType.DIFFUSE:
            # 1. Sample direction from Cosine-Weighted Hemisphere, already biased towards the normal
            new_dir = sampler.sample_cosine_hemisphere(N)
            
            # 2. PDF calculation
            cos_theta = max(0.0, np.dot(new_dir, N))
            pdf = cos_theta / np.pi
            
            # 3. Throughput (The attenuation of light)
            bsdf = self.data.albedo * (1.0 / np.pi)   # Lambertian BRDF
            return new_dir, bsdf, pdf
        
        # --- D. SPECULAR (Metal/Mirror) ---
        if self.data.type == MaterialType.SPECULAR:
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
                    return N, Color(0.0, 0.0, 0.0), 0.0

            # 3. Throughput
            # Metals tint the reflection with their Albedo
            return reflected, self.evaluate_metallic_component(), 0.0
        
        # --- E. GLASS (Dielectric) ---
        # skipped due to complexity, instead delegated to a seperate function sample_glass_contribution
                
        return N, Color(0.0, 0.0, 0.0), 0.0 # No contribution
    
    def sample_glass_contribution(
            self,
            incident_dir: np.ndarray,
            surface_normal: np.ndarray,
            sampler: Sampler,
            other_ior: float
        ) -> Tuple[np.ndarray, Color]:
        """
        Handles refraction and reflection for glass/dielectric materials using Fresnel equations.
        Chooses between reflection and refraction probabilistically based on Schlick's Fresnel approximation.
        Accounts for total internal reflection (TIR) when rays cannot refract.
        
        :param incident_dir: Incoming ray direction
        :type incident_dir: np.ndarray
        :param surface_normal: Surface normal at intersection point
        :type surface_normal: np.ndarray
        :param sampler: Random number sampler for reflection/refraction decision
        :type sampler: Sampler
        :param other_ior: Index of refraction of the surrounding medium (typically 1.0 for air)
        :type other_ior: float
        :return: Tuple of (new ray direction, throughput color)
        :rtype: Tuple[np.ndarray, Color]
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
            refracted = refract(I, N, n1 / n2)
            
            # Check for Total Internal Reflection (TIR)
            if refracted is None:
                return reflected, Color(1.0, 1.0, 1.0)
            
            return refracted, Color(1.0, 1.0, 1.0)

    def evaluate_bsdf(
            self,
            incident_dir: np.ndarray,
            view_dir: np.ndarray,
            normal: np.ndarray
        ) -> Color:
        """
        Evaluates the Bidirectional Scattering Distribution Function (BSDF) for given ray directions.
        Used for Next Event Estimation and direct light sampling.
        Returns the reflectance/scattering value f_r for the ray configuration.
        
        :param incident_dir: Direction of incoming light
        :type incident_dir: np.ndarray
        :param view_dir: Direction toward observer/camera
        :type view_dir: np.ndarray
        :param normal: Surface normal at intersection
        :type normal: np.ndarray
        :return: BSDF value (color reflectance)
        :rtype: Color
        """
        # Use your existing evaluate_specular_component logic
        L = unit(incident_dir)
        V = unit(view_dir)
        N = unit(normal)

        if self.data.type == MaterialType.DIFFUSE:
            return self.data.albedo / np.pi
        
        # Microfacet BRDF (GGX with roughness)
        elif self.data.type == MaterialType.SPECULAR:
            # Only evaluate if roughness > 0 (otherwise it's a delta distribution)
            if self.data.roughness > 0.01:
                # Calculate the microfacet BRDF
                spec_arr = calculate_microfacet_brdf(self.data.roughness, self.data.specular_intensity,L, V, N, self.evaluate_metallic_component().to_np_array())
                specular_brdf = Color.from_np(spec_arr)

                # Add diffuse component (scaled by metallic)
                diffuse_brdf = (self.data.albedo / np.pi) * (1.0 - self.data.metallic)
                
                return diffuse_brdf + specular_brdf
            else:
                # Perfect mirror - delta distribution
                return Color(0.0, 0.0, 0.0)
        
        # Glass/Dielectric with microfacets
        elif self.data.type == MaterialType.GLASS:
            if self.data.roughness > 0.01:
                # Evaluate both reflection and refraction lobes
                # This is complex - see below
                glass_arr = evaluate_glass_bsdf(self.data.roughness, self.data.ior, L, V, N)
                return Color.from_np(glass_arr)
            else:
                # Perfect glass - delta distribution
                return Color(0.0, 0.0, 0.0)

        return Color(0.0, 0.0, 0.0)

    def evaluate_diffuse_component(
            self,
            light_dir: np.ndarray,
            surface_normal: np.ndarray
        ) -> Color:
        """
        Calculates Lambertian diffuse reflection component.
        Applies energy conservation by reducing diffuse contribution for metallic surfaces.
        
        :param light_dir: Direction toward the light source
        :type light_dir: np.ndarray
        :param surface_normal: Surface normal at intersection point
        :type surface_normal: np.ndarray
        :return: Diffuse color contribution
        :rtype: Color
        """
        NdotL = max(0.0, np.dot(surface_normal, light_dir))
        
        # 1. Standard Lambert Diffuse (Simple but physically consistent)
        diffuse =  self.data.albedo * NdotL
        
        # 2. ENERGY CONSERVATION: Metals have NO diffuse.
        # As metallic approaches 1.0, diffuse must approach 0.0.
        diffuse = diffuse * (1.0 - self.data.metallic)
        
        return diffuse

    def evaluate_specular_component(
            self,
            light_dir: np.ndarray,
            surface_normal: np.ndarray,
            view_dir: np.ndarray
        ) -> Color:
        """
        Evaluates specular reflection using microfacet BRDF (GGX roughness model).
        Incorporates Normal Distribution Function (NDF), Geometric Shadowing (GSF), and Fresnel equations.
        Handles metallic and non-metallic specular highlights.
        
        :param light_dir: Direction toward the light source
        :type light_dir: np.ndarray
        :param surface_normal: Surface normal at intersection point
        :type surface_normal: np.ndarray
        :param view_dir: Direction from surface to viewer
        :type view_dir: np.ndarray
        :return: Specular color contribution
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
        F0 = self.evaluate_metallic_component()
        
        # F_schlick calculation (Color operations are handled correctly)
        FF_arr = schlick_fresnel_metalic(VdotH, F0.to_np_array())
        FF = Color.from_np(FF_arr)

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
        Calculates the base reflectivity (F0) for Fresnel calculations.
        Blends between dielectric (0.04 grey) and metallic (albedo-based) reflectivity.
        Supports specular tinting for non-metallic surfaces.
        
        :param bias: Small value to prevent division by zero in normalization
        :type bias: float
        :return: F0 base reflectivity color for Fresnel computation
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
        Calculates self-illumination contribution for emissive materials.
        Returns emission color modulated by emission intensity.
        
        :return: Emissive light color
        :rtype: Color
        """
        return self.data.emission_color * self.data.emission_intensity

    def evaluate_volumetric_component(self, distance: float) -> Color:
        """
        Calculates light attenuation through volumetric absorption.
        Models how light is absorbed over distance in translucent materials (fog, glass, water).
        Uses exponential attenuation based on absorption color and density.
        
        :param distance: Distance light travels through the volume
        :type distance: float
        :return: Attenuation factor (1.0 = no absorption, 0.0 = fully absorbed)
        :rtype: Color
        """
        sigma = Color(1.0, 1.0, 1.0) - self.data.absorption_color
        
        # Calculate attenuation
        attenuation = sigma * attenuate_distance_exponential(distance, self.data.absorption_density)
        
        return attenuation

    def evaluate_ambient_color(self, ambient_color: Color, ambient_intensity: float) -> Color:
        """
        Calculates ambient lighting contribution.
        Modulates ambient light by material albedo and intensity parameter.
        
        :param ambient_color: Global ambient light color
        :type ambient_color: Color
        :param ambient_intensity: Intensity/brightness of ambient light
        :type ambient_intensity: float
        :return: Ambient contribution to final color
        :rtype: Color
        """
        ambient = ambient_color * ambient_intensity * self.data.albedo
        return ambient

    def __repr__(self):
        return (
            f"Material(albedo={self.data.albedo}, other={self.data})"
        )