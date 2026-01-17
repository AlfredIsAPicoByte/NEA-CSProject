from __future__ import annotations
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import replace

from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from ..Core import TracingStats
from src.Material.Core import PBRMaterial, MaterialType
from src.Material.BSDF import * 
from src.Lighting.Optics import REFRACTIVE_INDICES
from src.Data.Sampling import Sampler

class InteractionStrategy(ABC):
    @abstractmethod
    def interact(
        self,
        ray: TracingRay,
        hit_info: HitInfo,
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Optional[TracingRay]:
        ...

class TerminalInteraction(InteractionStrategy):
    """
    A 'Null' interaction.
    The ray is absorbed or the calculation is finished.
    - Debug Views (X-Ray, Normals, Depth) where shading is self-contained.
    - Matte/Black hole materials.
    - Light sources (if they don't reflect).
    """
    def interact(self, *args, **kwargs) -> Optional[TracingRay]:
        return None

class PassthroughInteraction(InteractionStrategy):
    """
    A passthrough interaction. 
    The ray passes perfectly straight through the object, ignoring refraction.
    Using stochastic alpha clipping/opacity checking to determine if the ray interacts or not.
    - 'Ghost' objects (semi-transparent overlays).
    - Volumetric boundaries (fog containers).
    - Debugging geometry without occluding the scene.
    """
    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Optional[TracingRay]:
        if not hit_info.hit:
            return None
        
        P = getattr(hit_info, "point", None)
        if P is None:
            return None
        
        # Alpha Clipping
        material: Optional[PBRMaterial] = getattr(hit_info.obj, "material", None)
        if material is None:
            return None
        
        # Get alpha value (0.0 = fully transparent, 1.0 = fully opaque)
        alpha = np.clip(material.data.albedo.a, 0.0, 1.0)
        
        # Random value in [0, 1)
        random_value = sampler.next_1d()
        
        # If random value < alpha, the surface is opaque (block the ray)
        # If random value >= alpha, the surface is transparent (pass through)
        if random_value < alpha:
            # Opaque - ray is blocked, terminate it
            if stats: 
                stats.rays_transparency += 1
            return None
        else:
            # Transparent - pass through
            next_origin = P + (ray.orientation * bias)
            return replace(ray, origin=next_origin)

class StandardInteraction(InteractionStrategy):
    """
    A unified PBR-style interaction.
    - Can simulate: Mirrors, Glass, Matte, Metal, and Glossy surfaces.
    """
    def __init__(
        self,
        scene_ior: float = REFRACTIVE_INDICES["air"],
    ):
        self.scene_ior = scene_ior

    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Optional[TracingRay]:
        if not hit_info.hit:
            if stats: stats.roulette_kills += 1
            return None
        
        alpha_cliped = PassthroughInteraction().interact(ray, hit_info, sampler, bias, stats)
        if alpha_cliped is not None:
            if stats: stats.roulette_kills += 1
            return alpha_cliped
        
        # 1. Russian Roulette (Path Termination)
        # If the ray is very dim (low throughput), randomly kill it to save time.
        # ray.throughput is carried over from previous bounces
        current_throughput = np.array(ray.throughput)  
        max_component = max(current_throughput[0], current_throughput[1], current_throughput[2])
        
        # Only start killing after a few bounces (e.g. depth > 3) to reduce noise
        if ray.current_depth > 3:
            probability = float(max(min(max_component, 0.95), 0.05))  # Clamp to [0.05, 0.95]
            
            if sampler.sample_roulette() > probability:
                if stats: stats.roulette_kills += 1
                return None
            
            # Boost throughput to remain unbiased
            current_throughput = current_throughput / probability

        # 2. Material Sampling
        # Ask material: "Give me a random direction based on your roughness"
        material: Optional[PBRMaterial] = getattr(hit_info.obj, "material", None)

        if material is None:
            return None
        
        # Emissive sources terminate ray, becuase they are also considered light sources.
        if material.type == MaterialType.EMISSIVE:
            return None
        
        # Prepare Geometry
        N = unit(getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0])))
        I = unit(ray.orientation) # Incident vector

        P = getattr(hit_info, "point", None)
        if P is None:
            return None

        # ==========================================================
        # CASE 1: DIFFUSE (Matte / Plastic / Wood)
        # ==========================================================
        if material.type == MaterialType.DIFFUSE:
            if stats: stats.rays_reflection += 1

            # 1. Importance Sample the Cosine Hemisphere
            # (Prefers directions close to the normal)
            new_dir = sampler.sample_cosine_hemisphere(N)

            new_throughput = current_throughput[:3] * material.data.albedo.to_np_array(False)[:3]
            
            return TracingRay(
                origin=P + (N * bias),
                orientation=new_dir,
                throughput=new_throughput,
                current_depth=ray.current_depth + 1,
            )

        # ==========================================================
        # CASE 2: SPECULAR (Metals: Gold, Copper, Mirror)
        # ==========================================================
        elif material.type == MaterialType.SPECULAR:
            if stats: stats.rays_reflection += 1

            reflect_dir = reflect(I, N)

            roughness = material.data.roughness
            if roughness > bias:
                # Add a random vector in a sphere and normalize
                fuzz = sampler.sample_unit_sphere() * roughness
                reflect_dir = unit(reflect_dir + fuzz)
                
                # Check if we reflected back into the surface (absorb ray)
                if np.dot(reflect_dir, N) <= 0:
                    return None

            # Throughput for metals is usually the Albedo (they tint reflection)
            new_throughput = current_throughput[:3] * material.data.albedo.to_np_array(include_alpha=False)[:3]

            return TracingRay(
                origin=P + (N * bias),
                orientation=reflect_dir,
                throughput=new_throughput,
                current_depth=ray.current_depth + 1,
                # medium_density=ray.medium_density,
                # medium_color=ray.medium_color
            )
            
        # ==========================================================
        # CASE 3: GLASS (Dielectric Refraction)
        # ==========================================================
        elif material.type == MaterialType.GLASS:
            if stats: stats.rays_refraction += 1
            
            ior = getattr(material.data, "ior", 1.5)

            # Determine Entering vs Exiting
            I_normalized = unit(I)
            dt = np.dot(I_normalized, N)

            if dt > 0:
                # Inside going out
                outward_normal = -N
                n1, n2 = ior, self.scene_ior
                cosine = abs(dt)
                entering = False
            else:
                # Outside going in
                outward_normal = N
                n1, n2 = self.scene_ior, ior
                cosine = abs(dt)
                entering = True

            ni_over_nt = n1 / n2

            # 1. Calculate Fresnel (Reflection Probability)
            reflect_prob = schlick_fresnel_refactive(cosine, n1, n2)
            
            # 2. Decide: Reflect or Refract?
            if sampler.sample_roulette() < reflect_prob:
                # --- REFLECTION ---
                reflected = reflect(I, N)
                return TracingRay(
                    origin=P + (outward_normal * bias),
                    orientation=reflected,
                    throughput=current_throughput, # Glass reflection is white (usually)
                    current_depth=ray.current_depth + 1,
                    is_inside=ray.is_inside
                )
            else:
                # --- REFRACTION ---
                refracted = refract(I, outward_normal, ni_over_nt)
                
                if refracted is None:
                    # Total Internal Reflection (TIR) - Force Reflection
                    reflected = reflect(I, N)
                    return TracingRay(
                        origin=P + (outward_normal * bias),
                        orientation=reflected,
                        throughput=current_throughput,
                        current_depth=ray.current_depth + 1,
                        is_inside=ray.is_inside
                    )
                
                # Successful Refraction
                new_ray = TracingRay(
                    origin=P - (outward_normal * bias), # Push THROUGH surface
                    orientation=refracted,
                    throughput=current_throughput, # Start with current throughput
                    current_depth=ray.current_depth + 1,
                    is_inside=not ray.is_inside
                )
                
                # Apply Beer's Law absorption when EXITING the material
                if not entering:
                    # Get volumetric absorption as Color
                    absorption = material.get_volumetric_component(hit_info.distance)
                    
                    # Multiply throughput by absorption (Beer's Law filtering)
                    absorption_rgb = absorption.to_np_array(include_alpha=False)[:3]
                    new_ray.throughput = new_ray.throughput * absorption_rgb  # ← FIXED: multiply instead of add
                    
                return new_ray
            
        return None