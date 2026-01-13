from __future__ import annotations
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import replace

from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from Core import RenderStats
from src.Lighting.Optics import REFRACTIVE_INDICES
from src.Utilities.Sampling import Sampler

class InteractionStrategy(ABC):
    @abstractmethod
    def interact(
        self,
        ray: TracingRay,
        hit_info: HitInfo,
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["RenderStats"] = None
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
    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo,
        bias: float = 1e-4,
        stats: Optional["RenderStats"] = None,
        *args, **kwargs
    ) -> Optional[TracingRay]:
        hit_point = getattr(hit_info, "point", None)
        if hit_point is None:
            stats.roulette_kills += 1
            return None
        
        next_origin = hit_point + (ray.orientation * bias)
        
        if stats is not None:
            stats.rays_transparency += 1

        return replace(ray, origin=next_origin)

class PassthroughInteraction(InteractionStrategy):
    """
    A passthrough interaction. 
    The ray passes perfectly straight through the object, ignoring refraction.
    - 'Ghost' objects (semi-transparent overlays).
    - Volumetric boundaries (fog containers).
    - Debugging geometry without occluding the scene.
    """
    def __init__(
        self,
        opacity_cutoff: float = 0.0, # 0.0 = Invisible/Clear, 1.0 = Opaque (Absorbs ray)
    ):
        # Initialize base to handle samplers/IOR if needed later
        self.opacity_cutoff = np.clip(opacity_cutoff, 0.0, 1.0)
    
    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["RenderStats"] = None
    ) -> Optional[TracingRay]:
        # 1. Stochastic Opacity Check (Russian Roulette)
        # If opacity is 0.5, 50% of rays are blocked, 50% pass through.
        # This simulates semi-transparency without splitting rays.
        if self.opacity_cutoff > 0.0:
            if sampler.random_float() < self.opacity_cutoff:
                stats.roulette_kills += 1
                return None # Ray is absorbed/blocked by the "smoke"

        # 2. Passthrough Logic
        # We spawn a new ray continuing in the exact same direction.
        hit_point = getattr(hit_info, "point", None)
        if hit_point is None:
            stats.roulette_kills += 1
            return None
        
        next_origin = hit_point + (ray.orientation * bias)
        
        if stats is not None:
            stats.rays_transparency += 1

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
        stats: Optional["RenderStats"] = None
    ) -> Optional[TracingRay]:
        new_ray = ray

        new_ray.current_depth += 1

        return new_ray
