from __future__ import annotations
import numpy as np
from typing import Optional, Callable, cast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from src.Data.Color import Color, ColorGradient
from .Core import RenderStats
from src.Material.Core import PBRMaterial, MaterialType
from src.Lighting.Core import LightSource
from src.Utilities.Sampling import Sampler
from src.Utilities.Scene import Scene
from src.Utilities.Common import unit, attenuate_distance_exponential

@dataclass
class AmbienceSettings:
    enabled: bool = True
    color: Color = field(default_factory=lambda: Color(0.03, 0.03, 0.03, 1.0))
    intensity: float = 0.1

    occlusion_enabled: bool = False

@dataclass
class ShadowSettings:
    enabled: bool = True
    samples: int = 8
    bias: float = 1e-3

@dataclass
class BackgroundSettings:
    enabled: bool = True
    default: Color = field(default_factory=lambda: Color(0.0, 0.0, 1.0, 0.0))
    custom: Optional[Color | ColorGradient | np.ndarray] = field(default_factory=lambda: Color(1.0, 1.0, 1.0, 0.0))

    def get_background_color(self, direction: np.ndarray) -> Color:
        """
        Return the background color based on the ray's direction vector.
        Handles Solid Color, ColorGradient (Skybox), or Texture Map safely.
        """
        if not self.enabled:
            return self.default
        
        type_name = type(self.custom).__name__

        # --- Solid ---
        if type_name == 'Color':
            return self.custom
        
        # --- Skybox --- 
        elif type_name == 'ColorGradient':
            # Resolve Direction
            dir = unit(direction)

            # Map Y [-1, 1] to [0, 1]
            t = 0.5 * (dir[1] + 1.0)
            return self.custom.get_color(t)

        # --- Texture Map ---
        elif isinstance(self.custom, np.ndarray):
            # Resolve Direction (reuse logic or recalculate)
            dir = unit(direction)
            # Ensure we never feed invalid values to asin by normalizing & clamping
            dir = dir / (np.linalg.norm(dir) + 1e-12)
            return self._sample_equirectangular_map(self.custom, dir)

        return self.default
    
    def _sample_equirectangular_map(self, texture: np.ndarray, direction: np.ndarray) -> Color:
        """
        Samples a 2D texture using Spherical (Equirectangular) mapping.
        Texture is assumed to be a numpy array of shape (H, W, 3).
        """
        # Convert 3D Direction -> 2D UV Coordinates
        # u = atan2(z, x) / 2pi + 0.5
        # v = asin(y) / pi + 0.5
        x, y, z = direction

        # Compute u robustly
        u = np.arctan2(z, x) / (2 * np.pi) + 0.5

        # Compute v with safe clamping to avoid domain errors
        y_clamped = float(max(-1.0, min(1.0, y)))
        v = float(np.arcsin(y_clamped)) / np.pi + 0.5
        
        # Map UV to Pixel Coordinates (robust to tiny floating errors)
        height, width, _ = texture.shape

        # Ensure u/v are in [0,1)
        u = u % 1.0
        # Clamp v to [0,1]
        v = max(0.0, min(1.0, v))

        # Convert to pixel indices
        u_idx = int(np.floor(u * width)) % width
        v_idx = int(np.floor(v * height))
        v_idx = max(0, min(height - 1, v_idx)) # Clamp vertical to avoid out of bounds

        # Retrieve pixel (assume float 0-1 or uint8 0-255)
        pixel = texture[v_idx, u_idx]

        # Normalize if the texture is 0-255 (integers)
        if texture.dtype.kind in 'iu': # int or uint
            pixel = pixel / 255.0

        # Defensive: ensure we return 3 components
        pix = np.asarray(pixel, dtype=float)
        if pix.size >= 3:
            return Color(pix[0], pix[1], pix[2], pix[3] if pix.size > 3 else 1.0)
        if pix.size == 1:
            return Color(pix[0], pix[0], pix[0], 1.0)
        # Fallback
        return Color(0.0, 0.0, 0.0, 1.0)

class ShadingStrategy(ABC):
    def __init__(
            self,
            ambience_settings: Optional[AmbienceSettings] = None,
            shadow_settings: Optional[ShadowSettings] = None,
            background_settings: Optional[BackgroundSettings] = None
        ):
        self.ambience_settings = ambience_settings if ambience_settings is not None else AmbienceSettings()
        self.shadow_settings = shadow_settings if shadow_settings is not None else ShadowSettings()
        self.background_settings = background_settings if background_settings is not None else BackgroundSettings() 

    @abstractmethod
    def shade(
        self,
        scene: Scene,
        ray: TracingRay,
        hit_info: HitInfo,
        recursions_left: int,
        trace_function: Callable[[Scene, TracingRay, int, Sampler], Color],
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["RenderStats"] = None
    ) -> Color:
        ...
    
    def _random_point_on_disc(self, center: np.ndarray, normal: np.ndarray, radius: float, sampler: Sampler) -> np.ndarray:
        if abs(normal[1]) > 0.99:
            helper_axis = np.array([1.0, 0.0, 0.0])
        else:
            helper_axis = np.array([0.0, 1.0, 0.0])
            
        tangent = np.cross(helper_axis, normal)
        tangent = unit(tangent)
        bitangent = np.cross(normal, tangent)

        u1 = sampler.next_1d()
        u2 = sampler.next_1d()

        # We use sqrt(u1) to distribute points evenly by area (prevents clustering in the center)
        r = np.sqrt(u1) * radius
        theta = u2 * 2.0 * np.pi
        
        offset = tangent * (r * np.cos(theta)) + bitangent * (r * np.sin(theta))
        return center + offset

class NormalShading(ShadingStrategy):
    """
    Debug shader that maps surface normals to RGB colors.
    
    Usage:
    - Red indicates the normal points right (+X)
    - Green indicates the normal points up (+Y)
    - Blue indicates the normal points forward (+Z)
    
    This is critical for Glass scenes. If a sphere's normal is inverted, 
    the refraction calculations (Snell's Law) will be wrong.
    """
    def shade(self, hit_info: HitInfo, *args, **kwargs) -> Color:
        # 1. Get Normal
        normal = getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0]))
        
        # 2. Map from range [-1, 1] to [0, 1] for color display
        # Normal (0,0,0) becomes Black (0, 0, 0)
        r = np.clip((normal[0]), 0.0, 1.0)
        g = np.clip((normal[1]), 0.0, 1.0)
        b = np.clip((normal[2]), 0.0, 1.0)
        
        return Color(r, g, b)

class DistanceShading(ShadingStrategy):
    """
    Debug shader that visualizes the distance from the camera to the object.
    
    Objects closer than 'min_distance' are White.
    Objects further than 'max_distance' are Black.
    Gradient in between.
    """
    def __init__(
            self,
            min_distance: float = 0.0,
            max_distance: float = 20.0,
            color_gradient: Optional[ColorGradient] = None
        ):
        super().__init__()
        self.min_dist = min_distance
        self.max_dist = max_distance

    def shade(self, hit_info: HitInfo, *args, **kwargs) -> Color:
        dist = hit_info.distance

        if dist < 0:
            return Color(0.0, -1.0, 0.0) # Negative Distance

        range_dist = self.max_dist - self.min_dist
        if range_dist == 0: range_dist = 1.0
        
        normalized = (dist - self.min_dist) / range_dist
        normalized = max(0.0, min(1.0, normalized))
        
        # Close = White (1.0), Far = Black (0.0)
        val = 1.0 - normalized
        
        return Color(val, val, val)

class FlatShading(ShadingStrategy):
    """
    Renders objects with their raw Albedo color only. 
    No lighting, no shadows, no recursion. 
    Fastest possible render mode.
    """
    def shade( self, hit_info: HitInfo, *args, **kwargs) -> Color:
        # Material validation
        material: Optional[PBRMaterial] = getattr(hit_info.obj, 'material', None)
        if material is None:
            return Color(1.0, 0.0, 1.0) # Material Error

        # Just return the base color (Albedo)
        return material.data.albedo

class LambertShading(ShadingStrategy):
    """
    Simple Lambertian shader (Direct Light Only).
    Calculates lighting from scene lights but does NOT recurse for reflections/refractions.
    """
    def shade(
            self,
            scene: "Scene",
            ray: TracingRay,
            hit_info: "HitInfo",
            recursions_left: int,
            trace_function: Callable[[Scene, TracingRay, int, Sampler], Color],
            sampler: Sampler,
            bias: float = 1e-4,
            stats: Optional["RenderStats"] = None
        ) -> Color:

        # Material validation
        material: Optional[PBRMaterial] = getattr(hit_info.obj, 'material', None)
        if material is None:
            return Color(1.0, 0.0, 1.0) # Material Error

        # 1. Setup Geometry
        # We need the View Direction (V) for specular highlights
        view_dir = -unit(ray.orientation) 
        hit_point = hit_info.point
        final_color = Color(0.0, 0.0, 0.0)

        # 2. Handle Emission (Self-Illumination)
        # Even in basic shading, emissive objects should glow.
        if material.type == MaterialType.EMISSIVE:
            return material.get_emissive_component()

        # 3. Iterate Over Scene Lights
        # ----------------------------
        def visibility_fn(point: np.ndarray, light: LightSource) -> float:
            # Calculate direction to this specific light (or sample point)
            # Note: _calculate_shadow_visibility expects light_dir and we pass the hit object to avoid self-shadowing
            light_dir_to_source, _ = light.get_direction_and_dist(point)
            return self._calculate_shadow_visibility(scene, point, light, light_dir_to_source, sampler, exclude_obj=hit_info.obj)

        # 4. Evaluate Direct Lighting
        # The material class already contains the logic to loop over lights
        # and apply the BRDF (Diffuse + Specular).
        
        direct_light = material.evaluate_direct_light(
            scene_lights=scene.get_lights(),
            hit_info=hit_info,
            view_dir=view_dir,
            visibility_function=visibility_fn,
            bias=bias
        )
        
        final_color += direct_light

        # 6. Ambient Light (Optional)
        # Adds a flat base color so shadowed areas aren't pitch black
        if self.ambience_settings.enabled:
            final_color += material.get_ambient_color(self.ambience_settings.color, self.ambience_settings.intensity)

        return final_color

    def _calculate_shadow_visibility(self, scene: Scene, point: np.ndarray, light: LightSource, light_dir: np.ndarray, sampler: Sampler, exclude_obj = None) -> float:
        """
        Calculates what fraction of the light is visible from 'point'.
        Returns 0.0 (Fully Blocked) to 1.0 (Fully Visible).
        """
        if not self.shadow_settings.enabled:
            return 1.0

        radius = getattr(light, "radius", 0.0) or getattr(light, "size", 0.0)
        
        # Case A: Point Light (Hard Shadows)
        if radius <= 0.0 or self.shadow_settings.samples <= 1:
            # Check if a ray from point -> light is blocked
            is_blocked = scene.is_occluded(point, light.position, bias=self.shadow_settings.bias, exclude_obj=exclude_obj)
            return 0.0 if is_blocked else 1.0
        
        # Case B: Area Light (Soft Shadows)
        else:
            visible_count = 0
            for _ in range(self.shadow_settings.samples):
                # Pick a random point on the light source
                # Note: light_dir here is the general direction, but for area lights 
                # we usually sample the disc facing the point.
                sample_pos = self._random_point_on_disc(light.position, -light_dir, float(radius), sampler)
                
                if not scene.is_occluded(point, sample_pos, bias=self.shadow_settings.bias, exclude_obj=exclude_obj):
                    visible_count += 1
            
            return float(visible_count) / float(self.shadow_settings.samples)
        
class RecursiveLambertShading(LambertShading):
    def shade(
            self,
            scene: Scene,
            ray: TracingRay,
            hit_info: HitInfo,
            recursions_left: int,
            trace_function: Callable[[Scene, TracingRay, int, Sampler], Color],
            sampler: Sampler,
            bias: float = 1e-4,
            stats: Optional["RenderStats"] = None
        ) -> Color:

        # Material validation
        material = cast(PBRMaterial, getattr(hit_info.obj, 'material', None))
        if material is None:
            return Color(1.0, 0.0, 1.0) # Material Error

        final_color = Color(0.0, 0.0, 0.0)
        
        # --- 1. Direct Lighing, reuse basic lambert shading
        if material.type == MaterialType.EMISSIVE:
            return material.get_emissive_component()
        
        direct_light = super().shade(scene, ray, hit_info, recursions_left, trace_function, sampler, bias, stats)

        # --- 2. Indirect Lighting (Recursion) ---
        indirect_light = Color(0.0, 0.0, 0.0)
        indirect_light += trace_function(scene, ray, recursions_left, sampler)

        # --- 3. Combine ---
        final_color = direct_light

        return final_color

class VolumetricShading(ShadingStrategy):
    """
    Renders objects based on their volume. 
    Useful for medical visualization, sub-surface scattering approximation, or sci-fi energy shields.
    """
    def __init__(
        self,
        intersection_function: Callable,
        density: float = 1.0,                               # Beer's Law coefficient (Higher = rapid absorption)
        absorption_color: Color = Color(0.2, 0.8, 1.0),     # The color of the object material
        invert_style: bool = True,                          # True = Sci-Fi (Thick is Bright), False = Glass (Thick is Dark)
        rim_power: float = 3.0,                             # Edge highlighting strength (0.0 to disable)
        rim_color: Color = Color(1.0, 1.0, 1.0),            # Color of the edge highlight
        max_thickness: float = 10.0,                        # Clamping value to prevent infinite vals on open meshes
    ):
        self.density = density
        self.absorption_color = absorption_color
        self.invert_style = invert_style
        self.rim_power = rim_power
        self.rim_color = rim_color
        self.max_thickness = max_thickness

        self.intersection_function = intersection_function
        

    def shade(
        self,
        scene: "Scene",
        ray: TracingRay,
        hit_info: "HitInfo",
        bias: float = 1e-4,
        stats: Optional["RenderStats"] = None,
        *args, **kwargs
    ) -> Color:
        # 1. Setup Vectors
        # Normal pointing OUT of the surface
        normal = getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0]))
        hit_point = getattr(hit_info, "point", None)
        
        if hit_point is None: return Color(0.0, 0.0, 0.0)

        view_dir = unit(-ray.orientation)

        # 2. Calculate Thickness (The "Through" Ray)
        # We push the origin slightly INSIDE the object (opposite to normal) to avoid self-intersection at the entry.
        # Note: If geometry is single-sided planes, this might fail. Assumes closed volume.
        inside_origin = hit_point - (normal * bias) 
        
        inside_ray = replace(ray, origin=inside_origin, is_inside=True)      # Mark this ray as originating INSIDE the object so the intersector treats it as an exit ray

        # Find where the ray leaves the object (pass stats along)
        exit_hit = self.intersection_function(scene, inside_ray, stats)

        thickness = 0.0
        if exit_hit and exit_hit.hit:
            thickness = exit_hit.distance
        else:
            thickness = 0.0

        # Clamp thickness for safety
        thickness = min(thickness, self.max_thickness)

        # 3. Apply Beer's Law (Attenuation)
        # Transmission = exp(-density * distance)
        # Result is 1.0 (Thin) to 0.0 (Thick)
        transmission = attenuate_distance_exponential(thickness, self.density)

        # 4. Determine Core Color
        final_color = Color(0.0, 0.0, 0.0)

        if self.invert_style:
            # --- SCI-FI / ADDITIVE STYLE ---
            # Thicker parts glow brighter (like accumulating energy)
            # Intensity = 1.0 - transmission (0.0 at edge, 1.0 at center)
            intensity = 1.0 - transmission
            final_color = self.absorption_color * intensity
        else:
            # --- ABSORPTION / SUBTRACTIVE STYLE ---
            # Thicker parts absorb light (look darker/tinted)
            # This mimics looking through colored glass or liquid.
            # We assume a white background light source for this visualization.
            final_color = self.absorption_color * transmission

        # 5. Add Rim Lighting (Fresnel-like effect)
        # Highlighting edges makes x-ray geometry readable.
        if self.rim_power > 0.0:
            # NdotV: 1.0 looking straight on, 0.0 at glancing angle
            NdotV = max(0.0, np.dot(normal, view_dir))
            
            # Invert: 0.0 at center, 1.0 at edge
            rim_factor = 1.0 - NdotV
            
            # Power curve to tighten the rim
            rim_intensity = rim_factor ** self.rim_power
            
            final_color += self.rim_color * rim_intensity

        return final_color
