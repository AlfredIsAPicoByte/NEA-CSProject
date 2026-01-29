from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional, Callable, List, cast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from src.Data.Sampling.Core import Sampler
from src.Data.Scene import Scene, SceneNode
from src.Data.Color import Color, ColorGradient
from src.Geometry.BVH import BVHNode, build_bvh_tree
from src.Lighting.Core import Light
from src.Material.BSDF import calculate_throughput_weight
from src.Material.Core import PBRMaterial, MaterialType
from src.Utilities.Common import unit, attenuate_distance_exponential, attenuate_distance_coefficents

if TYPE_CHECKING:
    from .Core import TracingStats

@dataclass
class AmbienceSettings:
    enabled: bool = True
    color: Color = field(default_factory=lambda: Color(0.03, 0.03, 0.03, 1.0))
    intensity: float = 0.1

    occlusion_map_enabled: bool = False
    occlusion_sample_count: int = 16
    occlusion_radius: float = 1.0
    occlusion_bias: float = 1e-4

@dataclass
class ShadowSettings:
    enabled: bool = True
    samples: int = 8
    bias: float = 1e-3

    use_light_tree: bool = False # Optimize shadow rays with a BVH over lights

@dataclass
class BackgroundSettings:
    enabled: bool = True
    default_color: Color = field(default_factory=lambda: Color(0.0, 0.0, 1.0, 0.0))
    current_value: Optional[Color | ColorGradient | np.ndarray] = field(default_factory=lambda: Color(1.0, 1.0, 1.0, 0.0))
    environment_effect_enabled: bool = False
    environment_contribution_factor: float = 0.05
    

    def get_background_color(self, direction: np.ndarray) -> Color:
        """
        Return the background color based on the ray's direction vector.
        Handles Solid Color, ColorGradient (Skybox), or Texture Map safely.
        """
        if not self.enabled:
            # Ensure we always return a Color even if default is None or an unexpected type
            if isinstance(self.default_color, Color):
                return self.default_color
            return Color(0.0, 0.0, 0.0, 1.0)
        
        type_name = type(self.current_value).__name__

        # --- Solid ---
        if type_name == 'Color':
            return self.current_value
        
        # --- Skybox --- 
        elif type_name == 'ColorGradient':
            # Resolve Direction
            dir = unit(direction)

            # Map Y [-1, 1] to [0, 1]
            t = 0.5 * (dir[1] + 1.0)
            return self.current_value.get_color(t)

        # --- Texture Map ---
        elif isinstance(self.current_value, np.ndarray):
            # Resolve Direction (reuse logic or recalculate)
            dir = unit(direction)
            # Ensure we never feed invalid values to asin by normalizing & clamping
            dir = dir / (np.linalg.norm(dir) + 1e-12)
            return self._sample_equirectangular_map(self.current_value, dir)

        # Ensure we always return a Color even if default is None or an unexpected type
        return self.default_color if isinstance(self.default_color, Color) else Color(0.0, 0.0, 0.0, 1.0)
    
    def _sample_equirectangular_map(self, texture: np.ndarray, direction: np.ndarray) -> Color:
        """
        Samples a 2D texture using Spherical (Equirectangular) mapping.
        Texture is assumed to be a numpy array of SignedDistanceShape (H, W, 3).
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

@dataclass
class ShadingSettings:
    ambience_settings: AmbienceSettings = field(default_factory=lambda: AmbienceSettings())
    shadow_settings: ShadowSettings = field(default_factory=lambda: ShadowSettings())
    background_settings: BackgroundSettings = field(default_factory=lambda: BackgroundSettings())

class ShadingStrategy(ABC):
    def __init__(self, settings: Optional[ShadingSettings] = None):
        self.settings = settings if settings is not None else ShadingSettings()
        self.ambience_settings = self.settings.ambience_settings
        self.shadow_settings = self.settings.shadow_settings
        self.background_settings = self.settings.background_settings

    @abstractmethod
    def shade(
        self,
        scene: Scene,
        ray: TracingRay,
        hit_info: HitInfo,
        recursions_left: int,
        trace_function: Callable[[Scene, TracingRay, int, Sampler], Color],
        sampler: Sampler,
        intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], Optional[HitInfo]],
        occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool],
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Color:
        """
        Shading function to compute the color at a ray-object intersection point.
        
        :param self: Description
        :param scene: Description
        :type scene: Scene
        :param ray: Description
        :type ray: TracingRay
        :param hit_info: Description
        :type hit_info: HitInfo
        :param recursions_left: Description
        :type recursions_left: int
        :param trace_function: Description
        :type trace_function: Callable[[Scene, TracingRay, int, Sampler], Color]
        :param sampler: Description
        :type sampler: Sampler
        :param intersection_function: Description
        :type intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], Optional[HitInfo]]
        :param occlusion_function: Description
        :type occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool]
        :param bias: Description
        :type bias: float
        :param stats: Description
        :type stats: Optional["TracingStats"]
        :return: Description
        :rtype: Color
        """
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

    def _random_point_in_sphere(self, center: np.ndarray, radius: float, sampler: Sampler) -> np.ndarray:
        u1 = sampler.next_1d()
        u2 = sampler.next_1d()
        u3 = sampler.next_1d()

        r = radius * (u1 ** (1/3))  # Cube root for uniform distribution in volume
        theta = np.arccos(1 - 2 * u2)  # Polar angle
        phi = 2 * np.pi * u3           # Azimuthal angle

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return center + np.array([x, y, z])

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
        # Normal (0,0,0) becomes Grey (0.5, 0.5, 0.5)
        r = np.clip((normal[0] * 0.5) + 0.5, 0.0, 1.0)
        g = np.clip((normal[1] * 0.5) + 0.5, 0.0, 1.0)
        b = np.clip((normal[2] * 0.5) + 0.5, 0.0, 1.0)
        
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
            settings: Optional[ShadingSettings] = None,
            min_distance: float = 0.0,
            max_distance: float = 20.0,
            color_gradient: Optional[ColorGradient] = None
            ):
        super().__init__(settings)
        self.min_dist = min_distance
        self.max_dist = max_distance

        if color_gradient is None:
            # Default: Black (0.0) -> White (1.0)
            self.color_gradient = ColorGradient(
                [Color(0, 0, 0), Color(1, 1, 1)], 
                np.array([0.0, 1.0])
            )
        else:
            self.color_gradient = color_gradient

    def shade(self, hit_info: HitInfo, *args, **kwargs) -> Color:
        dist = hit_info.distance

        if dist < 0:
            return Color(0.0, 1.0, 0.0) # Debug: Negative Distance is usually an error

        # 1. Calculate Range
        range_dist = self.max_dist - self.min_dist
        if range_dist == 0: range_dist = 1.0
        
        # 2. Normalize and Invert in one step
        # Original: normalized = (dist - min) / range; val = 1.0 - normalized
        # Optimized: val = (max - dist) / range
        val = (self.max_dist - dist) / range_dist
        
        # 3. Clamp between 0.0 and 1.0
        # (This handles both 'too close' and 'too far' cases)
        val = max(0.0, min(1.0, val))
        
        return self.color_gradient.get_color(val)

class FlatShading(ShadingStrategy):
    """
    Renders objects with their raw Albedo color only. 
    No lighting, no shadows, no recursion. 
    Fastest possible render mode.
    """
    def shade(self, hit_info: HitInfo, *args, **kwargs) -> Color:
        if not hit_info.hit:
            return Color(0.0, 0.0, 0.0)  # No hit, return black
        
        hit_obj = cast(SceneNode, hit_info.obj)
        
        material: Optional[PBRMaterial] = getattr(hit_obj.context, 'material', None)
        if material is None:
            return Color(1.0, 0.0, 1.0) # Material Error

        if material.data.type == MaterialType.EMISSIVE:
            return material.data.emission_color
        
        if material.data.type == MaterialType.GLASS:
            return Color(1.0, 1.0, 1.0) - material.data.absorption_color
        
        # Just return the base color (Albedo)
        return material.data.albedo
        
class VolumetricShading(ShadingStrategy):
    """
    Renders objects based on their volume. 
    Useful for medical visualization, sub-surface scattering approximation, or sci-fi energy shields.
    """
    def __init__(
        self,
        settings: Optional[ShadingSettings] = None,
        intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], HitInfo] = lambda scene, ray, stats: HitInfo.miss(),
        density: float = 1.0,                               # Beer's Law coefficient (Higher = rapid absorption)
        absorption_color: Color = Color(0.2, 0.8, 1.0),     # The color of the object material
        invert_style: bool = True,                          # True = Sci-Fi (Thick is Bright), False = Glass (Thick is Dark)
        rim_power: float = 3.0,                             # Edge highlighting strength (0.0 to disable)
        rim_color: Color = Color(1.0, 1.0, 1.0),            # Color of the edge highlight
        max_thickness: float = 10.0,                        # Clamping value to prevent infinite vals on open meshes
    ):
        super().__init__(settings)
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
        stats: Optional["TracingStats"] = None,
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

@dataclass
class PhysicalShadingSettings(ShadingSettings):
    """
    Settings specific to physically-based shading strategies.
    Inherits from ShadingSettings and adds more options.
    """
    enable_reflection_caustics: bool = False
    enable_refraction_caustics: bool = False
    enable_ambient_occlusion: bool = False
    enable_bidirectional_scattering: bool = False

class PhysicalShadingStrategy(ShadingStrategy):
    """
    Base class for physically-based shading models.
    Implements common functionality for direct lighting, shadows, ambient occlusion, etc.
    Specific BRDFs (e.g., Lambertian, Cook-Torrance) should inherit from this class.
    """
    _cache_ambient_occlusion_map: Optional[np.ndarray] = None
    _cache_light_tree: Optional[BVHNode] = None

    def __init__(self, settings: Optional[PhysicalShadingSettings] = None):
        self.settings = settings if settings is not None else PhysicalShadingSettings()
        self.ambience_settings = self.settings.ambience_settings
        self.shadow_settings = self.settings.shadow_settings
        self.background_settings = self.settings.background_settings
    
    def _calculate_shadow_visibility(
            self,
            scene: Scene,
            point: np.ndarray,
            light_node: SceneNode,
            sampler: Sampler,
            occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool],
            bias: float = 1e-4,
            exclude_obj: Optional[SceneNode] = None,
            stats: Optional["TracingStats"] = None
        ) -> float:
        """
        Calculates what fraction of the light is visible from 'point'.
        Returns 0.0 (Fully Blocked) to 1.0 (Fully Visible).
        
        :param scene: Description
        :type scene: Scene
        :param point: Description
        :type point: np.ndarray
        :param light: Description
        :type light: SceneNode
        :param sampler: Description
        :type sampler: Sampler
        :param occlusion_function: Description
        :type occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool]
        :param bias: Description
        :type bias: float
        :param exclude_obj: Description
        :type exclude_obj: Optional[SceneNode]
        :param stats: Description
        :type stats: Optional["TracingStats"]
        :return: Description
        :rtype: float
        """
        if not self.shadow_settings.enabled:
            return 1.0
        
        if not isinstance(light_node.context, "Light"):
            return 1.0  # Non-light objects do not cast shadows
        
        light = cast(Light, light_node.context)

        radius = getattr(light, "radius", 0.0) or getattr(light, "size", 0.0)
        occluding_objects = Scene.get_objects_by_types(scene.get_scene_objects_flattened(), ["SDF_Material", "Mesh_Material"])
        
        # Case A: Point SceneNode (Hard Shadows)
        if radius <= 0.0 or self.shadow_settings.samples <= 1:
            # Check if a ray from point -> light is blocked
            is_blocked = occlusion_function(point, light_node.world_transform.position, occluding_objects, bias, exclude_obj, stats)
            return 0.0 if is_blocked else 1.0
        
        # Case B: Flat Area SceneNode (Soft Shadows)
        else:
            visible_count = 0
            for _ in range(self.shadow_settings.samples):
                # Pick a random point on the light source
                # Note: light_dir here is the general direction, but for area lights 
                # we usually sample the disc facing the point.
                sample_pos = self._random_point_on_disc(light_node.world_transform.position, -light.get_direction(light_node.world_transform.position, point), float(radius), sampler)
                
                if not occlusion_function(point, sample_pos, occluding_objects, bias, exclude_obj, stats):
                    visible_count += 1
            
            return float(visible_count) / float(self.shadow_settings.samples)
        
        # Case C: Spherical SceneNode (Soft Shadows)
        # Not implemented here, but could be added similarly.
        
    def _calculate_shadow_visibility_spherical(
            self,
            scene: Scene,
            point: np.ndarray,
            light_node: SceneNode,
            sampler: Sampler,
            occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool],
            bias: float = 1e-4,
            exclude_obj: Optional[SceneNode] = None,
            stats: Optional["TracingStats"] = None
        ) -> float:
        """
        Calculates what fraction of the light is visible from 'point' for spherical lights.
        Returns 0.0 (Fully Blocked) to 1.0 (Fully Visible).
        """
        if not self.shadow_settings.enabled:
            return 1.0
        
        if not isinstance(light_node.context, "Light"):
            return 1.0  # Non-light objects do not cast shadows
        
        light = cast(Light, light_node.context)

        radius = getattr(light, "radius", 0.0) or getattr(light, "size", 0.0)
        occluding_objects = Scene.get_objects_by_types(scene.get_scene_objects_flattened(), ["SDF_Material", "Mesh_Material"])
        
        # Case A: Point SceneNode (Hard Shadows)
        if radius <= 0.0 or self.shadow_settings.samples <= 1:
            # Check if a ray from point -> light is blocked
            is_blocked = occlusion_function(point, light_node.world_transform.position, occluding_objects, bias, exclude_obj, stats)
            return 0.0 if is_blocked else 1.0
        
        # Case B: Spherical SceneNode (Soft Shadows)
        else:
            visible_count = 0
            for _ in range(self.shadow_settings.samples):
                # Pick a random point inside the sphere
                sample_pos = self._random_point_in_sphere(light_node.world_transform.position, float(radius), sampler)
                
                if not occlusion_function(point, sample_pos, occluding_objects, bias, exclude_obj, stats):
                    visible_count += 1
            
            return float(visible_count) / float(self.shadow_settings.samples)

    def _calculate_occlusion_factor(
            self,
            point: np.ndarray,
            normal: np.ndarray,
            objects: List[SceneNode],
            sampler: Sampler,
            occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool],
            stats: Optional["TracingStats"] = None) -> float:
        """
        Generates an ambient occlusion factor based on surrounding geometry.
        Returns 0.0 (Fully Occluded) to 1.0 (Fully Open).
        Assumed the objects can occulude geometry.
        """
        if not self.ambience_settings.enabled or self.ambience_settings.occlusion_sample_count <= 0:
            return 1.0

        occluded_count = 0
        for _ in range(self.ambience_settings.occlusion_sample_count):
            # Sample a random direction in the hemisphere around the normal
            sample_dir = sampler.sample_cosine_hemisphere(normal)
            sample_origin = point + normal * self.ambience_settings.occlusion_bias

            if occlusion_function(sample_origin, sample_dir, objects, self.ambience_settings.occlusion_radius, None, stats):
                occluded_count += 1

        occlusion_factor = 1.0 - (float(occluded_count) / float(self.ambience_settings.occlusion_sample_count))
        return occlusion_factor
    
    def _generate_ambient_occlusion_map(
            self,
            scene: Scene,
            sampler: Sampler,
            intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], Optional[HitInfo]],
            occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool],
            stats: Optional["TracingStats"] = None
        ) -> np.ndarray:
        """
        Precomputes an ambient occlusion map for the entire scene.
        Stores results in self._cache_ambient_occlusion_map for reuse.
        """
        width = scene.camera.width
        height = scene.camera.height

        ao_map = np.zeros((height, width), dtype=float)
        occluding_objects = Scene.get_objects_by_types(scene.get_scene_objects_flattened(), ["SDF_Material", "Mesh_Material"])

        for y in range(height):
            for x in range(width):
                ray = scene.camera.generate_ray(x + 0.5, y + 0.5)
                hit_info = intersection_function(scene, TracingRay(ray.origin, ray.orientation), stats)

                if hit_info and hit_info.hit:
                    point = hit_info.point
                    normal = hit_info.normal
                    ao_factor = self._calculate_occlusion_factor(point, normal, occluding_objects, sampler, occlusion_function, stats)
                    ao_map[y, x] = ao_factor
                else:
                    ao_map[y, x] = 1.0  # No hit means fully open

        self._cache_ambient_occlusion_map = ao_map
        return ao_map

class LambertShading(PhysicalShadingStrategy):
    """
    Simple Lambertian shader (Direct SceneNode Only).
    Calculates lighting from scene lights but does NOT recurse for reflections/refractions.
    """
    def shade(
            self,
            scene: "Scene",
            ray: TracingRay,
            hit_info: "HitInfo",
            sampler: Sampler,
            intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], Optional[HitInfo]],
            occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool],
            bias: float = 1e-4,
            stats: Optional["TracingStats"] = None,
            *args, **kwargs
        ) -> Color:
        # Material validation
        if not hit_info.hit:
            return Color(0.0, 0.0, 0.0)  # No hit, return black
        
        hit_obj = cast(SceneNode, hit_info.obj)
        
        material: Optional[PBRMaterial] = getattr(hit_obj.context, 'material', None)
        if material is None:
            return Color(1.0, 0.0, 1.0) # Material Error

        # 1. Setup Geometry
        # We need the View Direction (V) for specular highlights
        view_dir = -unit(ray.orientation)
        final_color = Color(0.0, 0.0, 0.0)

        # 2. Handle Emission (Self-Illumination)
        # Even in basic shading, emissive objects should glow.
        if material.data.type == MaterialType.EMISSIVE:
            return material.evaluate_emissive_component()

        # 3. Iterate Over Scene Lights
        # ----------------------------
        def visibility_fn(point: np.ndarray, light: SceneNode) -> float:
            # Calculate direction to this specific light (or sample point)
            shadow_factor = self._calculate_shadow_visibility(scene, point, light, sampler, occlusion_function, exclude_obj=hit_info.obj, stats=stats)
            ambient_occlusion = 1.0
            if self.ambience_settings.occlusion_map_enabled:
                if self._cache_ambient_occlusion_map is None:
                    self._cache_ambient_occlusion_map = self._generate_ambient_occlusion_map(
                        scene,
                        sampler,
                        intersection_function,
                        occlusion_function,
                        stats
                    )
                screen_coords = scene.camera.world_to_screen(hit_info.point)
                x_idx = int(round(screen_coords[0]))
                y_idx = int(round(screen_coords[1]))
                ambient_occlusion = self._cache_ambient_occlusion_map[y_idx, x_idx]
            else:
                ambient_occlusion = 1.0

            return shadow_factor * ambient_occlusion
        
        # 4. Evaluate Direct Lighting
        # The material class already contains the logic to loop over lights
        # and apply the BRDF (Diffuse + Specular).
        light_nodes = Scene.get_objects_by_type(scene.get_scene_objects_flattened(), "Light")
        if light_nodes is None or len(light_nodes) == 0:
            return final_color  # No lights in scene
        
        active_light_nodes = [ln for ln in light_nodes if ln.active]
        
        direct_light = material.evaluate_direct_light(
            light_nodes=active_light_nodes,
            hit_info=hit_info,
            view_dir=view_dir,
            visibility_function=visibility_fn,
            bias=bias
        )
        
        final_color += direct_light

        # 5. Ambient SceneNode (Optional)
        # Adds a flat base color so shadowed areas aren't pitch black
        if self.ambience_settings.enabled:
            final_color += material.evaluate_ambient_color(self.ambience_settings.color, self.ambience_settings.intensity)

        return final_color

class RecursiveLambertShading(LambertShading):
    """
    Extends Lambertian shading with recursive reflections and refractions.
    """
    
    a = 3.0
    b = 0.7
    c = 1.0

    def shade(
            self,
            scene: "Scene",
            ray: TracingRay,
            hit_info: "HitInfo",
            recursions_left: int,
            trace_function: Callable[[Scene, TracingRay, int, Sampler], Color],
            sampler: Sampler,
            intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], Optional[HitInfo]],
            occlusion_function: Callable[[np.ndarray, np.ndarray, List[SceneNode], float, Optional[SceneNode], Optional["TracingStats"]], bool],
            bias: float = 1e-4,
            stats: Optional["TracingStats"] = None,
            *args, **kwargs
        ) -> Color:
        if not hit_info.hit:
            return Color(0.0, 0.0, 0.0)  # No hit, return black

        # Start with base Lambertian shading
        final_color = super().shade(
            scene,
            ray,
            hit_info,
            sampler,
            intersection_function,
            occlusion_function,
            bias,
            stats,
            *args,
            **kwargs
        )
        
        hit_obj = cast(SceneNode, hit_info.obj)

        # Material validation
        material: Optional[PBRMaterial] = getattr(hit_obj.context, 'material', None)
        if material is None:
            return final_color  # Material Error already handled in base

        if recursions_left <= 0:
            return final_color
        
        indirect_color = Color(0.0, 0.0, 0.0)

        # Sample indirect lighting contribution
        if material.data.type == MaterialType.GLASS:
            if self.settings.enable_bidirectional_scattering:
                direction, throughput = material.sample_glass_contribution(hit_info.direction, hit_info.normal, sampler, 1.0003)
                pdf = 1.0
            else:
                direction = hit_info.normal
                throughput = material.data.albedo
                pdf = 0.0
        else:
            direction, throughput, pdf = material.sample_indirect_contribution(hit_info.direction, hit_info.normal, sampler)
        
        if pdf > 1e-6 and np.linalg.norm(throughput.to_np_array()[:3]) > 1e-6:
            weighted_throughput = Color(*calculate_throughput_weight(direction, hit_info.normal, throughput, pdf))
            attenuation = attenuate_distance_coefficents(hit_info.distance, self.a, self.b, self.c)
            # Create new ray for indirect bounce
            new_origin = hit_info.point + direction * bias
            indirect_ray = TracingRay(new_origin, direction, is_inside=ray.is_inside)

            # Trace the indirect ray
            bounced_color = trace_function(scene, indirect_ray, recursions_left - 1, sampler)

            # Accumulate indirect contribution
            indirect_color += weighted_throughput * bounced_color
        
        final_color += indirect_color
        return final_color