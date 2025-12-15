import numpy as np
import random
import math
from typing import Any, Optional, List, Tuple, Callable
from abc import ABC, abstractmethod

from Scene import Scene
from Camera import VCamera
from Geometry import VObject
from Luminance import Color, Material, LightSource
from Algorithims import Algorithm, register_algorithm
from PrimaryStructures import Ray
from Sampling import Sampler

# Define a ray that holds the ray and data
class TracingRay(Ray):
    def __init__(self, origin: np.ndarray, orientation: np.ndarray, name: str = "Ray", **kwargs):
        super().__init__(origin, orientation, name)
        
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set_interaction_function(self, func: Callable[["TracingRay", VObject, np.ndarray], Tuple[Optional["TracingRay"], Optional["TracingRay"]]]) -> None:
        self.interaction_function = func

    def interact(self, hit_object: VObject, hit_point: np.ndarray) -> Tuple[Optional["TracingRay"], Optional["TracingRay"]]:
        """Invoke the interaction function if set.
           Returns a tuple of (reflected_ray, refracted_ray).
        """
        if hasattr(self, "interaction_function") and callable(self.interaction_function):
            return self.interaction_function(self, hit_object, hit_point)
        return None, None

    def __repr__(self):
        return f"TracingRay(name={self.name}, origin={self.origin}, orientation={self.orientation})"
    pass

# Strategy interfaces and simple implementations
class RayGenerator(ABC):
    """Abstract base class for ray generation strategies."""
    @abstractmethod
    def generate(
        self,
        camera: "VCamera",
        rays_per_pixel: int,
        region: Optional[Tuple[int, int, int, int]] = None,  # (x, y, width, height)
        sampler: Optional[Any] = None,  # SamplingManager or Sampler-compatible
        seed: Optional[int] = None,
    ) -> List[TracingRay]:
        ...

class BasicRayGenerator(RayGenerator):
    def generate(
        self,
        camera: "VCamera",
        rays_per_pixel: int,
        region: Optional[Tuple[int, int, int, int]] = None,
        sampler: Optional[Sampler] = None,
        seed: Optional[int] = None,
    ) -> List[TracingRay]:
        # Simple implementation that generates one ray per pixel without jitter
        cam_width, cam_height = camera.width, camera.height

        # default to full image
        if region is None:
            x_start, y_start, region_w, region_h = 0, 0, cam_width, cam_height
        else:
            x_start, y_start, region_w, region_h = region
            # clamp region to camera dimensions
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            region_w = max(0, min(region_w, cam_width - x_start))
            region_h = max(0, min(region_h, cam_height - y_start))
        
        rays: List[TracingRay] = []
        for y in range(y_start, y_start + region_h):
            for x in range(x_start, x_start + region_w):
                for r in range(max(1, rays_per_pixel)):
                    u = (x + 0.5) / cam_width
                    v = (y + 0.5) / cam_height

                    orientation = camera.transform.forward + (u - 0.5) * camera.transform.right + (v - 0.5) * camera.transform.up
                    orientation = orientation / np.linalg.norm(orientation)
                    ray = TracingRay(
                        origin=camera.transform.position,
                        orientation=orientation,
                        pixel_x=x,
                        pixel_y=y,
                        color=Color(),
                        name=f"Camera Ray ({x},{y}) #{r}"
                    )
                    rays.append(ray)
        return rays

class JitterRayGenerator(RayGenerator):
    """Generate camera rays with per-pixel jitter (anti-aliasing).
       Supports optional region/tile to generate rays for a chunk of the image.
       Accepts an optional sampler or SamplingManager to produce deterministic,
       stratified or quasi-random sample positions per pixel (useful for tiles).
    """
    def generate(
        self,
        camera: "VCamera",
        rays_per_pixel: int,
        region: Optional[Tuple[int, int, int, int]] = None,
        sampler: Optional[Sampler] = None,
        seed: Optional[int] = None,
    ) -> List[TracingRay]:
        if seed is not None:
            rand = random.Random(seed)
        else:
            rand = random.Random()

        cam_width, cam_height = camera.width, camera.height
        # default to full image
        if region is None:
            x_start, y_start, region_w, region_h = 0, 0, cam_width, cam_height
        else:
            x_start, y_start, region_w, region_h = region
            # clamp region to camera dimensions
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            region_w = max(0, min(region_w, cam_width - x_start))
            region_h = max(0, min(region_h, cam_height - y_start))

        rays: List[TracingRay] = []
        for y in range(y_start, y_start + region_h):
            for x in range(x_start, x_start + region_w):
                # If sampler is a SamplingManager, prefer its precomputed/per-pixel list
                if sampler is not None and hasattr(sampler, "get_samples_for_pixel"):
                    samples_list = sampler.get_samples_for_pixel(x, y)
                    for r in range(max(1, rays_per_pixel)):
                        if r < len(samples_list):
                            s = samples_list[r]
                            u = (x + s.u) / cam_width
                            v = (y + s.v) / cam_height
                        else:
                            u, v = self.jitter_within_pixel(rand, x, y, cam_width, cam_height)

                        orientation = camera.transform.forward + (u - 0.5) * camera.transform.right + (v - 0.5) * camera.transform.up
                        orientation = orientation / np.linalg.norm(orientation)
                        ray = TracingRay(
                            origin=camera.transform.position,
                            orientation=orientation,
                            pixel_x=x,
                            pixel_y=y,
                            color=Color(),
                            name=f"Camera Ray ({x},{y}) #{r}"
                        )
                        rays.append(ray)
                    continue

                # If sampler conforms to Sampler interface (start_pixel/next_2d)
                if sampler is not None and hasattr(sampler, "start_pixel") and hasattr(sampler, "next_2d"):
                    try:
                        sampler.start_pixel(x, y)
                    except Exception:
                        pass
                    for r in range(max(1, rays_per_pixel)):
                        try:
                            off_u, off_v = sampler.next_2d()
                            u = (x + off_u) / cam_width
                            v = (y + off_v) / cam_height
                        except Exception:
                            u, v = self.jitter_within_pixel(rand, x, y, cam_width, cam_height)

                        orientation = camera.transform.forward + (u - 0.5) * camera.transform.right + (v - 0.5) * camera.transform.up
                        orientation = orientation / np.linalg.norm(orientation)
                        ray = TracingRay(
                            origin=camera.transform.position,
                            orientation=orientation,
                            pixel_x=x,
                            pixel_y=y,
                            color=Color(),
                            name=f"Camera Ray ({x},{y}) #{r}"
                        )
                        rays.append(ray)
                    continue

                # fallback: no sampler provided; produce jittered rays per pixel
                for r in range(max(1, rays_per_pixel)):
                    u, v = self.jitter_within_pixel(rand, x, y, cam_width, cam_height)
                    orientation = camera.transform.forward + (u - 0.5) * camera.transform.right + (v - 0.5) * camera.transform.up
                    orientation = orientation / np.linalg.norm(orientation)
                    ray = TracingRay(
                        origin=camera.transform.position,
                        orientation=orientation,
                        pixel_x=x,
                        pixel_y=y,
                        color=Color(),
                        name=f"Camera Ray ({x},{y}) #{r}"
                    )
                    rays.append(ray)
        return rays
    
    def jitter_within_pixel(self, rand: random.Random, x: int, y: int, cam_width: int, cam_height: int) -> Tuple[float, float]:
        jitter_u = rand.uniform(-0.5, 0.5) / cam_width
        jitter_v = rand.uniform(-0.5, 0.5) / cam_height
        u = (x + 0.5 + jitter_u) / cam_width
        v = (y + 0.5 + jitter_v) / cam_height
        return u, v

class IntersectionStrategy(ABC):
    @abstractmethod
    def find_hit(self, scene: Any, ray: TracingRay) -> Tuple[Optional[VObject], float]:
        ...

class BasicRayIntersection(IntersectionStrategy):
    def find_hit(self, scene: Scene, ray: TracingRay) -> Tuple[Optional[VObject], float]:
        closest_object = None
        closest_distance = float("inf")

        for obj in scene.objects:
            try:
                distance = obj.shape.intersect(ray)
                if distance is not None and 0 < distance < closest_distance:
                    closest_distance = distance
                    closest_object = obj
            except Exception:
                continue

        if closest_object is not None:
            return closest_object, closest_distance
        else:
            return None, float("inf")

class RayMarchingIntersection(IntersectionStrategy):
    def __init__(self, epsilon: float = 1e-4, max_distance: float = 100.0, max_steps: int = 256):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.max_steps = max_steps

    def find_hit(self, scene: Scene, ray: TracingRay) -> Tuple[Optional[VObject], float]:
        distance_traveled = 0.0

        for _ in range(self.max_steps):
            point = ray.point_at(distance_traveled)
            dist_obj = scene.distance_estimator(point)
            # support either (dist, obj) or single distance return
            if isinstance(dist_obj, tuple):
                distance_to_closest, closest_object = dist_obj
            else:
                distance_to_closest, closest_object = dist_obj, None
            if distance_to_closest <= self.epsilon:
                return closest_object, distance_traveled
            
            distance_traveled += distance_to_closest
            if distance_traveled >= self.max_distance:
                return None, float("inf")
        return None, float("inf")

class ShadingStrategy(ABC):
    @abstractmethod
    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float) -> Color:
        ...

class BasicLambertShading(ShadingStrategy):
    def __init__(self, enable_shadows: bool = True, shadow_samples: int = 8, shadow_bias: float = 1e-4):
        self.enable_shadows = enable_shadows
        self.shadow_samples = max(1, int(shadow_samples))
        self.shadow_bias = float(shadow_bias)

    def _random_point_on_disc(self, center: np.ndarray, normal: np.ndarray, radius: float, seed: Optional[int] = None,) -> np.ndarray:
        if seed is not None:
            rand = random.Random(seed)
        else:
            rand = random.Random()

        # build orthonormal basis around normal
        up = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(up, normal)
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        # sample uniformly on disk
        r = math.sqrt(random.random()) * radius
        theta = random.random() * 2.0 * math.pi
        offset = tangent * (r * math.cos(theta)) + bitangent * (r * math.sin(theta))
        return center + offset
    
    def _calculate_shadow_visibility(self, scene: Scene, point: np.ndarray, light: LightSource, light_dir: np.ndarray):
        visibility = 1.0
        if self.enable_shadows:
            # ... (Existing Hard/Soft Shadow Calculation Logic goes here) ...
                
            # Placeholder for your complex shadow logic:
            # visibility = self._calculate_shadow_visibility(scene, point, light, light_dir)
            
            # --- START Existing Shadow Logic ---
            radius = getattr(light, "radius", 0.0) or getattr(light, "size", 0.0)
            if not radius or self.shadow_samples == 1:
                # single test for occlusion
                occluded = scene.is_occluded(point, light.position, bias=self.shadow_bias)
                visibility = 0.0 if occluded else 1.0
            else:
                # soft shadow by sampling area light
                visible_count = 0
                for _ in range(self.shadow_samples):
                    sample_pos = self._random_point_on_disc(light.position, -light_dir, float(radius))
                    if not scene.is_occluded(point, sample_pos, bias=self.shadow_bias):
                        visible_count += 1
                visibility = visible_count / float(self.shadow_samples)

        return visibility
        

    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float) -> Color:
        # 1. Geometry Context
        point = ray.point_at(distance)
        normal = hit_object.shape.GetNormal(point)
        view_dir = -ray.orientation
        
        # 2. Resolve Material
        material: Material | None = getattr(hit_object, "material", None) or getattr(hit_object.shape, "material", None)
        
        if material is None:
            return Color(1,0,1) # Or fallback color

        # 3. Define the Visibility Callback
        # The Material calls this to ask "Is this light visible?"
        def check_visibility(light, light_dir, light_dist):
            if not self.enable_shadows: return 1.0
            return self._calculate_shadow_visibility(scene, point, light, light_dir)

        # 4. Delegate to Material
        # The loop happens inside here!
        return material.apply_material_color(scene.get_lights(), point, normal, view_dir, Color(), check_visibility)

# Raytracer using strategies
@register_algorithm("raytracer")
class Raytracer(Algorithm):
    def __init__(
        self,
        rays_per_pixel: int = 1,
        ray_generator: Optional[RayGenerator] = None,
        intersection_strategy: Optional[IntersectionStrategy] = None,
        shading_strategy: Optional[ShadingStrategy] = None
    ):
        super().__init__()
        self.rays_per_pixel = max(1, rays_per_pixel)
        self.ray_generator = ray_generator if ray_generator is not None else BasicRayGenerator()
        self.intersector = intersection_strategy if intersection_strategy is not None else RayMarchingIntersection()
        self.shader = shading_strategy if shading_strategy is not None else BasicLambertShading()

    def render(self, scene: Scene, camera: Optional[VCamera] = None, seed: Optional[int] = None, tile_size: Optional[Tuple[int,int]] = None, sampler: Optional[Sampler] = None) -> List[Color]:
        """
        Render the scene and return pixel colors as a flat list (row-major order).
        Backward compatible: If the second positional argument passed is a VCamera instance,
        it will be used; otherwise we treat the second arg as the seed (legacy calls).
        """
        # Backwards-compatible camera/seed handling:
        if isinstance(camera, VCamera):
            cam = camera
        elif camera is None:
            cam = scene.camera
        else:
            # camera param actually used as seed in old call signature: render(scene, camera_as_seed, ...)
            seed = camera
            cam = scene.camera

        if cam is None:
            raise ValueError("No camera provided to Raytracer.render and scene.camera is None")

        cam_w, cam_h = cam.width, cam.height

        # Use a flat list for accum: index = y * cam_w + x
        total_pixels = cam_w * cam_h
        pixel_accum: List[List[Color]] = [[] for _ in range(total_pixels)]

        # Helper to push a color for pixel coordinates (x,y) if valid
        def push_pixel_color(x: int, y: int, color: Color) -> None:
            if x is None or y is None:
                return
            if x < 0 or x >= cam_w or y < 0 or y >= cam_h:
                return
            pixel_accum[y * cam_w + x].append(color)

        # Adjust ray generation call to pass the actual 'cam' camera
        def _gen_rays_for_region(region):
            return self.ray_generator.generate(cam, self.rays_per_pixel, region=region, sampler=sampler, seed=seed)

        if tile_size is None:
            rays = _gen_rays_for_region(None)
            print(f"Generated {len(rays)} rays for full image.")
            for ray in rays:
                hit_obj, dist = self.intersector.find_hit(scene, ray)

                x = getattr(ray, "pixel_x", None)
                y = getattr(ray, "pixel_y", None)

                # Skip if pixel coordinates missing/out of bounds
                if x is None or y is None or x < 0 or x >= cam_w or y < 0 or y >= cam_h:
                    continue

                if hit_obj is None:
                    try:
                        bg = scene.get_background_color(np.asarray(ray.orientation))
                    except Exception:
                        bg = Color()
                    push_pixel_color(x, y, bg)
                else:
                    shaded = self.shader.shade(scene, ray, hit_obj, dist)
                    push_pixel_color(x, y, shaded)

            print("Completed rendering full image.")
        else:
            # Tile-based processing
            tile_w, tile_h = tile_size
            print(f"Rendering in tiles of size {tile_w}x{tile_h}...")
            for y0 in range(0, cam_h, tile_h):
                for x0 in range(0, cam_w, tile_w):
                    w = min(tile_w, cam_w - x0)
                    h = min(tile_h, cam_h - y0)
                    region = (x0, y0, w, h)
                    rays = _gen_rays_for_region(region)
                    print(f"Generated {len(rays)} rays for tile at ({x0},{y0}) size {w}x{h}.")
                    for ray in rays:
                        hit_obj, dist = self.intersector.find_hit(scene, ray)
                        x = getattr(ray, "pixel_x", None)
                        y = getattr(ray, "pixel_y", None)
                        if x is None or y is None or x < 0 or x >= cam_w or y < 0 or y >= cam_h:
                            continue
                        if hit_obj is None:
                            try:
                                bg = scene.get_background_color(np.asarray(ray.orientation))
                            except Exception:
                                bg = Color()
                            push_pixel_color(x, y, bg)
                        else:
                            shaded = self.shader.shade(scene, ray, hit_obj, dist)
                            push_pixel_color(x, y, shaded)

        # Build final colors (row-major), use background where empty
        pixel_colors: List[Color] = []
        for y in range(cam_h):
            for x in range(cam_w):
                idx = y * cam_w + x
                colors = pixel_accum[idx]
                if colors:
                    avg_color = Color.average_colors(colors)
                else:
                    try:
                        avg_color = scene.get_background_color(cam.transform.forward)
                    except Exception:
                        avg_color = Color()
                pixel_colors.append(avg_color)

        # Sanity check and pad if missing (very unlikely with above logic)
        if len(pixel_colors) != total_pixels:
            print(f"Warning: pixel_colors length {len(pixel_colors)} != expected {total_pixels}, padding with background color.")
            while len(pixel_colors) < total_pixels:
                try:
                    bg = scene.get_background_color(cam.transform.forward)
                except Exception:
                    bg = Color()
                pixel_colors.append(bg)

        return pixel_colors

    def __repr__(self):
        return f"Raytracer(rays_per_pixel={self.rays_per_pixel}, intersector={self.intersector}, shader={self.shader})"
