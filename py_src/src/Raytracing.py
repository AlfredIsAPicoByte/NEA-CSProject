from Scene import Scene
from Camera import VCamera
from Geometry import VObject
from Luminance import Color, Material
from Algorithims import Algorithm, register_algorithm
from PrimaryStructures import Ray
from Sampling import Sampler

import numpy as np
import random
from typing import Any, Optional, List, Tuple, Callable
from abc import ABC, abstractmethod

# Define a ray that holds the ray and data
class TracingRay(Ray):
    def __init__(self, origin: np.ndarray, orientation: np.ndarray, name: str = "Ray", **kwargs):
        super().__init__(origin, orientation, name)
        
        for k, v in kwargs.items():
            if hasattr(self, k):
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
        seed: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,  # (x, y, width, height)
        sampler: Optional[Any] = None,  # SamplingManager or Sampler-compatible
    ) -> List[TracingRay]:
        ...

class BasicRayGenerator(RayGenerator):
    def generate(
        self,
        camera: "VCamera",
        rays_per_pixel: int,
        seed: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        sampler: Optional[Sampler] = None,
    ) -> List[TracingRay]:
        # Simple implementation that generates one ray per pixel without jitter
        rays: List[TracingRay] = []
        cam_width, cam_height = camera.width, camera.height
        
        for y in range(cam_height):
            for x in range(cam_width):
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
        seed: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        sampler: Optional[Sampler] = None,
    ) -> List[TracingRay]:
        if seed is not None:
            rand = random.Random(seed)
        else:
            rand = random.Random()
        
        if rays_per_pixel < 1:
            rays_per_pixel = 1

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
                break  # fallback to basic jitter
            break
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
            
            if hasattr(ray, "color"):
                ray.color = Color.attenuate_diatance_max(ray.color, distance_traveled, self.max_distance)
            
            distance_traveled += distance_to_closest
            if distance_traveled >= self.max_distance:
                return None, float("inf")
        return None, float("inf")

class ShadingStrategy(ABC):
    @abstractmethod
    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float) -> Color:
        ...

class BasicLambertShading(ShadingStrategy):
    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float) -> Color:
        point = ray.point_at(distance)

        if hit_object is None:
            return Color()
            
        material = getattr(hit_object.shape, "material", None)
        
        # Emissive surfaces return their color
        if material and hasattr(material, "emissive"):
            emissive_color = material.emissive
            # print(f"[Shade] Emissive surface at {point}: color={emissive_color}")
            return emissive_color
        
        color = Color()  # Start with black
        
        if not material:
            print(f"[Shade] No material at {point}; returning black")
            return color
        
        mat_color = material.color
        print(f"[Shade] Hit at point {point}")
        print(f"  Material color: {mat_color}")
        print(f"  Number of lights in scene: {len(scene.lights)}")
        
        # Direct lighting: process each light
        for light_idx, light in enumerate(scene.lights):
            print(f"  Light {light_idx}: {light.name}")
            print(f"    Position: {light.position}")
            print(f"    Color: {light.color}")
            print(f"    Intensity: {light.intensity}")
            
            # Light direction from hit point to light
            light_vec = light.position - point
            light_dist = np.linalg.norm(light_vec)
            light_dir = light_vec / light_dist if light_dist > 0 else np.array([0, 0, 0])
            
            print(f"    Distance to light: {light_dist}")
            print(f"    Light direction (normalized): {light_dir}")
            
            # Get surface normal (approximation; requires shape to provide normal_at)
            normal = np.array([0, 1, 0])  # TODO: compute from hit_object.shape.normal_at(point)
            print(f"    Surface normal (placeholder): {normal}")
            
            # Lambert's cosine law
            cos_theta = max(0.0, np.dot(normal, light_dir))
            print(f"    cos(theta) = {cos_theta}")
            
            # Contribution = material_color * light_color * intensity * cos_theta
            light_contrib = mat_color * light.color * light.intensity * cos_theta
            print(f"    Contribution: {mat_color} * {light.color} * {light.intensity} * {cos_theta} = {light_contrib}")
            
            color += light_contrib
        
        print(f"  Final shaded color: {color}")
        return color

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

    def render(self, scene: Scene, seed: Optional[int] = None, tile_size: Optional[Tuple[int,int]] = None, sampler: Optional[Sampler] = None) -> List[Color]:
        """
        Render the scene and return pixel colors as a flat list (row-major order).
        Index: pixel_colors[y * camera.width + x] is the color at pixel (x, y).
        """
        cam_w, cam_h = scene.camera.width, scene.camera.height
        
        # Accumulation buffers: per-pixel color sum and sample count
        pixel_accum = {}  # (x, y) -> list of Color values
        
        for y in range(cam_h):
            for x in range(cam_w):
                pixel_accum[(x, y)] = []

        if tile_size is None:
            # Full-image generation
            rays = self.ray_generator.generate(scene.camera, self.rays_per_pixel, seed, region=None, sampler=sampler)

            print(f"Generated {len(rays)} rays for full image.")
            
            for ray in rays:
                hit_obj, dist = self.intersector.find_hit(scene, ray)

                if hasattr(ray, "pixel_x") and hasattr(ray, "pixel_y"):
                    x, y = ray.pixel_x, ray.pixel_y
                
                if hit_obj is None:
                    # Ray missed; use background color
                    try:
                        bg = scene.get_background_color(ray.orientation)
                    except Exception:
                        print("Error getting background color; using black. [full image]")
                        bg = Color()
                    pixel_accum[(x, y)].append(bg)
                else:
                    # Ray hit; compute shaded color
                    shaded = self.shader.shade(scene, ray, hit_obj, dist)
                    pixel_accum[(x, y)].append(shaded)
            
            print(f"Completed rendering full image.")
        else:
            # Tile-based processing
            tile_w, tile_h = tile_size
            print(f"Rendering in tiles of size {tile_w}x{tile_h}...")
            
            for y0 in range(0, cam_h, tile_h):
                for x0 in range(0, cam_w, tile_w):
                    w = min(tile_w, cam_w - x0)
                    h = min(tile_h, cam_h - y0)
                    region = (x0, y0, w, h)
                    rays = self.ray_generator.generate(scene.camera, self.rays_per_pixel, seed, region=region, sampler=sampler)
                    print(f"Generated {len(rays)} rays for tile at ({x0},{y0}) size {w}x{h}.")
                    
                    for ray in rays:
                        hit_obj, dist = self.intersector.find_hit(scene, ray)
                        x, y = ray.pixel_x, ray.pixel_y
                        
                        if hit_obj is None:
                            try:
                                bg = scene.get_background_color(ray.orientation)
                            except Exception:
                                print("Error getting background color; using black. [tile]")
                                bg = Color()
                            pixel_accum[(x, y)].append(bg)
                        else:
                            shaded = self.shader.shade(scene, ray, hit_obj, dist)
                            pixel_accum[(x, y)].append(shaded)
        
        # Average accumulated colors per pixel and build flat list (row-major)
        pixel_colors: List[Color] = []
        for y in range(cam_h):
            for x in range(cam_w):
                colors = pixel_accum[(x, y)]
                if colors:
                    avg_color = colors[0]
                    for c in colors[1:]:
                        avg_color = avg_color + c
                    avg_color = avg_color * (1.0 / len(colors))
                else:
                    # Fallback: background color
                    try:
                        avg_color = scene.get_background_color(ray.orientation)
                    except Exception:
                        print("Error getting background color; using black. [accumilation fallback]")
                        avg_color = Color()
                pixel_colors.append(avg_color)
        
        return pixel_colors

    def __repr__(self):
        return f"Raytracer(rays_per_pixel={self.rays_per_pixel}, intersector={self.intersector}, shader={self.shader})"
