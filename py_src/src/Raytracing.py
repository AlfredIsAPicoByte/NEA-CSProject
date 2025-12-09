from Scene import Scene
from Camera import VCamera
from Geometry import VObject
from Luminance import LightRay, Color
from Algorithims import Algorithm, register_algorithm
from Sampling import Sampler

import numpy as np
import random
from typing import Any, Optional, List, Tuple
from abc import ABC, abstractmethod

# Strategy interfaces and simple implementations
class RayGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        camera: "VCamera",
        rays_per_pixel: int,
        seed: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,  # (x, y, width, height)
        sampler: Optional[Any] = None,  # SamplingManager or Sampler-compatible
    ) -> List[LightRay]:
        ...

class CameraJitterRayGenerator(RayGenerator):
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
    ) -> List[LightRay]:
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

        rays: List[LightRay] = []
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
                            jitter_u = rand.uniform(-0.5, 0.5) / cam_width
                            jitter_v = rand.uniform(-0.5, 0.5) / cam_height
                            u = (x + 0.5 + jitter_u) / cam_width
                            v = (y + 0.5 + jitter_v) / cam_height

                        orientation = camera.transform.forward + (u - 0.5) * camera.transform.right + (v - 0.5) * camera.transform.up
                        orientation = orientation / np.linalg.norm(orientation)
                        ray = LightRay(
                            origin=camera.transform.position,
                            orientation=orientation,
                            color=Color(1.0, 1.0, 1.0, 1.0),
                            name=f"Camera Ray ({x},{y}) #{r}"
                        )
                        ray.pixel_x = x
                        ray.pixel_y = y
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
                            jitter_u = rand.uniform(-0.5, 0.5) / cam_width
                            jitter_v = rand.uniform(-0.5, 0.5) / cam_height
                            u = (x + 0.5 + jitter_u) / cam_width
                            v = (y + 0.5 + jitter_v) / cam_height

                        orientation = camera.transform.forward + (u - 0.5) * camera.transform.right + (v - 0.5) * camera.transform.up
                        orientation = orientation / np.linalg.norm(orientation)
                        ray = LightRay(
                            origin=camera.transform.position,
                            orientation=orientation,
                            color=Color(1.0, 1.0, 1.0, 1.0),
                            name=f"Camera Ray ({x},{y}) #{r}"
                        )
                        rays.append(ray)
                    continue

                # fallback: random jitter as before
                for r in range(max(1, rays_per_pixel)):
                    jitter_u = rand.uniform(-0.5, 0.5) / cam_width
                    jitter_v = rand.uniform(-0.5, 0.5) / cam_height
                    u = (x + 0.5 + jitter_u) / cam_width
                    v = (y + 0.5 + jitter_v) / cam_height
                    orientation = camera.transform.forward + (u - 0.5) * camera.transform.right + (v - 0.5) * camera.transform.up
                    orientation = orientation / np.linalg.norm(orientation)
                    ray = LightRay(
                        origin=camera.transform.position,
                        orientation=orientation,
                        color=Color(1.0, 1.0, 1.0, 1.0),
                        name=f"Camera Ray ({x},{y}) #{r}"
                    )
                    rays.append(ray)
        return rays

class IntersectionStrategy(ABC):
    @abstractmethod
    def find_hit(self, scene: Any, ray: LightRay) -> Tuple[Optional[VObject], float]:
        ...

class RayMarchingIntersection(IntersectionStrategy):
    def __init__(self, epsilon: float = 1e-4, max_distance: float = 100.0, max_steps: int = 256, attenuation: float = 0.9):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.max_steps = max_steps

        self.attenuation = attenuation
        

    def find_hit(self, scene: Scene, ray: LightRay) -> Tuple[Optional[VObject], float]:
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
            
            ray = ray.AttenuateFactor(self.attenuation / distance_traveled if distance_traveled > 0 else 1.0)

            distance_traveled += distance_to_closest
            if distance_traveled >= self.max_distance:
                return None, float("inf")
        return None, float("inf")

class ShadingStrategy(ABC):
    @abstractmethod
    def shade(self, scene: Scene, ray: LightRay, hit_object: VObject, distance: float) -> Color:
        ...

class BasicLambertShading(ShadingStrategy):
    def shade(self, scene: Scene, ray: LightRay, hit_object: VObject, distance: float) -> Color:
        point = ray.point_at(distance)

        # try common method names for normal
        if hasattr(hit_object.shape, "get_normal"):
            normal = hit_object.shape.get_normal(point)
        elif hasattr(hit_object.shape, "GetNormal"):
            normal = hit_object.shape.GetNormal(point)
        else:
            # fallback normal
            normal = np.array([0.0, 1.0, 0.0])

        color = Color(0.0, 0.0, 0.0, 1.0)
        for light in scene.get_lights():
            light_dir = light.position - point
            light_dist = np.linalg.norm(light_dir)
            if light_dist > 0:
                light_dir = light_dir / light_dist
            intensity = max(0.0, np.dot(normal, light_dir))

            # Support material stored on the VObject or on its Shape
            base = getattr(hit_object, "material", None)
            if base is None and hasattr(hit_object, "shape"):
                base = getattr(hit_object.shape, "material", None)

            if isinstance(base, Color):
                mat_color = base
            elif base is not None and hasattr(base, "base_color"):
                mat_color = base.base_color
            elif base is not None and hasattr(base, "color"):
                mat_color = base.color
            else:
                mat_color = Color(1.0, 1.0, 1.0, 1.0)

            light_int = getattr(light, "intensity", 1.0)
            color = Color(
                color.red + mat_color.red * light.color.red * intensity * light_int,
                color.green + mat_color.green * light.color.green * intensity * light_int,
                color.blue + mat_color.blue * light.color.blue * intensity * light_int,
                1.0,
            )
        return color.clamp()

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
        self.ray_generator = ray_generator if ray_generator is not None else CameraJitterRayGenerator()
        self.intersector = intersection_strategy if intersection_strategy is not None else RayMarchingIntersection()
        self.shader = shading_strategy if shading_strategy is not None else BasicLambertShading()

    def render(self, scene: Scene, camera: "VCamera", seed: Optional[int] = None, tile_size: Optional[Tuple[int,int]] = None, sampler: Optional[Sampler] = None) -> Any:
        """
        Render the scene. Accepts optional sampler (SamplingManager or Sampler) and optional tile_size.
        """
        cam_w, cam_h = camera.width, camera.height
        output_rays: List[LightRay] = []
        hits: list[Tuple[LightRay, VObject, float]] = []
        all_rays: List[LightRay] = []

        if tile_size is None:
            # full-image generation (existing behaviour)
            rays = self.ray_generator.generate(camera, self.rays_per_pixel, seed, region=None, sampler=sampler)
            all_rays = rays
            print(f"Generated {len(rays)} rays for full image.")
            for ray in rays:
                hit_obj, dist = self.intersector.find_hit(scene, ray)
                if hit_obj is None:
                    output_rays.append(ray)
                    print(f"No hit for ray {ray.name}")
                    continue
                hits.append((ray, hit_obj, dist))
                shaded = self.shader.shade(scene, ray, hit_obj, dist)
                ray.color = shaded
                output_rays.append(ray)
                print(f"Hit {hit_obj.name} at distance {dist:.4f} for ray {ray.name}")

            print(f"Completed rendering full image with {len(hits)} hits.")
            return all_rays, hits, output_rays

        # Tile-based processing
        tile_w, tile_h = tile_size
        print(f"Rendering in tiles of size {tile_w}x{tile_h}...")
        for y0 in range(0, cam_h, tile_h):
            for x0 in range(0, cam_w, tile_w):
                w = min(tile_w, cam_w - x0)
                h = min(tile_h, cam_h - y0)
                region = (x0, y0, w, h)
                rays = self.ray_generator.generate(camera, self.rays_per_pixel, seed, region=region, sampler=sampler)
                all_rays.extend(rays)
                print(f"Generated {len(rays)} rays for tile at ({x0},{y0}) size {w}x{h}.")
                for ray in rays:
                    hit_obj, dist = self.intersector.find_hit(scene, ray)
                    if hit_obj is None:
                        output_rays.append(ray)
                        print(f"No hit for ray {ray.name}")
                        continue
                    hits.append((ray, hit_obj, dist))
                    shaded = self.shader.shade(scene, ray, hit_obj, dist)
                    ray = LightRay.from_ray(ray, shaded, ray.intensity)
                    output_rays.append(ray)
                    print(f"Hit {hit_obj.name} at distance {dist:.4f} for ray {ray.name}")
        return all_rays, hits, output_rays

    def __repr__(self):
        return f"Raytracer(rays_per_pixel={self.rays_per_pixel}, intersector={self.intersector}, shader={self.shader})"
