import numpy as np
import random
import math
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod

from PrimaryStructures import Ray
from Scene import Scene
from Camera import VCamera, CameraType
from Geometry import VObject
from Luminance import Color, Material, LightSource
from Algorithims import Algorithm, RenderStats, register_algorithm

# Define a ray that holds the ray and data
class TracingRay(Ray):
    def __init__(self, origin: np.ndarray, orientation: np.ndarray, name: str = "Ray", **kwargs):
        super().__init__(origin, orientation, name)
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"TracingRay(name={self.name}, origin={self.origin}, orientation={self.orientation})"
    pass

# Strategy interfaces for ray generation, intersection, and shading
class RayGenerator(ABC):
    def __init__(self, rays_per_pixel: int = 1):
        self.rays_per_pixel = max(1, rays_per_pixel)

    """Abstract base class for ray generation strategies."""
    @abstractmethod
    def generate(
        self,
        camera: VCamera,
        region: Optional[Tuple[int, int, int, int]] = None,  # (x1, y1, w, h)
        seed: Optional[int] = None,
    ) -> List[TracingRay]:
        ...

    def camera_rotate_ray(self,
        camera: VCamera,
        u: float, v: float,
    ) -> np.ndarray:
        if camera is None:
            raise ValueError("Camera cannot be None when generating rays.")
        
        # For rays coming from an orthographic camera
        if camera.type == CameraType.ORTHOGRAPHIC:
            direction = camera.transform.forward
            return direction / np.linalg.norm(direction)

        # For rays coming from a perspective camera
        ndc_x, ndc_y = 2 * u - 1, 1 - 2 * v
        half_h = math.tan(math.radians(camera.fov) * 0.5)     # cam.fov = vertical FOV in degrees
        half_w = (camera.width / camera.height) * half_h
        px = ndc_x * half_w
        py = ndc_y * half_h
        direction = (camera.transform.forward + px * camera.transform.right + py * camera.transform.up)
        direction = direction / np.linalg.norm(direction)
        return direction

class IntersectionStrategy(ABC):
    def __init__(
            self,
            epsilon: float = 1e-4,
            max_distance: float = 100.0,
            max_steps: int = 256
        ):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.max_steps = max_steps
    
    @abstractmethod
    def find_hit(
        self,
        scene: Scene,
        ray: TracingRay,
    ) -> Tuple[Optional[VObject], float]:
        ...

class InteractionStrategy(ABC):
    def __init__(
            self,
            bias: float = 1e-4
        ):
        self.bias = bias

    @abstractmethod
    def interact(
        self,
        ray: TracingRay,
        hit_object: VObject,
        hit_point: np.ndarray
    ) -> Optional[TracingRay]:
        ...

class ShadingStrategy(ABC):
    def __init__(
            self,
            ambient_enabled: bool = True,
            ambient_color: Optional[Color] = None,
            ambient_intensity: Optional[float] = None,
            enable_shadows: bool = True,
            shadow_samples: int = 8,
            shadow_bias: float = 1e-3,
        ):
        self.ambient_enabled = ambient_enabled
        self.ambient_color = ambient_color
        self.ambient_intensity = ambient_intensity
        self.enable_shadows = enable_shadows
        self.shadow_samples = max(1, int(shadow_samples))
        self.shadow_bias = float(shadow_bias)

    @abstractmethod
    def shade(
        self,
        scene: Scene,
        ray: TracingRay,
        hit_object: Optional[VObject],
        hit_distance: Optional[float],
        current_depth: int,
        trace_function: Callable,
        interaction_function: Callable,
    ) -> Color:
        ...

# Ray generation implementations
class BasicRayGenerator(RayGenerator):
    """
    Generate basic camera rays without jitter.
    Supports optional region/tile to generate rays for a chunk of the image.
    """
    def generate(
        self,
        camera: VCamera,
        region: Optional[Tuple[int, int, int, int]] = None, # (x1, y1, w, h)
        seed: Optional[int] = None,
    ) -> List[TracingRay]:
        # Simple implementation that generates one ray per pixel without jitter or sampler
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
                for r in range(max(1, self.rays_per_pixel)):
                    u = (x + 0.5) / cam_width
                    v = (y + 0.5) / cam_height
                    orientation = self.camera_rotate_ray(camera, u, v)

                    # For orthographic cameras, the origin must be offset in the image plane
                    if camera.type == CameraType.ORTHOGRAPHIC:
                        half_h = math.tan(math.radians(camera.fov) * 0.5) if hasattr(camera, 'fov') else 0.5
                        half_w = (camera.width / camera.height) * half_h
                        px = (2 * u - 1) * half_w
                        py = (1 - 2 * v) * half_h
                        ndc_x = 2 * u - 1
                        ndc_y = 1 - 2 * v
                        ray_origin = camera.transform.position + px * camera.transform.right + py * camera.transform.up
                    else:
                        ray_origin = camera.transform.position

                    ray = TracingRay(
                        origin=ray_origin,
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
        camera: VCamera,
        region: Optional[Tuple[int, int, int, int]] = None, # (x1, y1, w, h)
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
                for r in range(max(1, self.rays_per_pixel)):
                    u, v = self.jitter_within_pixel(rand, x, y, cam_width, cam_height)

                    orientation = self.camera_rotate_ray(camera, u, v)

                    # For orthographic cameras, the origin must be offset in the image plane
                    if camera.type == CameraType.ORTHOGRAPHIC:
                        half_h = math.tan(math.radians(camera.fov) * 0.5) if hasattr(camera, 'fov') else 0.5
                        half_w = (camera.width / camera.height) * half_h
                        px = (2 * u - 1) * half_w
                        py = (1 - 2 * v) * half_h
                        ray_origin = camera.transform.position + px * camera.transform.right + py * camera.transform.up
                    else:
                        ray_origin = camera.transform.position

                    ray = TracingRay(
                        origin=ray_origin,
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

# Ray intersection implementations
class RayMarchingIntersection(IntersectionStrategy):
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

# Interaction implementations
class SimpleInteraction(InteractionStrategy):
    def interact(self, ray: TracingRay, hit_object: VObject, hit_point: np.ndarray) -> Optional[TracingRay]:
        # Simple interaction that just returns None (no bounce)
        return None

class OpticalMaterialInteraction(InteractionStrategy):
    def interact(self, ray: TracingRay, hit_object: VObject, hit_point: np.ndarray) -> Optional[TracingRay]:
        # 1. Resolve Material
        material = getattr(hit_object, "material", None) or getattr(hit_object.shape, "material", None)
        
        if not material:
            return None

        # 2. Calculate Context (Normal)
        # We need the normal to decide reflection/refraction
        if hasattr(hit_object.shape, "GetNormal"):
            normal = hit_object.shape.GetNormal(hit_point)
        else:
            return None

        # 3. Calculate Origin Offset (Prevent Shadow Acne)
        new_origin = hit_point + (normal * self.bias)
        distance = np.linalg.norm(new_origin - ray.origin)
        if distance < self.bias:
            new_origin = ray.point_at(distance + self.bias)

        # 4. Invoke the Material's Logic (with robust handling)
        current_throughput = getattr(ray, "throughput", Color(1.0, 1.0, 1.0))

        # Attempt to let the material produce the redirected ray & attenuation.
        # If the material implementation raises or returns a malformed orientation,
        # we fall back to a safe perfect-reflection ray.
        try:
            new_ray_geom, current_throughput = material.calculate_optical_redirection(
                incoming_ray=ray,
                surface_normal=normal,
                incoming_color=current_throughput,
                new_origin=new_origin
            )

            # Ensure orientation is a 1D numpy vector of length 3
            try:
                ori = np.asarray(new_ray_geom.orientation, dtype=float).reshape(-1)
                if ori.ndim != 1 or ori.size != 3:
                    raise ValueError("Malformed orientation")
                # normalize
                ori = ori / np.linalg.norm(ori)
                orientation_vec = ori
                origin_vec = np.asarray(new_ray_geom.origin, dtype=float).reshape(-1)
            except Exception:
                # malformed orientation/origin from material -> fallback to reflection
                incoming = np.asarray(ray.orientation, dtype=float).reshape(-1)
                incoming = incoming / np.linalg.norm(incoming)
                reflect_dir = incoming - 2.0 * np.dot(incoming, normal) * normal
                orientation_vec = reflect_dir / np.linalg.norm(reflect_dir)
                origin_vec = new_origin

        except Exception:
            # Material call failed -> fallback to perfect reflection
            incoming = np.asarray(ray.orientation, dtype=float).reshape(-1)
            incoming = incoming / np.linalg.norm(incoming)
            reflect_dir = incoming - 2.0 * np.dot(incoming, normal) * normal
            orientation_vec = reflect_dir / np.linalg.norm(reflect_dir)
            origin_vec = new_origin

        # 5. Convert/Return TracingRay and Update Throughput
        next_ray = TracingRay(
            origin=origin_vec,
            orientation=orientation_vec,
            name=f"{ray.name}_bounce",
            throughput=current_throughput,
            depth=getattr(ray, "depth", 0) + 1
        )

        return next_ray

# Shading implementations
class FlatShading(ShadingStrategy):
    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float, depth: int, trace_function: Callable, interaction_function: Callable) -> Color:
        # Simply return the base color of the material without any lighting
        material: Optional[Material] = getattr(hit_object, "material", None) or getattr(hit_object.shape, "material", None) if hasattr(hit_object, "shape") else None
        
        if not material:
            return Color(1, 0, 1) # Error Pink

        base_color = getattr(material, "base_color", Color(1.0, 1.0, 1.0))
        emissive_color = getattr(material, "emissive_color", Color(0.0, 0.0, 0.0))
        return (base_color + emissive_color).clamp()

class SimpleShading(ShadingStrategy):
    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float, depth: int, trace_function: Callable, interaction_function: Callable) -> Color:
        point = ray.point_at(distance)
        
        # Safely get normal
        if not hasattr(hit_object, "shape") or not hasattr(hit_object.shape, "GetNormal"):
            return Color(1, 0, 1)  # Error Pink
        
        normal = hit_object.shape.GetNormal(point)
        view_dir = -ray.orientation
        
        # Simple diffuse shading with ambient light
        material: Optional[Material] = getattr(hit_object, "material", None) or getattr(hit_object.shape, "material", None) if hasattr(hit_object, "shape") else None
        
        if not material:
            return Color(1, 0, 1) # Error Pink

        ambient_col = self.ambient_color if self.ambient_color is not None else getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03))
        direct_light = material.apply_material_color(scene.get_lights(), point, normal, view_dir, ambient_col, lambda l, l_dir, l_dist: 1.0)

        return direct_light

class BasicLambertShading(ShadingStrategy):
    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float, depth: int, trace_function: Callable, interaction_function: Callable) -> Color:
        point = ray.point_at(distance)
        
        # Safely get normal
        if not hasattr(hit_object, "shape") or not hasattr(hit_object.shape, "GetNormal"):
            return Color(1, 0, 1)  # Error Pink
        
        normal = hit_object.shape.GetNormal(point)
        view_dir = -ray.orientation
        
        # Simple diffuse shading with ambient light
        material: Optional[Material] = getattr(hit_object, "material", None) or getattr(hit_object.shape, "material", None) if hasattr(hit_object, "shape") else None
        
        if not material:
            return Color(1, 0, 1) # Error Pink

        ambient_col = self.ambient_color if self.ambient_color is not None else getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03))
        direct_light = material.apply_material_color(scene.get_lights(), point, normal, view_dir, ambient_col, lambda l, l_dir, l_dist: 1.0)

        return direct_light

class RecursiveLambertShading(ShadingStrategy):
    def _random_point_on_disc(self, center: np.ndarray, normal: np.ndarray, radius: float, seed: Optional[int] = None,) -> np.ndarray:
        if seed is not None:
            rand_gen = random.Random(seed)
        else:
            rand_gen = random.Random()

        # Fix: If normal is parallel to World Up (0,1,0), use X-axis instead to avoid Zero Vector crash.
        if abs(normal[1]) > 0.99:
            helper_axis = np.array([1.0, 0.0, 0.0])
        else:
            helper_axis = np.array([0.0, 1.0, 0.0])
            
        tangent = np.cross(helper_axis, normal)
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)

        u1 = rand_gen.random() 
        u2 = rand_gen.random()

        # We use sqrt(u1) to distribute points evenly by area (prevents clustering in the center)
        r = math.sqrt(u1) * radius
        theta = u2 * 2.0 * math.pi
        
        offset = tangent * (r * math.cos(theta)) + bitangent * (r * math.sin(theta))
        
        return center + offset
    
    def _calculate_shadow_visibility(self, scene: Scene, point: np.ndarray, light: LightSource, light_dir: np.ndarray):
        visibility = 1.0

        if self.enable_shadows:
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

    def shade(self, scene: Scene, ray: TracingRay, hit_object: VObject, distance: float, depth: int, trace_function: Callable, interaction_function: Callable) -> Color:
        """
        Calculates the color of a point on an object, including direct lighting and recursive reflections.
        """
        # --- 1. Setup Geometry Context ---
        point = ray.point_at(distance)
        
        # Safely get normal
        if not hasattr(hit_object, "shape") or not hasattr(hit_object.shape, "GetNormal"):
            return Color(1, 0, 1)  # Error Pink
        
        normal = hit_object.shape.GetNormal(point)
        view_dir = -ray.orientation
        
        # 2. Resolve Material
        material: Optional[Material] = getattr(hit_object, "material", None) or getattr(hit_object.shape, "material", None) if hasattr(hit_object, "shape") else None
        
        if not material:
            return Color(1, 0, 1) # Error Pink

        # --- 3. Direct Lighting (Your existing logic) ---
        # Define shadow check callback
        def check_visibility(light, light_dir, light_dist):
            if not self.enable_shadows: return 1.0
            return self._calculate_shadow_visibility(scene, point, light, light_dir)

        # Calculate local lighting (Diffuse + Specular from light sources)
        if self.ambient_enabled:
            ambient_col = self.ambient_color if self.ambient_color is not None else getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03))
            direct_light = material.apply_material_color(scene.get_lights(), point, normal, view_dir, ambient_col, check_visibility)
        else:
            direct_light = material.apply_material_color(scene.get_lights(), point, normal, view_dir, Color(0.0, 0.0, 0.0), check_visibility)

        # --- 4. Indirect Lighting (Recursive Bounce) ---
        indirect_light = Color(0.0, 0.0, 0.0)

        # Only recurse if we have depth left and if the material is reflective/refractive
        # Fully diffuse materials (glossiness=0, roughness=1) don't contribute to indirect lighting
        if depth > 0:
            glossiness = getattr(material, "glossiness", 0.0)
            roughness = getattr(material, "roughness", 1.0)
            is_transparent = getattr(material, "is_transparent", False)
            
            # Only compute indirect lighting if material is somewhat specular or transparent
            if glossiness > 0.01 or is_transparent:
                try:
                    # A. Probe the material to get the bounce ray
                    # We use a white probe to get the material's pure attenuation color
                    probe_color = Color(1.0, 1.0, 1.0)
                    
                    bounce_ray = interaction_function(ray, hit_object, point)

                    # B. Recursive Trace
                    # This calls 'self._trace_ray' which you passed in
                    incoming_light = trace_function(scene, bounce_ray, depth - 1)

                    # C. Combine
                    # Result = Light from world * Material Attenuation
                    indirect_light = incoming_light * getattr(bounce_ray, "throughput", probe_color)
                except Exception:
                    # If optical redirection fails, skip indirect lighting
                    indirect_light = Color(0.0, 0.0, 0.0)

        # --- 5. Final Composition ---
        final_color = direct_light + indirect_light
        return final_color

class TracingStats(RenderStats):
    rays_traced: int = 0
    hits: int = 0
    misses: int = 0
    bounces: int = 0
    max_depth_reached: int = 0

# Raytracer using strategies
@register_algorithm("raytracer")
class Raytracer(Algorithm):
    def __init__(
        self,
        ray_generator: Optional[RayGenerator] = None,
        intersection_strategy: Optional[IntersectionStrategy] = None,
        interaction_strategy: Optional[InteractionStrategy] = None,
        shading_strategy: Optional[ShadingStrategy] = None,
    ):
        super().__init__()
        self.ray_generator: RayGenerator = ray_generator if ray_generator is not None else JitterRayGenerator()
        self.intersector: IntersectionStrategy = intersection_strategy if intersection_strategy is not None else RayMarchingIntersection()
        self.interactor: InteractionStrategy = interaction_strategy if interaction_strategy is not None else OpticalMaterialInteraction()
        self.shader: ShadingStrategy = shading_strategy if shading_strategy is not None else RecursiveLambertShading()

        self.stats = TracingStats()

    def _trace_ray(self, scene: Scene, ray: TracingRay, depth: int) -> Color:
        """
        The recursive engine. It takes a ray, finds what it hits, and calculates the color.
        If the surface is reflective, this function will be called again by the shader.
        """
        # 1. Base Case: Stop bouncing if we run out of depth
        if depth < 0:
            return Color(0.0, 0.0, 0.0)

        self.stats.rays_traced += 1

        # 2. Intersect: Find the closest object
        hit_obj, dist = self.intersector.find_hit(scene, ray)

        # 3. Hit: If we hit something, ask the shader to calculate color
        if hit_obj:
            self.stats.hits += 1

            return self.shader.shade(scene, ray, hit_obj, dist, depth, self._trace_ray, self.interactor.interact)

        # 4. Miss: If we hit nothing, return background color
        try:
            # Ensure direction is a numpy array for your background logic
            return scene.get_background_color(np.asarray(ray.orientation))
        except Exception:
            return Color(0.0, 0.0, 0.0)

    def render(
            self,
            scene: Scene,
            seed: Optional[int] = None,
            region: Optional[Tuple[int, int, int, int]] = None, # (x1, y1, w, h)
        ) -> List[Color]:
        """
        Render the scene and return pixel colors as a flat list (row-major order).
        """
        # --- Setup ---
        cam = scene.camera

        if cam is None:
            raise ValueError("No camera provided to Raytracer.render")

        cam_w, cam_h = cam.width, cam.height
        total_pixels = cam_w * cam_h
        pixel_accum: List[List[Color]] = [[] for _ in range(total_pixels)]
        
        # Define max bounces (recursion depth)
        MAX_DEPTH = 4 

        def push_pixel_color(x: int, y: int, color: Color) -> None:
            if x is not None and y is not None and 0 <= x < cam_w and 0 <= y < cam_h:
                if region is not None:
                    rx, ry, rw, rh = region
                    if not (rx <= x < rx + rw and ry <= y < ry + rh):
                        return
                
                if color is not None:
                    pixel_accum[y * cam_w + x].append(color)
                else:
                    pixel_accum[y * cam_w + x].append(Color())

        def _gen_rays(region=None, seed=None):
            return self.ray_generator.generate(scene.camera, region=region, seed=seed)

        # --- Render Loop ---
        def process_rays(rays):
            """Helper to process a batch of rays using the trace logic."""
            for ray in rays:
                x = getattr(ray, "pixel_x", None)
                y = getattr(ray, "pixel_y", None)

                # Skip invalid pixels
                if x is None or y is None or x < 0 or x >= cam_w or y < 0 or y >= cam_h or (region is not None and not (region[0] <= x < region[0] + region[2] and region[1] <= y < region[1] + region[3])):
                    continue

                # Use the tracing logic to get the final color
                final_color = self._trace_ray(scene, ray, MAX_DEPTH)
                
                push_pixel_color(x, y, final_color)

        # --- Image rendering ---
        rays = _gen_rays(region=region)
        process_rays(rays)
        print("Completed rendering image.")

        # --- Color accumulation ---
        pixel_colors: List[Color] = []
        for idx in range(total_pixels):
            colors = pixel_accum[idx]
            if colors:
                pixel_colors.append(Color.average_colors(colors))
            else:
                # Fill missing pixels with black
                pixel_colors.append(Color(0, 0, 0)) 

        return pixel_colors

    def __repr__(self):
        return f"Raytracer(ray_generator={self.ray_generator}, intersector={self.intersector}, interactor={self.interactor}, shader={self.shader})"