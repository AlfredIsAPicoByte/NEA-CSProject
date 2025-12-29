import numpy as np
import math
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod

from PrimaryStructures import HitInfo, Ray
from Scene import Scene
from Camera import VCamera, CameraType
from Geometry import VObject
from Reflections import calculate_reflection_vector
from Refractions import REFRACTIVE_INDICES
from Luminance import Color, Material, MaterialType, LightSource, calculate_redirection_ray, attenuate_color, attenuate_sqr_distance
from RenderingAlgorithims import Algorithm, RenderStats, register_algorithm
from Sampling import SamplingManager, SampleSettings, Sampler, Sample, RandomSampler, reconstruct_pixel

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
    def __init__(
            self,
            gen_sampler: Sampler = RandomSampler(),
            rays_per_pixel: int = 1
        ):
        self.gen_sampler = gen_sampler
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

    def _camera_rotate_ray(self,
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

        fov_scale = math.tan(math.radians(camera.fov * 0.5))

        camera_x = ndc_x * camera.aspect.value * fov_scale
        camera_y = ndc_y * fov_scale
        camera_z = 1.0

        local_dir = np.array([camera_x, camera_y, camera_z], dtype=float)
        world_dir = (
            (local_dir[0] * camera.transform.right) +
            (local_dir[1] * camera.transform.up) +
            (local_dir[2] * camera.transform.forward)
        )

        world_dir = world_dir / np.linalg.norm(world_dir)
        return world_dir

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
    ) -> HitInfo:
        ...

class InteractionStrategy(ABC):
    def __init__(
            self,
            surface_sampler: Sampler = RandomSampler(),
            refractive_index: float = REFRACTIVE_INDICES["air"],
            bias: float = 1e-4
        ):
        self.surface_sampler = surface_sampler
        self.scene_ior = refractive_index
        self.bias = bias

    @abstractmethod
    def interact(self, ray: TracingRay, hit_info: HitInfo) -> Optional[TracingRay]:
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
        self.shadow_samples = shadow_samples
        self.shadow_bias = float(shadow_bias)

    @abstractmethod
    def shade(
        self,
        scene: Scene,
        ray: TracingRay,
        hit_info: HitInfo,
        current_depth: int,
        trace_function: Callable[[Scene, TracingRay, int], Color],
        interaction_function: Callable[[TracingRay, HitInfo, Optional[float]], Optional[float]],
        seed: Optional[int] = None
    ) -> Color:
        ...
    
    def _random_point_on_disc(self, center: np.ndarray, normal: np.ndarray, radius: float, seed: Optional[int] = None,) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # Fix: If normal is parallel to World Up (0,1,0), use X-axis instead to avoid Zero Vector crash.
        if abs(normal[1]) > 0.99:
            helper_axis = np.array([1.0, 0.0, 0.0])
        else:
            helper_axis = np.array([0.0, 1.0, 0.0])
            
        tangent = np.cross(helper_axis, normal)
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)

        u1 = rng.random() 
        u2 = rng.random()

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

# Ray generation implementations
class JitterRayGenerator(RayGenerator):
    """
    Generate camera rays with per-pixel jitter (anti-aliasing).
    - Iterates over the requested region.
    - Ask the Sampler for (u,v) offsets for every pixel.
    - Calculates the Ray Origin and Direction based on Camera Type.
    """
    def generate(
        self,
        camera: VCamera,
        region: Optional[Tuple[int, int, int, int]] = None, # (x, y, w, h)
        seed: Optional[int] = None,
    ) -> List[TracingRay]:
        cam_width, cam_height = camera.width, camera.height
        
        # 1. Resolve Region
        if region is None:
            x_start, y_start, region_w, region_h = 0, 0, cam_width, cam_height
        else:
            x_start, y_start, req_w, req_h = region
            # Clamp to image bounds
            x_start = max(0, min(x_start, cam_width))
            y_start = max(0, min(y_start, cam_height))
            region_w = max(0, min(req_w, cam_width - x_start))
            region_h = max(0, min(req_h, cam_height - y_start))

        rays: List[TracingRay] = []

        # 2. Iterate over pixels
        for y in range(y_start, y_start + region_h):
            for x in range(x_start, x_start + region_w):
                
                # 3. Get Samples
                # The SamplingManager returns a list of Sample objects (offsets 0.0-1.0)
                # matching the configured Samples Per Pixel (SPP).
                pixel_samples = self.gen_sampler.get_samples_for_pixel(x, y)

                for i, sample in enumerate(pixel_samples):
                    # Calculate Global Normalized Coordinates [0, 1]
                    # We add the sample offset (0.0 to 1.0) to the integer pixel coordinate
                    u = (x + sample.u) / cam_width
                    v = (y + sample.v) / cam_height
                    
                    # 4. Calculate Ray Geometry
                    if camera.type == CameraType.ORTHOGRAPHIC:
                        # --- ORTHOGRAPHIC ---
                        # Direction: Always straightforward (Camera Forward)
                        # Origin: Shifts across the image plane based on (u, v)
                        
                        # Calculate physical dimensions of the sensor/plane
                        # If 'orthographic_scale' isn't present, assume FOV-based scaling or default
                        ortho_scale = getattr(camera, 'orthographic_scale', 1.0)
                        aspect_ratio = cam_width / cam_height
                        
                        plane_height = ortho_scale
                        plane_width = plane_height * aspect_ratio

                        # Map u,v (0..1) to Plane Coordinates (-Width/2 .. +Width/2)
                        px = (u - 0.5) * plane_width
                        py = (0.5 - v) * plane_height # Flip Y if needed for standard coordinate systems

                        # Origin = CameraPos + (Right * px) + (Up * py)
                        ray_origin = (
                            camera.transform.position + 
                            (camera.transform.right * px) + 
                            (camera.transform.up * py)
                        )
                        ray_orientation = camera.transform.forward

                    else:
                        # --- PERSPECTIVE ---
                        # Origin: Always the camera position (pinhole)
                        # Direction: Diverges from origin through the pixel
                        ray_origin = camera.transform.position
                        ray_orientation = self._camera_rotate_ray(camera, u, v)

                    # 5. Build Ray
                    ray = TracingRay(
                        origin=ray_origin,
                        orientation=ray_orientation,
                        pixel_x=x,
                        pixel_y=y,
                        color=Color(), # Start black/empty
                        name=f"Ray ({x},{y}) #{i}",
                        throughput=Color(1.0, 1.0, 1.0) # Used for path tracing accumulation
                    )
                    rays.append(ray)

        return rays

# Ray intersection implementations
class RayMarchingIntersection(IntersectionStrategy):
    def find_hit(self, scene: Scene, ray: TracingRay) -> HitInfo:
        distance_traveled = 0.0

        for _ in range(self.max_steps):
            point = ray.point_at(distance_traveled)
            closest_object, distance_to_closest = scene.distance_estimator(point, ignore=getattr(ray, "previous_obj", None))

            if closest_object is None and closest_object is not getattr(ray, "previous_obj", None): # ignore the object if it is the previous
                break

            surface_normal = closest_object.shape.GetNormal(point) if closest_object and hasattr(closest_object, "shape") and hasattr(closest_object.shape, "GetNormal") else np.array([0.0, 1.0, 0.0])
            
            if distance_to_closest <= self.epsilon:
                return HitInfo(
                    did_hit=True,
                    hit_point=point,
                    incoming_direction=None,
                    surface_normal=surface_normal,
                    distance=distance_traveled,
                    obj=closest_object,
                )
            
            distance_traveled += distance_to_closest
            if distance_traveled >= self.max_distance:
                break

        return HitInfo(False)

# Interaction implementations
class SimpleMaterialInteraction(InteractionStrategy):
    def interact(self, ray: TracingRay, hit_info: HitInfo, seed: Optional[int]) -> Optional[TracingRay]:
        # 1. Resolve Material
        v_object: VObject = hit_info.object
        
        material: Material = getattr(v_object.shape, "material", None)
        if not material:
            return None

        # 2. Calculate Context (Normal)
        # We need the normal to decide reflection/refraction
        if hasattr(v_object.shape, "GetNormal"):
            normal: np.ndarray = v_object.shape.GetNormal(hit_info.point)
        else:
            normal = np.array([0.0, 1.0, 0.0])

        # 3. Calculate Origin Offset (Prevent Shadow Acne)
        new_origin = hit_info.point + (normal * self.bias)
        distance = np.linalg.norm(new_origin - ray.origin)
        if distance < self.bias:
            new_origin = ray.point_at(distance + self.bias)

        # 4. Invoke the Material's Logic (with robust handling)
        current_throughput = getattr(ray, "throughput", Color(1.0, 1.0, 1.0))
        pdf = 0

        # Attempt to let the material produce the redirected ray & attenuation.
        # If the material implementation raises or returns a malformed orientation,
        # we fall back to a safe perfect-reflection ray.
        try:
            new_ray, pdf, did_reflect, is_inside = calculate_redirection_ray(
                incoming_ray=ray,
                surface_normal=normal,
                new_origin=new_origin,
                roughness=material.roughness,
                sampler=self.surface_sampler,
                current_refactive_index=self.scene_ior,
                incoming_refactive_index=material.ior,
                seed=seed,
                bias=self.bias,
                fast=False
            )

            if not did_reflect and is_inside:
                current_throughput = material.get_volumetric_component(current_throughput, hit_info.distance)
            
            if material.is_transparent:
                current_throughput = attenuate_color(material.get_emissive_component(), attenuate_sqr_distance(hit_info.distance))

        except Exception:
            new_ray = Ray(
                origin=new_origin,
                orientation=calculate_reflection_vector(normal, ray.orientation)
            )

        # 5. Convert/Return TracingRay and Update Throughput     
        next_ray = TracingRay(
            origin=new_ray.origin,
            orientation=new_ray.orientation,
            name=f"{ray.name}_bounce",
            throughput=current_throughput,
            pdf=pdf,
            previous_obj=v_object,
            depth=getattr(ray, "depth", 0) + 1
        )

        return next_ray

# Shading implementations
class RecursiveLambertShading(ShadingStrategy):
    def shade(
            self,
            scene: Scene,
            ray: TracingRay,
            hit_info: HitInfo,
            depth: int,
            trace_function: Callable[[Scene, TracingRay, int, Optional[int]], Color],
            interaction_function: Callable[[TracingRay, HitInfo, Optional[int]], Optional[TracingRay]],
            seed: Optional[int]
        ) -> Color:
        """
        Calculates the color of a point on an object, including direct lighting and recursive reflections.
        """
        # --- 1. Setup Geometry Context ---
        point = ray.point_at(hit_info.distance)
        v_object: VObject = hit_info.object
        
        # Safely get normal
        if not hasattr(v_object, "shape") or not hasattr(v_object.shape, "GetNormal"):
            return Color(1, 0, 1)  # Error Pink

        normal = v_object.shape.GetNormal(point)
        view_dir = -ray.orientation
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # 2. Resolve Material
        material: Optional[Material] = getattr(v_object.shape, "material", None) if hasattr(v_object, "shape") else None
        
        if not material:
            return Color(1, 0, 1) # Error Pink

        # --- 3. Direct Lighting (Your existing logic) ---
        # Define shadow check callback
        def check_visibility(hit_point, light_pos):
            if not self.enable_shadows:
                return 1.0
            occluded = scene.is_occluded(hit_point, light_pos, bias=self.shadow_bias)
            return 0.0 if occluded else 1.0

        # Calculate local lighting (Diffuse + Specular from light sources)
        direct_light = material.apply_material(scene.get_lights(), hit_info, view_dir, check_visibility)
        
        if self.ambient_enabled:
            ambient_col = self.ambient_color if self.ambient_color is not None else getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03))
            ambient_intensity = self.ambient_intensity if self.ambient_intensity is not None else getattr(scene, "ambient_intensity", 0.1)
            direct_light += material.apply_ambient_color(normal, view_dir, ambient_col, ambient_intensity)
            
        # --- 4. Indirect Lighting (Recursive Bounce) ---
        indirect_light = Color(0.0, 0.0, 0.0)

        # Only recurse if we have depth left and if the material is reflective/refractive
        # Fully diffuse materials (glossiness=0, roughness=1) don't contribute to indirect lighting
        if depth > 0:
            roughness = getattr(material, "roughtness", 0)
            is_transparent = getattr(material, "is_transparent", False)
            
            # Only compute indirect lighting if material is transparent or is not completely diffuse
            if is_transparent or roughness < 0.99:
                # A. Probe the material to get the bounce ray
                # We use a white probe to get the material's pure attenuation color
                probe_color = Color(1.0, 1.0, 1.0) # use scene probes in the future
                    
                new_ray = interaction_function(ray, hit_info, seed)
                incoming_light = Color(0.0, 0.0, 0.0)
                if new_ray is not None:
                    # B. Recursive Trace
                    # This calls 'self._trace_ray' which you passed in
                    incoming_light = trace_function(scene, new_ray, depth - 1, seed)

                # C. 
                pdf = getattr(new_ray, "pdf", 0)
                throughput = getattr(new_ray, "throughput", probe_color)
                brdf_val = material.evaluate_brdf(normal, view_dir, new_ray.orientation)
                cos_theta = max(np.dot(normal, new_ray.orientation), 0.0)
                if pdf > 0:
                    throughput_factor = (brdf_val * cos_theta) / pdf
                else:
                    throughput_factor = 0

                # C. Combine
                # Result = Light from world * Material Attenuation
                indirect_light = incoming_light * throughput * throughput_factor

        # --- 5. Final Composition ---
        final_color = direct_light + indirect_light
        return final_color

# Stats for ray tracing
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
        sampling_manager: Optional[SamplingManager] = None,
        ray_generator: Optional[RayGenerator] = None,
        intersection_strategy: Optional[IntersectionStrategy] = None,
        interaction_strategy: Optional[InteractionStrategy] = None,
        shading_strategy: Optional[ShadingStrategy] = None,
        sample_settings: Optional[SampleSettings] = None,
    ):
        super().__init__()
        self.sampling_manager = sampling_manager
        self.sample_settings = sample_settings or SampleSettings()

        self.ray_generator: RayGenerator = ray_generator if ray_generator is not None else JitterRayGenerator()
        self.intersector: IntersectionStrategy = intersection_strategy if intersection_strategy is not None else RayMarchingIntersection()
        self.interactor: InteractionStrategy = interaction_strategy if interaction_strategy is not None else SimpleMaterialInteraction()
        self.shader: ShadingStrategy = shading_strategy if shading_strategy is not None else RecursiveLambertShading()

        self.stats = TracingStats()

    def _trace_ray(self, scene: Scene, ray: TracingRay, depth: int, seed: Optional[int]) -> Color:
        """
        The recursive engine. It takes a ray, finds what it hits, and calculates the color.
        If the surface is reflective, this function will be called again by the shader.
        """
        if depth < 0:
            return Color(0.0, 0.0, 0.0)

        self.stats.rays_traced += 1

        hit = self.intersector.find_hit(scene, ray)

        if hit.object is not None and hit.hit:
            self.stats.hits += 1
            return self.shader.shade(scene, ray, hit, depth, self._trace_ray, self.interactor.interact, seed)

        # The ray missed all scene objects
        return scene.get_background_color(np.asarray(ray.orientation))

    def render(
        self,
        scene: Scene,
        seed: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Color]:
        cam = scene.camera
        if cam is None:
            raise ValueError("No camera provided")

        cam_w, cam_h = cam.width, cam.height
        MAX_DEPTH = 4
        
        self.sample_settings.width = cam_w
        self.sample_settings.height = cam_h

        # FIX 1: Storage - Use a simplified structure if possible, but we will stick 
        # to your list logic for now. 
        # Warning: High SPP will crash memory here.
        pixel_samples_and_colors = [[] for _ in range(cam_w * cam_h)]

        def _gen_rays(region=None, seed=None):
            return self.ray_generator.generate(scene.camera, region=region, seed=seed)

        def process_rays(rays: List[TracingRay]):
            for ray in rays:
                # SKIP invalid rays
                if ray is None: continue

                x = getattr(ray, "pixel_x", -1)
                y = getattr(ray, "pixel_y", -1)
                
                # Bounds check
                if not (0 <= x < cam_w and 0 <= y < cam_h):
                    continue

                # Region check
                if region:
                     rx, ry, rw, rh = region
                     if not (rx <= x < rx + rw and ry <= y < ry + rh):
                         continue

                # Trace
                final_color = self._trace_ray(scene, ray, MAX_DEPTH, seed)
                
                # FIX 3: Calculate proper Sample UVs if missing from the Ray
                # We reconstruct 'u' and 'v' from the ray orientation if the generator didn't save them.
                # Ideally, RayGenerator should save ray.u and ray.v. 
                # Here, we assume the ray might lack them and patch it:
                if hasattr(ray, 'sample_u'):
                    s_u, s_v = ray.sample_u, ray.sample_v
                else:
                    # FALLBACK: If generator didn't store normalized coords, 
                    # we can't filter accurately. We force center of pixel.
                    # This forces a BOX filter look.
                    s_u = (x + 0.5) / cam_w
                    s_v = (y + 0.5) / cam_h

                sample = Sample(s_u, s_v, 1.0)
                
                pixel_idx = int(y * cam_w + x)
                pixel_samples_and_colors[pixel_idx].append((sample, final_color))

        # --- EXECUTION ---
        rays = _gen_rays(region=region, seed=seed)
        process_rays(rays)

        # --- RECONSTRUCTION ---
        pixel_colors: List[Color] = []
        
        # We iterate purely by index to keep the flattened list structure correct
        for pixel_idx in range(len(pixel_samples_and_colors)):
            samples_and_colors = pixel_samples_and_colors[pixel_idx]
            
            # Calculate x, y from index for the filter context
            y = pixel_idx // cam_w
            x = pixel_idx % cam_w

            if not samples_and_colors:
                pixel_colors.append(Color(0, 0, 0))
                continue

            samples = [sc[0] for sc in samples_and_colors]
            # Convert Color objects to numpy arrays for the reconstructor math
            colors = [np.array([sc[1].r, sc[1].g, sc[1].b]) for sc in samples_and_colors]
            
            # Note: This reconstruction is still strictly "Bucketed" 
            # (only looks at samples inside this pixel). 
            reconstructed = reconstruct_pixel(x, y, samples, colors, self.sample_settings)
            
            pixel_colors.append(Color(reconstructed[0], reconstructed[1], reconstructed[2]))

        return pixel_colors

    def __repr__(self):
        return f"Raytracer(ray_generator={self.ray_generator}, intersector={self.intersector}, interactor={self.interactor}, shader={self.shader})"