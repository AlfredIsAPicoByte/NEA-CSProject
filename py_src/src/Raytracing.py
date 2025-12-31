import numpy as np
import math
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from CommonUtils import lerp, unit, attenuate_distance_exponential
from PrimaryStructures import HitInfo, TracingRay
from Scene import Scene
from Camera import VCamera, CameraType
from Geometry import Shape, VObject, get_transformed_exit_point
from Reflections import calculate_reflection_vector
from Refractions import calculate_refraction_vector, REFRACTIVE_INDICES
from Luminance import Color, PBRMaterial, MaterialType, LightSource, calculate_fresnel_ratio, schlick_fresnel
from RenderingAlgorithims import Algorithm, RenderStats, register_algorithm
from Sampling import SamplingManager, SampleSettings, Sampler, Sample, RandomSampler, reconstruct_pixel
from MemoryUtils import get_process_id, get_memory_mb

# Strategy interfaces for ray generation, intersection, and shading
class RayGenerationStrategy(ABC):
    def __init__(
            self,
            pixel_sampler: Optional[Sampler] = None
        ):
        self.pixel_sampler = pixel_sampler if pixel_sampler is not None else RandomSampler()

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
            epsilon: float = 1e-4,          # Hit Threshold
            max_steps: int = 128,           # Performance Cap
            max_distance: float = 1000.0,   # Far Clip Plane
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
            surface_sampler: Optional[Sampler] = None
        ):
        self.surface_sampler = surface_sampler if surface_sampler is not None else RandomSampler()

    @abstractmethod
    def interact(self, ray: TracingRay, hit_info: HitInfo, seed: Optional[int] = None, bias: float = 1e-4) -> Optional[TracingRay]:
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
        intersection_function: Callable[[Scene, TracingRay], HitInfo],
        interaction_function: Callable[[TracingRay, HitInfo, Optional[float]], Optional[TracingRay]],
        seed: Optional[int] = None,
        bias: float = 1e-4
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
class JitterRayGenerator(RayGenerationStrategy):
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

        self.pixel_sampler.seed = seed

        # 2. Iterate over pixels
        for y in range(y_start, y_start + region_h):
            for x in range(x_start, x_start + region_w):
                # 3. Get Samples
                # The SamplingManager returns a list of Sample objects (offsets 0.0-1.0)
                # matching the configured Samples Per Pixel (SPP).

                pixel_samples = self.pixel_sampler.get_samples_per_pixel(x, y)
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
                        # Origin = CameraPos + (Right * px) + (Up * py) + camera near plane
                        ray_orientation = camera.transform.forward
                        ray_origin = (
                            camera.transform.position + 
                            (camera.transform.right * px) + 
                            (camera.transform.up * py)
                        ) + ray_orientation * camera.near
                    else:
                        # --- PERSPECTIVE ---
                        # Origin: Always the camera position + the near curve for the camera
                        # Direction: Diverges from origin through the pixel
                        ray_orientation = self._camera_rotate_ray(camera, u, v)
                        ray_origin = camera.transform.position + ray_orientation * camera.near

                    # 5. Build Ray
                    ray = TracingRay(
                        origin=ray_origin,
                        orientation=ray_orientation,
                        pixel_x=x,
                        pixel_y=y,
                        name=f"ray#{i}_sceen-coords({x},{y})",
                        throughput=Color(1.0, 1.0, 1.0) # Used for path tracing accumulation
                    )
                    rays.append(ray)
        return rays

# Ray intersection implementations
class RayMarchingIntersection(IntersectionStrategy):
    """
    Find hits between rays and objects using the Ray Marching method.
    """
    def find_hit(self, scene: Scene, ray: TracingRay) -> HitInfo:
        # --- LOGIC BRANCH A: Ray is escaping an object ---
        if getattr(ray, "is_inside", False):
            # We assume we are inside the object that was last hit.
            # If your ray doesn't store the 'current_container_object', 
            # we must find which object has a negative SDF at the origin.
            
            # (Simplification: We query the scene to find what we are inside)
            closest_object, dist = scene.distance_estimator(ray.origin)
            
            # If dist is negative, we are inside 'closest_object'.
            # If your SDFs are unsigned, this check is harder and relies on ray metadata.
            if closest_object:
                 # Calculate the matrices
                obj_matrix = closest_object.transform.get_global_matrix()
                inv_obj_matrix = np.linalg.inv(obj_matrix) # Invert for local space
                
                exit_point = get_transformed_exit_point(
                    ray.origin, 
                    ray.orientation, 
                    obj_matrix, 
                    inv_obj_matrix,
                    64
                )
                
                if exit_point is not None:
                    # Calculate distance to that exit point
                    dist_to_exit = np.linalg.norm(exit_point - ray.origin)
                    
                    # Calculate normal (pointing INWARDS for back-faces usually, 
                    # or OUTWARDS depending on your lighting logic). 
                    # For refraction, we usually want the normal pointing OUT into the air.
                    normal = closest_object.shape.get_normal(exit_point)
                    
                    # IMPORTANT: If we are hitting the 'inside' face, the normal 
                    # from GetNormal() points OUT. We might need to flip it 
                    # depending on your refraction math. Usually: normal = -normal
                    
                    return HitInfo(
                        did_hit=True,
                        hit_point=exit_point,
                        incoming_direction=ray.orientation,
                        surface_normal=-normal, 
                        distance=dist_to_exit,
                        obj=closest_object,
                    )
            
            # If we failed to find an exit (infinite solid?), return Miss
            return HitInfo(False)

        # --- LOGIC BRANCH B: Standard Raymarching (Outside) ---
        distance_traveled = 0.0

        for _ in range(self.max_steps):
            point = ray.point_at(distance_traveled)

            closest_object, distance_to_closest = scene.distance_estimator(point)

            # Optimization: If we marched into the void
            if closest_object is None:
                break
            
            # Hit Check
            if distance_to_closest <= self.epsilon:
                # Calculate Normal
                surface_normal = np.array([0.0, 1.0, 0.0])
                if hasattr(closest_object, "shape") and hasattr(closest_object.shape, "GetNormal"):
                    surface_normal = closest_object.shape.get_normal(point)

                return HitInfo(
                    did_hit=True,
                    hit_point=point,
                    incoming_direction=ray.orientation,
                    surface_normal=surface_normal,
                    distance=distance_traveled,
                    obj=closest_object,
                )
            
            # Advance
            distance_traveled += distance_to_closest
            if distance_traveled >= self.max_distance:
                break

        return HitInfo(False)

class InverseSDFStrategy(IntersectionStrategy):
    """
    Find hits between rays and objects using the Inverse SDF method.
    """
    def __init__(
            self,
            epsilon: float = 1e-4,
            max_steps: int = 128,
            max_distance: float = 1000,
            step_relaxation: float = 0.9,   # Step Safety Factor
            use_bounding_box: bool = True   # Optimization Flag
        ):
        super().__init__(epsilon, max_steps, max_distance)
        self.step_relaxation = step_relaxation
        self.use_bounding_box = use_bounding_box

    def find_hit(self, scene: Scene, ray: TracingRay) -> "HitInfo":
        closest_hit = HitInfo(did_hit=False, distance=float('inf'))
        
        # We check every object in the scene independently
        for obj in scene.objects:
            
            # Skip objects that don't have an SDF shape defined
            if not hasattr(obj, 'shape') or not hasattr(obj.shape, 'sdf'):
                continue

            # Attempt to intersect this specific object
            hit = self._intersect_object(obj, ray)
            
            # Keep track of the closest hit only
            if hit.did_hit and hit.distance < closest_hit.distance:
                closest_hit = hit
                
        return closest_hit

    def _intersect_object(self, obj: VObject, ray: "TracingRay") -> "HitInfo":
        """
        Performs the 'Inverse SDF' logic:
        1. Transform Ray -> Local Space
        2. March in Local Space (Unscaled)
        3. Transform Hit -> World Space
        """
        
        # --- 1. Transform Ray to Local Space ---
        # We assume the object handles the matrix math helpers
        local_origin = obj.shape.inverse_transform_point(ray.origin)
        local_dir_raw = obj.shape.inverse_transform_vector(ray.orientation)
        
        # CRITICAL FIX: Normalize local direction.
        # Scaling operations in the matrix change the vector length.
        # Raymarching requires a normalized direction to step correctly.
        dir_length = np.linalg.norm(local_dir_raw)
        if dir_length == 0:
            return HitInfo(did_hit=False)
            
        local_dir = local_dir_raw / dir_length

        # --- 2. Raymarch Loop ---
        t = 0.0
        
        # Check for "Inside-Out" logic (for X-ray/Dielectrics)
        # If we are inside, we treat negative distance as empty space (flip sign)
        sign_modifier = -1.0 if ray.is_inside else 1.0
        
        for _ in range(self.max_steps):
            p = local_origin + (local_dir * t)
            
            # Sample the Object's SDF (in local space)
            raw_dist = obj.shape.signed_distance(p)
            
            # Apply Modifier (flips distance if inside)
            dist = raw_dist * sign_modifier
            
            # HIT CONDITION
            if dist < self.epsilon:
                # We hit the surface in Local Space!
                
                # --- 3. Transform Back to World Space ---
                # We calculate the world hit point specifically based on the local hit
                p_local_hit = p
                
                # A. Transform Point
                p_world_hit = obj.transform.transform_point(p_local_hit)
                
                # B. Calculate Distance (Depth)
                # We calculate world distance explicitly to avoid scaling errors
                world_distance = np.linalg.norm(p_world_hit - ray.origin)
                
                # C. Calculate Normal
                # We need the gradient at the local point, then transformed
                local_normal = self._calc_local_gradient(obj.shape, p_local_hit)
                
                # If we are hitting the "inside" face (exiting), the normal should 
                # point towards the empty space (which is effectively 'out' for us)
                if ray.is_inside:
                    local_normal = -local_normal
                    
                world_normal = obj.transform_normal(local_normal)
                
                return HitInfo(
                    did_hit=True,
                    distance=world_distance,
                    point=p_world_hit,
                    normal=world_normal,
                    object=obj
                )
            
            # STEP
            # Note: We do NOT scale 'dist' here. 
            # We are marching in Local Space. 1 unit is 1 unit.
            t += dist
            
            # Far Plane Check
            if t > self.max_distance:
                break
                
        return HitInfo(did_hit=False)

    def _calc_local_gradient(self, shape, p: np.ndarray) -> np.ndarray:
        """
        Calculates the normal in Local Space using central differences.
        """
        h = 1e-4 # Small step for gradient
        dx = np.array([h, 0, 0])
        dy = np.array([0, h, 0])
        dz = np.array([0, 0, h])
        
        val = shape.sdf(p)
        
        grad = np.array([
            shape.sdf(p + dx) - shape.sdf(p - dx),
            shape.sdf(p + dy) - shape.sdf(p - dy),
            shape.sdf(p + dz) - shape.sdf(p - dz)
        ])
        
        # Normalize the gradient to get the normal
        norm = np.linalg.norm(grad)
        if norm > 0:
            return grad / norm
        return np.array([0.0, 1.0, 0.0]) # Fallback

# Interaction implementations
class TerminalInteraction(InteractionStrategy):
    """
    A 'Null' interaction.
    The ray is absorbed or the calculation is finished.
    - Debug Views (X-Ray, Normals, Depth) where shading is self-contained.
    - Matte/Black hole materials.
    - Light sources (if they don't reflect).
    """
    def __init__(
            self,
            surface_sampler: Sampler = None, # Default handled in base
            allow_passthrough: bool = False,   # If True, ray continues straight through
            opacity: float = 1.0,              # 1.0 = Solid, 0.0 = Invisible
        ):
        super().__init__(surface_sampler=surface_sampler)
        self.allow_passthrough = allow_passthrough
        self.opacity = max(0.0, min(opacity, 1.0)) # Clamp 0-1

    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        seed: Optional[int] = None, 
        bias: float = 1e-4
    ) -> Optional["TracingRay"]:
        
        # Case 1: Standard Terminal Interaction (Solid Wall)
        if not self.allow_passthrough:
            return None

        # Case 2: Passthrough / Ghost Mode
        # We handle opacity stochastically (Russian Roulette).
        # This is standard for path tracers to avoid branching rays.
        
        rng = np.random.default_rng(seed)
        
        # If opacity is 0.5, there is a 50% chance the ray is absorbed (Hit),
        # and a 50% chance it passes through (Miss).
        if self.opacity > 0.0:
            # Generate a random number [0.0, 1.0)
            if rng.random() < self.opacity:
                # Ray is absorbed by the "smoke/ghost"
                return None

        # If we survive the opacity check (or opacity is 0), the ray passes through.
        # It continues in the EXACT same direction.
        
        # Push ray slightly forward to avoid self-intersection with the exit point
        next_origin = hit_info.point + (ray.orientation * bias)
        
        return TracingRay(
            origin=next_origin,
            orientation=ray.orientation, # Maintain original direction
            is_inside=ray.is_inside      # Maintain current medium state
        )

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
        surface_sampler: Optional[Sampler] = None,
        opacity_cutoff: float = 0.0, # 0.0 = Invisible/Clear, 1.0 = Opaque (Absorbs ray)
    ):
        # Initialize base to handle samplers/IOR if needed later
        super().__init__(surface_sampler)
        self.opacity_cutoff = np.clip(opacity_cutoff, 0.0, 1.0)
    
    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        seed: Optional[int] = None, 
        bias: float = 1e-4
    ) -> Optional["TracingRay"]:
        
        # 1. Stochastic Opacity Check (Russian Roulette)
        # If opacity is 0.5, 50% of rays are blocked, 50% pass through.
        # This simulates semi-transparency without splitting rays.
        if self.opacity_cutoff > 0.0:
            rng = np.random.default_rng(seed)
            if rng.random() < self.opacity_cutoff:
                return None # Ray is absorbed/blocked by the "smoke"

        # 2. Passthrough Logic
        # We spawn a new ray continuing in the exact same direction.
        
        # Push ray slightly forward to avoid self-intersection
        next_origin = hit_info.point + (ray.orientation * bias)
        
        return TracingRay(
            origin=next_origin,
            orientation=ray.orientation, # Keep exact direction
            is_inside=ray.is_inside      # Maintain medium state (don't toggle)
        )

class StandardInteraction(InteractionStrategy):
    """
    A unified PBR-style interaction.
    - Can simulate: Mirrors, Glass, Matte, Metal, and Glossy surfaces.
    """
    def __init__(
        self,
        surface_sampler: Optional[Sampler] = None,
        scene_ior: float = REFRACTIVE_INDICES["air"]
    ):
        super().__init__(surface_sampler)
        self.scene_ior = scene_ior

    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        seed: Optional[int] = None, 
        bias: float = 1e-4
    ) -> Optional[TracingRay]:
        rng = np.random.default_rng(seed)
        material: PBRMaterial = hit_info.object.material

       # 1. Determine Optical Density (IOR)
        # Are we entering the glass or leaving it?
        if ray.is_inside:
            n1 = material.data.ior         # Inside glass
            n2 = 1.0                  # Outside (Air/Vacuum)
            normal = -hit_info.normal # Flip normal to point inward
        else:
            n1 = 1.0                  # Air
            n2 = material.data.ior         # Glass
            normal = hit_info.normal

        # 2. Calculate Fresnel Chance (Schlick)
        # "reflection_chance" acts as the probability for Russian Roulette
        reflection_chance = calculate_fresnel_ratio(
            ray.orientation, normal, n1, n2
        )

        # 3. Decision: Reflect or Refract? (Russian Roulette)
        # We start with the assumption based on the dice roll.
        do_refract = rng.random() > reflection_chance
        
        new_ray = None
        pdf = 0.0
        
        # --- PATH A: Attempt Refraction ---
        if do_refract:
            new_ray, pdf = material.calculate_refraction_ray(
                incoming_ray=ray,
                surface_normal=normal,
                new_origin=hit_info.point,
                sampler=Sampler(rng), # Wrap rng if needed
                ior_incident=n1,
                ior_transmitted=n2
            )
            
            # CHECK: Did Total Internal Reflection (TIR) occur?
            # If calculate_refraction_ray returns None, physics says we MUST reflect.
            if new_ray is None:
                do_refract = False # Fallback to reflection

        # --- PATH B: Reflection ---
        # Note: This is NOT an 'else'. If refraction failed above, we enter this block.
        if not do_refract:
            new_ray, pdf = material.calculate_microfacet_reflection(
                incoming_ray=ray,
                surface_normal=normal,
                new_origin=hit_info.point,
                sampler=Sampler(rng)
            )

        # 4. Finalize the Ray
        if new_ray:
            # Transfer the recursion depth and 'inside' status
            new_ray.depth = ray.depth + 1
            
            # If we successfully refracted, toggle the 'is_inside' flag
            if do_refract:
                new_ray.is_inside = not ray.is_inside
            else:
                new_ray.is_inside = ray.is_inside

            # 5. Calculate Throughput (Color Tint)
            # This is crucial for the Shader to know how much light to keep.
            # We normalize by the PDF to balance the Monte Carlo equation.
            
            # If we Reflected: We keep the Fresnel amount (F)
            # If we Refracted: We keep the Transmission amount (1 - F)
            
            # Simplify: In pure Russian Roulette, we divide by the probability 
            # of picking that path to keep the energy unbiased.
            if do_refract:
                 # Energy = (1-F) / Probability(Refract)
                 weight = (1.0 - reflection_chance) / (1.0 - reflection_chance)
                 new_ray.throughput = Color(1,1,1) * weight # equals 1.0 (white)
            else:
                 # Energy = F / Probability(Reflect)
                 # Note: If this was a forced TIR fallback, probability is effectively 1.0
                 weight = reflection_chance / reflection_chance if reflection_chance > 0 else 0
                 new_ray.throughput = Color(1,1,1) * 1.0

            # (Optional) Tinting: 
            # If you want colored glass reflection, multiply throughput by base color here.
            # new_ray.throughput *= material.albedo 

        return new_ray

# Shading implementations
class RecursiveLambertShading(ShadingStrategy):
    """
    Standard recursive shading. 
    Handles Direct Light (Shadow Rays) and Indirect Light (Reflection/Refraction Rays).
    """
    def shade(
            self,
            scene: Scene,
            ray: TracingRay,
            hit_info: HitInfo,
            current_depth: int,
            trace_function: Callable[[Scene, TracingRay, int, Optional[int]], Color],
            intersection_function: Callable[[Scene, TracingRay], HitInfo],
            interaction_function: Callable[[TracingRay, HitInfo, Optional[int]], Optional[TracingRay]],
            seed: Optional[int]
        ) -> Color:
        
        v_object = hit_info.object
        material = v_object.material
        
        # --- 1. Emission (The Glow) ---
        # Added first so even if we stop recursing, we see the light.
        final_color = Color(0, 0, 0)
        if material.type == MaterialType.EMISSIVE:
             final_color += material.get_emissive_component()
             # Emissive materials usually don't reflect/receive shadow, so we can return early
             if self.skip_lighting_on_emissive: return final_color

        # --- 2. Direct Lighting (Next Event Estimation) ---
        view_dir = -unit(ray.orientation)
        
        # We assume 'evaluate_direct_light' handles the loop over lights & shadow checks
        def visibility_checker(p1, p2):
             if not self.enable_shadows: return 1.0
             return 0.0 if scene.is_occluded(p1, p2, self.shadow_bias) else 1.0

        direct_light = material.evaluate_direct_light(
            scene.lights, hit_info, view_dir, visibility_checker
        )
        final_color += direct_light

        # --- 3. Ambient (Optional) ---
        if self.ambient_enabled:
            final_color += material.get_ambient_color(scene.ambient_color, scene.ambient_intensity)

        # --- 4. Indirect Lighting (Recursion) ---
        # The 'interaction_function' calculates the physics of the bounce.
        # It performs Russian Roulette (Reflect OR Refract) and sets 'new_ray.throughput'.
        if current_depth > 0:
            new_ray = interaction_function(ray, hit_info, seed)

            if new_ray is not None:
                # Recursively trace the new path
                incoming_light = trace_function(scene, new_ray, current_depth - 1, seed)
                
                # Apply the throughput (Color tint * Cosine Term / PDF)
                # This handles the Glass "Darkening" or Metal "Tinting" automatically.
                throughput = getattr(new_ray, "throughput", Color(1,1,1))
                final_color += incoming_light * throughput

        # --- 5. Volumetrics (Beer's Law) ---
        # If the ray passed THROUGH an object to get here, absorb some color.
        if getattr(ray, "is_inside", False):
            final_color = material.get_volumetric_component(final_color, hit_info.distance)

        return final_color

class XRayThicknessShading(ShadingStrategy):
    """
    Renders objects based on their thickness. 
    Useful for medical visualization, sub-surface scattering approximation, or sci-fi energy shields.
    """
    def __init__(
        self,
        density: float = 1.0,                               # Beer's Law coefficient (Higher = rapid absorption)
        absorption_color: Color = Color(0.2, 0.8, 1.0),     # The color of the object material
        invert_style: bool = True,                          # True = Sci-Fi (Thick is Bright), False = Glass (Thick is Dark)
        rim_power: float = 3.0,                             # Edge highlighting strength (0.0 to disable)
        rim_color: Color = Color(1.0, 1.0, 1.0),            # Color of the edge highlight
        max_thickness: float = 10.0                         # Clamping value to prevent infinite vals on open meshes
    ):
        self.density = density
        self.absorption_color = absorption_color
        self.invert_style = invert_style
        self.rim_power = rim_power
        self.rim_color = rim_color
        self.max_thickness = max_thickness

    def shade(
        self,
        scene: "Scene",
        ray: "TracingRay",
        hit_info: "HitInfo",
        current_depth: int,
        trace_function: Callable[[Scene, TracingRay, int], Color],
        intersection_function: Callable[[Scene, TracingRay], HitInfo],
        interaction_function: Callable[[TracingRay, HitInfo, Optional[float]], Optional[TracingRay]],
        seed: Optional[int] = None,
        bias: float = 1e-4
    ) -> Color:
        # 1. Setup Vectors
        # Normal pointing OUT of the surface
        normal = hit_info.normal
        view_dir = -ray.orientation
        view_dir = view_dir / np.linalg.norm(view_dir)

        # 2. Calculate Thickness (The "Through" Ray)
        # We push the origin slightly INSIDE the object (opposite to normal) to avoid self-intersection at the entry.
        # Note: If geometry is single-sided planes, this might fail. Assumes closed volume.
        inside_origin = hit_info.point - (normal * bias) 
        
        # Ray travels in the EXACT same direction as the camera ray
        inside_ray = TracingRay(
            origin=inside_origin, 
            orientation=ray.orientation,
            depth=ray.depth # Keep depth to prevent infinite recursion bugs
        )

        # Find where the ray leaves the object
        exit_hit = intersection_function(scene, inside_ray)

        thickness = 0.0
        if exit_hit.hit:
            thickness = exit_hit.distance
        else:
            # Ray didn't hit anything inside (Open mesh? Plane?). 
            # Fallback: Assume some arbitrary thickness or 0.
            thickness = 0.0

        # Clamp thickness for safety
        thickness = min(thickness, self.max_thickness)

        # 3. Apply Beer's Law (Attenuation)
        # Transmission = exp(-density * distance)
        # Result is 1.0 (Thin) to 0.0 (Thick)
        transmission = attenuate_distance_exponential(thickness, self.density)

        # 4. Determine Core Color
        final_color = Color(0,0,0)

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

# Stats for ray tracing
@dataclass
class TracingStats(RenderStats):
    # --- Basic Counters ---
    rays_primary: int = 0
    rays_shadow: int = 0
    rays_reflection: int = 0
    rays_refraction: int = 0
    
    # --- Intersection Performance (Crucial for BVH optimization) ---
    aabb_tests: int = 0      # How many boxes did we hit-test?
    triangle_tests: int = 0  # How many actual triangles did we hit-test?
    
    # --- Logic & Debugging ---
    pixels_processed: int = 0
    zero_contribution_paths: int = 0 # Rays that hit nothing or black materials (wasted work)
    nan_errors: int = 0      # Rays that resulted in Math Errors

    @property
    def total_rays(self) -> int:
        return self.rays_primary + self.rays_shadow + self.rays_reflection + self.rays_refraction

    @property
    def intersections_per_ray(self) -> float:
        """Lower is better. High numbers mean poor spatial partitioning."""
        if self.total_rays == 0: return 0.0
        return (self.aabb_tests + self.triangle_tests) / self.total_rays

    def __add__(self, other: 'TracingStats') -> 'TracingStats':
        new_stats: TracingStats = super.__add__(other)
        
        # Sum counters
        new_stats.rays_primary = self.rays_primary + other.rays_primary
        new_stats.rays_shadow = self.rays_shadow + other.rays_shadow
        new_stats.rays_reflection = self.rays_reflection + other.rays_reflection
        new_stats.rays_refraction = self.rays_refraction + other.rays_refraction
        new_stats.aabb_tests = self.aabb_tests + other.aabb_tests
        new_stats.triangle_tests = self.triangle_tests + other.triangle_tests
        new_stats.nan_errors = self.nan_errors + other.nan_errors
        
        return new_stats

    def print_verbose_report(self):
        print(f"\n=== Tracing Stats ===")
        print(f"Time: {self.time_taken_seconds:.3f}s")
        print(f"Memory: {self.memory_usage:.3f}MB")
        print(f"-------------------------")
        print(f"Total Rays:      {self.total_rays:,}")
        print(f"  - Primary:     {self.rays_primary:<10,} ({self.rays_primary/max(1,self.total_rays)*100:.1f}%)")
        print(f"  - Shadow:      {self.rays_shadow:<10,} ({self.rays_shadow/max(1,self.total_rays)*100:.1f}%)")
        print(f"  - Bounce:      {(self.rays_reflection+self.rays_refraction):<10,}")
        print(f"-------------------------")
        print(f"Optimization Metrics:")
        print(f"  - AABB Tests:  {self.aabb_tests:,}")
        print(f"  - Tri Tests:   {self.triangle_tests:,}")
        print(f"  - Cost/Ray:    {self.intersections_per_ray:.2f} tests per ray")
        
        if self.nan_errors > 0:
            print(f"!!! WARNING: {self.nan_errors} Math Errors (NaN/Inf) Detected !!!")

def update_memory_stats(stats: TracingStats) -> TracingStats:
    """
    Returns a NEW TracingStats object with updated memory usage,
    leaving the original object untouched (Immutability).
    """
    from dataclasses import replace
    
    current_mem = get_memory_mb(get_process_id())
    
    return replace(stats, memory_usage=current_mem)

# Raytracer using strategies
@register_algorithm("raytracer")
class Raytracer(Algorithm):
    def __init__(
        self,
        max_depth: int = 4,
        sampling_manager: Optional[SamplingManager] = None,
        ray_generator: Optional[RayGenerationStrategy] = None,
        intersection_strategy: Optional[IntersectionStrategy] = None,
        interaction_strategy: Optional[InteractionStrategy] = None,
        shading_strategy: Optional[ShadingStrategy] = None,
        sample_settings: Optional[SampleSettings] = None,
        custom_background: Optional[Color] = None,
        enable_scene_background: bool = False
    ):
        super().__init__()
        self.sampling_manager = sampling_manager
        self.sample_settings = sample_settings or SampleSettings()

        self.ray_generator: RayGenerationStrategy = ray_generator if ray_generator is not None else JitterRayGenerator()
        self.intersector: IntersectionStrategy = intersection_strategy if intersection_strategy is not None else RayMarchingIntersection()
        self.interactor: InteractionStrategy = interaction_strategy if interaction_strategy is not None else StandardInteraction()
        self.shader: ShadingStrategy = shading_strategy if shading_strategy is not None else RecursiveLambertShading()
        
        self.max_depth = max_depth

        self.custom_background = custom_background
        self.enable_scene_background = enable_scene_background

        self.stats = TracingStats()

    def _trace_ray(self, scene: Scene, ray: TracingRay, depth: int, seed: Optional[int]) -> Color:
        """
        The recursive engine. It takes a ray, finds what it hits, and calculates the color.
        If the surface is reflective, this function will be called again by the shader.
        """
        if depth < 0:
            return Color(0.0, 0.0, 0.0)

        hit_info = self.intersector.find_hit(scene, ray)

        self.stats = update_memory_stats(self.stats)
        if hit_info.object is not None and hit_info.hit:
            return self.shader.shade(
                scene=scene, 
                ray=ray, 
                hit_info=hit_info,
                current_depth=depth,
                trace_function=self._trace_ray,
                intersection_function=self.intersector.find_hit,
                interaction_function=self.interactor.interact,
                seed=seed
            )

        # The ray missed all scene objects
        if self.enable_scene_background:
            return self.custom_background
        return scene.get_background_color(np.asarray(ray.orientation))

    def render(
        self,
        scene: Scene,
        seed: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        tile_size: int = 64
    ) -> List[Color]:
        self.stats.start_timer()
        self.stats.rays_primary = 0
        self.stats.rays_shadow = 0
        self.stats.rays_reflection = 0
        self.stats.rays_refraction = 0
        
        cam = scene.camera
        if cam is None: raise ValueError("No camera provided in Scene")

        # Determine Render Bounds
        if region:
            rx, ry, rw, rh = region
        else:
            rx, ry, rw, rh = 0, 0, cam.width, cam.height

        # Initialize full image buffer (Black)
        # We use a 1D array to store the final result
        full_image_pixels = [Color(0,0,0) for _ in range(rw * rh)]

        self.sample_settings.width = cam.width
        self.sample_settings.height = cam.height

        # --- TILE LOOP ---
        # Iterate over the image in blocks (tiles)
        for tile_y in range(ry, ry + rh, tile_size):
            for tile_x in range(rx, rx + rw, tile_size):
                
                # Calculate current tile dimensions (handle edges)
                current_w = min(tile_size, (rx + rw) - tile_x)
                current_h = min(tile_size, (ry + rh) - tile_y)
                
                # Define tile region: (x, y, w, h)
                tile_region = (tile_x, tile_y, current_w, current_h)
                
                # 1. Generate Rays for this Tile ONLY
                rays = self.ray_generator.generate(cam, region=tile_region, seed=seed)
                self.stats.rays_primary += len(rays)

                # 2. Local Storage for this Tile
                # Map: (local_tile_index) -> List[(Sample, Color)]
                tile_samples = [[] for _ in range(current_w * current_h)]

                # 3. Trace Rays
                for ray in rays:
                    if ray is None: continue
                    
                    # Global Coordinates
                    px, py = getattr(ray, "pixel_x", -1), getattr(ray, "pixel_y", -1)
                    
                    # Convert to Local Tile Coordinates
                    local_x = px - tile_x
                    local_y = py - tile_y

                    # Safety Check
                    if not (0 <= local_x < current_w and 0 <= local_y < current_h):
                        continue

                    # Trace
                    pixel_color = self._trace_ray(scene, ray, self.max_depth, seed)

                    # Create Sample Object
                    s_u = getattr(ray, "sample_u", (px + 0.5) / cam.width)
                    s_v = getattr(ray, "sample_v", (py + 0.5) / cam.height)
                    sample = Sample(s_u, s_v, 1.0) # weight 1.0

                    # Store in Local Tile Buffer
                    local_idx = local_y * current_w + local_x
                    tile_samples[local_idx].append((sample, pixel_color))

                # 4. Reconstruct Tile (Resolve samples to final color)
                for i in range(len(tile_samples)):
                    samples_and_colors = tile_samples[i]
                    
                    if not samples_and_colors: continue

                    # Calculate Global Pixel Index
                    local_y_in_tile = i // current_w
                    local_x_in_tile = i % current_w
                    
                    global_x = tile_x + local_x_in_tile
                    global_y = tile_y + local_y_in_tile
                    
                    # Reconstruct
                    samples = [sc[0] for sc in samples_and_colors]
                    colors = [np.array([sc[1].r, sc[1].g, sc[1].b]) for sc in samples_and_colors]
                    
                    # Note: You need to import/define 'reconstruct_pixel'
                    rec_rgb = reconstruct_pixel(global_x, global_y, samples, colors, self.sample_settings)
                    final_color = Color(rec_rgb[0], rec_rgb[1], rec_rgb[2])

                    # Write to Final Image Buffer
                    # Calculate index in the *region* buffer
                    final_idx = (global_y - ry) * rw + (global_x - rx)
                    full_image_pixels[final_idx] = final_color

                # 5. End of Tile - Memory for 'tile_samples' and 'rays' is freed here automatically

        self.stats.stop_timer()
        self.stats.print_verbose_report()

        return full_image_pixels

    def __repr__(self):
        return f"Raytracer(ray_generator={self.ray_generator}, intersector={self.intersector}, interactor={self.interactor}, shader={self.shader})"