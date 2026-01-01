import numpy as np
import math
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PrimaryStructures import HitInfo, TracingRay
from Scene import Scene
from Camera import VCamera, CameraType
from Geometry import Shape, VObject, get_transformed_exit_point
from Reflections import calculate_reflection_vector
from Refractions import REFRACTIVE_INDICES
from Luminance import Color, Material, MaterialType, LightSource, calculate_redirection_ray, schlick_fresnel, lerp
from RenderingAlgorithims import Algorithm, RenderStats, register_algorithm
from Sampling import SamplingManager, SampleSettings, Sampler, Sample, RandomSampler, reconstruct_pixel
from MemoryUtils import get_process_id, get_memory_mb

# Strategy interfaces for ray generation, intersection, and shading
class RayGenerator(ABC):
    def __init__(
            self,
            gen_sampler: Sampler = RandomSampler()  
        ):
        self.gen_sampler = gen_sampler

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
            scene_refractive_index: float = REFRACTIVE_INDICES["air"]
        ):
        self.surface_sampler = surface_sampler
        self.scene_ior = scene_refractive_index

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

        self.gen_sampler.seed = seed

        # 2. Iterate over pixels
        for y in range(y_start, y_start + region_h):
            for x in range(x_start, x_start + region_w):
                # 3. Get Samples
                # The SamplingManager returns a list of Sample objects (offsets 0.0-1.0)
                # matching the configured Samples Per Pixel (SPP).

                pixel_samples = self.gen_sampler.get_samples_per_pixel(x, y)
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
    A 'Null' interaction. The ray is absorbed or the calculation is finished.
    
    Use this for:
    1. Debug Views (X-Ray, Normals, Depth) where shading is self-contained.
    2. Matte/Black hole materials.
    3. Light sources (if they don't reflect).
    """
    def interact(self, ray: "TracingRay", hit_info: "HitInfo", seed: Optional[int] = None, bias: float = 1e-4) -> Optional["TracingRay"]:
        # We return None to signal the end of the light path.
        # The ShadingStrategy has already calculated the final color.
        return None

class PassthroughInteraction(InteractionStrategy):
    """
    The ray passes perfectly straight through the object, ignoring refraction.
    Useful for 'Ghost' objects or volumetric overlays.
    """
    def interact(self, ray: TracingRay, hit_info: HitInfo, seed: Optional[int] = None, bias: float = 1e-4) -> Optional["TracingRay"]:
        
        # Spawn a new ray continuing in the exact same direction
        # We push it slightly forward to avoid self-intersection
        next_origin = hit_info.point + (ray.orientation * bias)
        
        return TracingRay(
            origin=next_origin,
            orientation=ray.orientation,
            is_inside=ray.is_inside # Maintain current state
        )

class SimpleMaterialInteraction(InteractionStrategy):
    def interact(self, ray: TracingRay, hit_info: HitInfo, seed: Optional[int], bias: float = 1e-4) -> Optional[TracingRay]:
        # 1. Resolve Material
        v_object: VObject = hit_info.object
        material: Material = getattr(v_object.shape, "material", None)
        if not material:
            return None

        # 2. Initialize State
        current_throughput = getattr(ray, "throughput", Color(1.0, 1.0, 1.0))
        
        # --- PHASE A: Volumetric Absorption (The journey SO FAR) ---
        if getattr(ray, "is_inside", False):
            if hasattr(material, "get_volumetric_component"):
                current_throughput = material.get_volumetric_component(current_throughput, hit_info.distance)

        # --- PHASE B: Calculate Redirection ---
        # Initialize variables to ensure they exist if 'try' fails
        new_ray_dir = None
        pdf = 0.0
        did_reflect = True
        is_next_ray_inside = False

        try:
            if getattr(ray, "is_inside", False):
                new_ray, pdf, did_reflect, is_next_ray_inside = calculate_redirection_ray(
                    incoming_ray=ray,
                    surface_normal=hit_info.normal,
                    hit_point=hit_info.point,
                    roughness=material.roughness,
                    sampler=self.surface_sampler,
                    refactive_index_incident=material.ior,
                    refactive_index=self.scene_ior,
                    seed=seed,
                    fast=False
                )
            else:
                new_ray, pdf, did_reflect, is_next_ray_inside = calculate_redirection_ray(
                    incoming_ray=ray,
                    surface_normal=hit_info.normal,
                    hit_point=hit_info.point,
                    roughness=material.roughness,
                    sampler=self.surface_sampler,
                    refactive_index_incident=self.scene_ior,
                    refactive_index=material.ior,
                    seed=seed,
                    fast=False
                )
            new_ray_dir = new_ray.orientation
        except Exception:
            # Fallback: Perfect Reflection
            # We explicitly set the variables needed below, rather than creating a Ray object
            new_ray_dir = calculate_reflection_vector(hit_info.normal, ray.orientation)
            pdf = 1.0
            is_next_ray_inside = getattr(ray, "is_inside", False)
            did_reflect = True

        # --- PHASE C: Apply Bias (Prevent Acne) ---
        dot_prod = np.dot(new_ray_dir, hit_info.normal)
        
        if dot_prod > 0:
            new_origin = hit_info.point + (hit_info.normal * bias)
        else:
            new_origin = hit_info.point - (hit_info.normal * bias)

        # --- PHASE D: Material Tinting ---
        if material.type != MaterialType.EMISSIVE:
            F0 = material.get_metallic_component()
            base_tint = Color(1.0, 1.0, 1.0)
            
            view_dir = -ray.orientation
            cos_theta = max(np.dot(view_dir, hit_info.normal), 0.0)
            fresnel_color = Color.from_array(schlick_fresnel(cos_theta, F0.to_np_ndarray()))
            
            if did_reflect:
                # --- REFLECTION CASE ---
                # Metals tint reflections with their Albedo.
                # Dielectrics (Glass/Plastic) reflect White (handled by F0).
                
                base_tint = Color(1.0, 1.0, 1.0)
                if material.type == MaterialType.SPECULAR:
                    base_tint = lerp(Color(1.0, 1.0, 1.0), material.albedo, material.metallic)
                
                # Energy = Fresnel * Intensity
                tint = base_tint * fresnel_color * material.specular_intensity
                
            else:
                # --- REFRACTION CASE (Transmission) ---
                # Conservation of Energy: Transmitted = 1.0 - Reflected
                transmission_factor = Color(1.0, 1.0, 1.0) - fresnel_color
                
                # Glass Albedo acts as a transmission filter (e.g. green glass)
                tint = material.albedo * transmission_factor

            current_throughput = current_throughput * tint

        # 3. Create Next Ray
        next_ray = TracingRay(
            origin=new_origin,
            orientation=new_ray_dir,
            throughput=current_throughput,
            pdf=pdf,
            depth=getattr(ray, "depth", 0) + 1,
            is_inside=is_next_ray_inside,
            name=f"{ray.name}_{'refl' if did_reflect else 'refr'}"
        )

        return next_ray

# Shading implementations
class RecursiveLambertShading(ShadingStrategy):
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
        
        # --- 1. Setup Geometry Context ---
        point = ray.point_at(hit_info.distance)
        v_object: VObject = hit_info.object
        
        if not hasattr(v_object, "shape") or not hasattr(v_object.shape, "GetNormal"):
            return Color(1, 0, 1)

        normal = v_object.shape.get_normal(point)
        view_dir = -ray.orientation
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        material: Optional[Material] = getattr(v_object.shape, "material", None)
        if not material:
            return Color(1, 0, 1)

        # --- 2. Calculate Self-Emission (The Glow) ---
        # FIX A: We must explicitly add the object's own light.
        # This ensures the yellow orb glows when seen by the camera AND by reflections.
        emissive_light = Color(0, 0, 0)
        if material.type == MaterialType.EMISSIVE:
            if hasattr(material, "get_emissive_component"):
                emissive_light = material.get_emissive_component()

        # --- 3. Direct Lighting (Shadows from Light Sources) ---
        direct_light = Color(0, 0, 0)
        
        # We classify materials into "Simple Reflective" vs "Complex Matte"
        # FIX B: Treat ALL Specular/Glass as "Simple" to avoid the broken BRDF math.
        # This forces the mirror (roughness 0.1) to use the simple recursion logic.
        is_simple_reflective = (material.type == MaterialType.GLASS) or (material.type == MaterialType.SPECULAR)

        if not is_simple_reflective:
            def check_visibility(hit_point, light_pos):
                if not self.enable_shadows: return 1.0
                return 0.0 if scene.is_occluded(hit_point, light_pos, bias=self.shadow_bias) else 1.0

            direct_light = material.apply_material(scene.get_lights(), hit_info, view_dir, check_visibility)
            
            if self.ambient_enabled:
                ambient_col = getattr(scene, "ambient_color", Color(0.03, 0.03, 0.03))
                ambient_intensity = getattr(scene, "ambient_intensity", 0.1)
                direct_light += material.apply_ambient_color(ambient_col, ambient_intensity)

        # --- 4. Indirect Lighting (Reflections/Refractions) ---
        indirect_light = Color(0.0, 0.0, 0.0)

        if current_depth > 0:
            new_ray = interaction_function(ray, hit_info, seed)

            if new_ray is not None:
                # Trace the bounce
                incoming_light = trace_function(scene, new_ray, current_depth - 1, seed)

                # Combine based on strategy
                if is_simple_reflective:
                    # FIX B (Part 2): For Metals/Glass, we rely on the 'throughput' calculated
                    # in interact() to handle the color/tint. We don't need complex PDF math here.
                    throughput = getattr(new_ray, "throughput", Color(1,1,1))
                    indirect_light = incoming_light * throughput
                    
                    if self.type == MaterialType.GLASS:
                        # 1. Calculate View-Angle Fresnel (How mirror-like is the edge?)
                        F0 = self.get_metallic_component()
                        NdotV = max(0.0, np.dot(hit_info.normal, view_dir))
                        
                        # Calculate F_view using Schlick approximation
                        F_view = F0 + (Color(1.0, 1.0, 1.0) - F0) * (1.0 - NdotV)**5
                        
                        # 2. Get the Background (Refraction)
                        background_color = self.get_transparency_component(indirect_light)
                        
                        # 3. Mix: The background is blocked by the reflection
                        # Conservation of Energy: Refraction = (1 - Fresnel)
                        final_color += background_color * (Color(1.0, 1.0, 1.0) - F_view)

                    elif self.type == MaterialType.TRANSPARENT:
                        # Simple Alpha Blending (Decals, etc.)
                        background_color = self.get_transparency_component(indirect_light)
                        final_color += background_color
                
                else:
                    # For Diffuse/Matte materials, we use the Rendering Equation Estimator
                    pdf = getattr(new_ray, "pdf", 0.0)
                    throughput = getattr(new_ray, "throughput", Color(1,1,1))
                    
                    if pdf > 1e-5:
                        brdf_val = material.evaluate_brdf(normal, view_dir, new_ray.orientation)
                        cos_theta = max(np.dot(normal, new_ray.orientation), 0.0)
                        
                        weight = (brdf_val * cos_theta) / pdf
                        indirect_light = incoming_light * weight * throughput

        # --- 5. Composition & Volumetrics ---
        # Add Emission + Direct + Indirect
        final_color = emissive_light + direct_light + indirect_light
        
        # Apply Volumetric Absorption (if the ray traveled through glass to get here)
        if getattr(ray, "is_inside", False):
            final_color = material.get_volumetric_component(final_color, hit_info.distance)

        return final_color

class XRayThicknessShading(ShadingStrategy):
    def __init__(
        self,
        thickness_scale: float = 0.5, # Controls brightness (lower = darker/thicker looking)
        invert_gamma: bool = False,   # If True, thicker parts look darker (absorption)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.thickness_scale = thickness_scale
        self.invert_gamma = invert_gamma

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
        bias: float = 1e-3
    ) -> Color:
        # 1. Background Check (Use Black for contrast)
        if not hit_info.did_hit:
            return Color(0.0, 0.0, 0.0)

        # 2. Calculate Exit Point
        ray_origin_inside = hit_info.point + (ray.orientation * bias)
        
        exit_ray = TracingRay(
            origin=ray_origin_inside,
            orientation=ray.orientation,
            is_inside=True
        )

        shape: Optional[Shape] = getattr(hit_info.object, "shape", None)
        if shape is None:
            # Error: Object has no shape?
            return Color(1.0, 0.0, 1.0) # Magenta Error
        
        iso_scene = Scene()
        iso_scene.add_object(shape)
        exit_hit = intersection_function(iso_scene, exit_ray)

        # 3. Calculate Thickness
        thickness = 0.0
        if exit_hit.did_hit:
            thickness = exit_hit.distance
        else:
            # Infinite/Leaky geometry detected
            return Color(1.0, 0.0, 0.0) # Red Alert

        # 4. Visualize
        # Use exponential falloff (Beer's Law) for better volumetric perception
        # High density = Darker centers
        density = self.thickness_scale # e.g. 0.5
        transmission = np.exp(-thickness * density)
        
        if self.invert_gamma:
            # "Medical X-Ray" style: Bones are White, Empty is Dark
            # We invert the transmission
            val = 1.0 - transmission
            return Color(val, val, val)
        else:
            # "Glowing Gel" style: Thick parts absorb light (darker)
            # Thin edges let light through (brighter)
            val = transmission
            return Color(val, val, val)


# Compatibility shim: BasicLambertShading (keeps older tests happy)
class BasicLambertShading(ShadingStrategy):
    """Simple Lambertian shader used by unit tests and quick checks.

    This is intentionally lightweight and compatible with older test call signatures
    that pass a VObject instance instead of a HitInfo.
    """
    def shade(self, scene: Scene, ray: TracingRay, obj_or_hit, distance_or_depth, depth_or_dummy=None, trace_fn=None):
        # Accept either HitInfo-like object or VObject
        if hasattr(obj_or_hit, 'shape'):
            v_obj = obj_or_hit
            # Create a minimal color using ambient only
            ambient = getattr(scene, 'ambient_color', Color(0.03,0.03,0.03))
            intensity = getattr(scene, 'ambient_intensity', 0.1)
            mat = getattr(v_obj, 'material', None) or getattr(v_obj, 'shape', None) and getattr(v_obj.shape, 'material', None)
            albedo = getattr(mat, 'albedo', Color(1,1,1)) if mat is not None else Color(1,1,1)
            return ambient * intensity * albedo

        # If a HitInfo was passed, fall back to a safe black
        return Color(0.0, 0.0, 0.0)

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
        ray_generator: Optional[RayGenerator] = None,
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

        self.ray_generator: RayGenerator = ray_generator if ray_generator is not None else JitterRayGenerator()
        self.intersector: IntersectionStrategy = intersection_strategy if intersection_strategy is not None else RayMarchingIntersection()
        self.interactor: InteractionStrategy = interaction_strategy if interaction_strategy is not None else SimpleMaterialInteraction()
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
    ) -> List[Color]:
        self.stats.start_timer()
        
        cam = scene.camera
        if cam is None:
            raise ValueError("No camera provided")

        cam_w, cam_h = cam.width, cam.height
        
        self.sample_settings.width = cam_w
        self.sample_settings.height = cam_h

        # FIX 1: Storage - Use a simplified structure if possible, but we will stick 
        # to your list logic for now. 
        # Warning: High SPP will crash memory here.
        pixel_samples_and_colors = [[] for _ in range(cam_w * cam_h)]

        def _gen_rays(region=None, seed=None):
            r = self.ray_generator.generate(scene.camera, region=region, seed=seed)
            self.stats.rays_primary = len(r)
            return r

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
                final_color = self._trace_ray(scene, ray, self.max_depth, seed)
                
                if hasattr(ray, 'sample_u'):
                    s_u, s_v = ray.sample_u, ray.sample_v
                else:
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
        
        self.stats.stop_timer()
        self.stats.print_verbose_report()

        return pixel_colors

    def __repr__(self):
        return f"Raytracer(ray_generator={self.ray_generator}, intersector={self.intersector}, interactor={self.interactor}, shader={self.shader})"