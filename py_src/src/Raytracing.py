import numpy as np
import math
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from CommonUtils import unit, attenuate_distance_exponential
from PrimaryStructures import HitInfo, TracingRay
from Scene import Scene
from Camera import VCamera
from Geometry import Shape, VObject, AABB, BVHNode 
from Refractions import REFRACTIVE_INDICES
from Luminance import Color, PBRMaterial, MaterialType, calculate_fresnel_ratio, Color
from RenderingAlgorithims import Algorithm, RenderStats, register_algorithm, update_memory_stats
from Sampling import SamplingManager, SampleSettings, Sampler, Sample, reconstruct_pixel, RandomSampler, RandomSampler

# TODO: Pool tracing rays and hit info to reduce memory useage at runtime
# TODO: Localise stat updates, dont use global referenced up until the end of the logic
# TODO: Freeup large object that aren't in use
# TODO: Figure out how to simplify memory intensive logic into chunks

# Strategy interfaces for ray generation, intersection, and shading
class RayGenerationStrategy(ABC):
    @abstractmethod
    def generate(
        self,
        camera: VCamera,
        sampler: Sampler,
        region: Optional[Tuple[int, int, int, int]] = None,  # (x1, y1, w, h)
    ) -> List[TracingRay]:
        ...

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
        stats: Optional["TracingStats"] = None,
    ) -> HitInfo:
        ...

class InteractionStrategy(ABC):
    @abstractmethod
    def interact(
        self,
        ray: TracingRay,
        hit_info: HitInfo,
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
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
        self.shadow_samples = shadow_samples
        self.shadow_bias = float(shadow_bias)

    @abstractmethod
    def shade(
        self,
        scene: Scene,
        ray: TracingRay,
        hit_info: HitInfo,
        current_depth: int,
        trace_function: Callable[[Scene, TracingRay, int, Sampler], Color], 
        intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], HitInfo],
        interaction_function: Callable[[TracingRay, HitInfo, Sampler, Optional[float], Optional["TracingStats"]], Optional[TracingRay]],
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
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

        u1 = sampler.random_float()
        u2 = sampler.random_float()

        # We use sqrt(u1) to distribute points evenly by area (prevents clustering in the center)
        r = math.sqrt(u1) * radius
        theta = u2 * 2.0 * math.pi
        
        offset = tangent * (r * math.cos(theta)) + bitangent * (r * math.sin(theta))
        return center + offset

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
        sampler: Sampler,
        region: Optional[Tuple[int, int, int, int]] = None, # (x, y, w, h)
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

                pixel_samples = sampler.get_samples_per_pixel(x, y)
                for i, sample in enumerate(pixel_samples):
                    # Normalize sample coordinates to 0..1 for camera functions
                    screen_x = (x + sample.u) / float(cam_width)
                    screen_y = (y + sample.v) / float(cam_height)
                    
                    # 4. Calculate Ray Geometry
                    ray_origin, ray_orientation = camera.get_camera_ray(screen_x, screen_y)

                    # 5. Build Ray
                    ray = TracingRay(
                        origin=ray_origin,
                        orientation=ray_orientation,
                        pixel_x=x,
                        pixel_y=y,
                        sample_u=sample.u,
                        sample_v=sample.v,
                        name=f"ray#{i}_({x},{y})",
                        throughput=Color(1.0, 1.0, 1.0) # Used for path tracing accumulation
                    )
                    rays.append(ray)
        return rays

# Ray intersection implementations
class RayMarchingIntersection(IntersectionStrategy):
    """
    Find hits between rays and objects using the Ray Marching method.
    """
    def find_hit(
            self,
            scene: Scene,
            ray: TracingRay,
            stats: Optional["TracingStats"] = None
        ) -> HitInfo:
        # --- LOGIC BRANCH A: Ray is escaping an object ---
        if getattr(ray, "is_inside", False):
            closest_object, _ = scene.distance_estimator(ray.origin)

            if closest_object is not None:
                obj_matrix = closest_object.transform.get_global_matrix()
                inv_obj_matrix = np.linalg.inv(obj_matrix)
                
                _, exit_point = closest_object.shape.inverse_signed_distance(
                    ray.origin, 
                    ray.orientation,
                    64
                )
                
                if exit_point is not None:
                    dist_to_exit = np.linalg.norm(exit_point - ray.origin)
                    normal = None
                    if hasattr(closest_object.shape, "get_normal"):
                        normal = closest_object.shape.get_normal(exit_point)
                    else:
                        normal = np.array([0.0, 1.0, 0.0])

                    # Update stats
                    if stats is not None:
                        stats.triangle_tests += 1

                    # Return using the canonical HitInfo fields used elsewhere:
                    return HitInfo(
                        did_hit=True,
                        point=exit_point,
                        direction=ray.orientation,
                        normal=-normal,
                        distance=dist_to_exit,
                        obj=closest_object
                    )
            
            # If we failed to find an exit (infinite solid?), return Miss
            if stats is not None:
                stats.missed_rays += 1
            return HitInfo.miss()

        # --- LOGIC BRANCH B: Standard Raymarching (Outside) ---
        distance_traveled = 0.0

        for _ in range(self.max_steps):
            # Update cheap test counters per iteration
            if stats is not None:
                stats.aabb_tests += 1

            point = ray.point_at(distance_traveled)

            closest_object, distance_to_closest = scene.distance_estimator(point)

            # Optimization: If we marched into the void
            if closest_object is None:
                break
            
            # Hit Check
            if distance_to_closest <= self.epsilon:
                surface_normal = np.array([0.0, 1.0, 0.0])
                if hasattr(closest_object, "shape"):
                    surface_normal = closest_object.shape.get_normal(point)

                if stats is not None:
                    stats.triangle_tests += 1

                return HitInfo(
                    did_hit=True,
                    point=point,
                    direction=ray.orientation,
                    normal=surface_normal,
                    distance=distance_traveled,
                    obj=closest_object,
                )
            
            # Advance
            distance_traveled += distance_to_closest
            if distance_traveled >= self.max_distance:
                break

        if stats is not None:
            stats.missed_rays += 1

        return HitInfo.miss()

class InverseSDFIntersection(IntersectionStrategy):
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

    def find_hit(
            self,
            scene: Scene,
            ray: TracingRay,
            stats: Optional["TracingStats"] = None
        ) -> "HitInfo":
        closest_hit = HitInfo.miss()
        
        # We check every object in the scene independently
        for obj in scene.objects:
            # Skip objects that don't have an SDF shape defined
            if not hasattr(obj, 'shape'):
                continue

            # Attempt to intersect this specific object
            hit_info = self._intersect_object(obj, ray, stats)
            
            # Keep track of the closest hit only
            if hit_info.hit and hit_info.distance < closest_hit.distance:
                closest_hit = hit_info
                
        if not closest_hit.hit and stats is not None:
            stats.missed_rays += 1

        return closest_hit

    def _intersect_object(
            self,
            obj: VObject,
            ray: "TracingRay",
            stats: Optional["TracingStats"] = None
        ) -> "HitInfo":
        """
        Performs the 'Inverse SDF' logic:
        1. Transform Ray -> Local Space
        2. March in Local Space (Unscaled)
        3. Transform Hit -> World Space
        """
        # --- 1. Transform Ray to Local Space ---
        local_origin = obj.transform.inverse_transform_point(ray.origin)
        local_dir_raw = obj.transform.inverse_transform_direction(ray.orientation)
        
        dir_length = np.linalg.norm(local_dir_raw)
        if dir_length == 0:
            return HitInfo.miss()
            
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
            
            if stats is not None:
                stats.triangle_tests += 1

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
                    
                world_normal = obj.transform.transform_normal(local_normal)
                
                return HitInfo(
                    did_hit=True,
                    distance=world_distance,
                    point=p_world_hit,
                    normal=world_normal,
                    obj=obj
                )
            
            # STEP
            t += dist
            
            # Far Plane Check
            if t > self.max_distance:
                break
                
        return HitInfo.miss()

    def _calc_local_gradient(self, shape: Shape, p: np.ndarray) -> np.ndarray:
        """
        Calculates the normal in Local Space using central differences.
        """
        h = 1e-4 # Small step for gradient
        dx = np.array([h, 0, 0])
        dy = np.array([0, h, 0])
        dz = np.array([0, 0, h])
        
        grad = np.array([
            shape.signed_distance(p + dx) - shape.signed_distance(p - dx),
            shape.signed_distance(p + dy) - shape.signed_distance(p - dy),
            shape.signed_distance(p + dz) - shape.signed_distance(p - dz)
        ])
        
        # Normalize the gradient to get the normal
        norm = np.linalg.norm(grad)
        if norm > 0:
            return grad / norm
        return np.array([0.0, 1.0, 0.0]) # Fallback
    
class BVHIntersection(IntersectionStrategy):
    def __init__(self):
        super().__init__()
        self._cached_bvh_root: Optional[BVHNode] = None
        self._cached_scene_id: Optional[int] = None

    def find_hit(self, scene: Scene, ray: TracingRay, stats: Optional["TracingStats"] = None) -> HitInfo:
        # 1. Check if we need to build/rebuild the BVH
        # (We use id(scene.objects) as a cheap way to detect if the list changed)
        current_scene_id = id(scene.objects)
        if self._cached_bvh_root is None or current_scene_id != self._cached_scene_id:
            print(f"[BVH] Building Hierarchy for {len(scene.objects)} objects...")
            self._cached_bvh_root = self._build_bvh(scene.objects)
            self._cached_scene_id = current_scene_id

        # 2. Traverse
        closest_hit = HitInfo.miss()
        
        # Stack-based traversal (avoids recursion overhead)
        # Stack stores tuples: (Node, Distance_To_Box)
        stack = [(self._cached_bvh_root, 0.0)]
        
        while stack:
            node, dist_to_box = stack.pop()
            
            # Optimization: If the box is further than our current closest hit,
            # there is no point checking inside it.
            if closest_hit.hit and dist_to_box >= closest_hit.distance:
                continue
                
            if stats: stats.bvh_nodes_visited += 1

            # Case A: Leaf Node (Test Objects)
            if node.objects:
                for obj in node.objects:
                    # We reuse your existing per-object test logic here
                    # Assuming you have a basic geometric intersector or reuse the standard logic
                    hit = self._test_single_object(obj, ray, stats)
                    
                    if hit.hit:
                        if not closest_hit.hit or hit.distance < closest_hit.distance:
                            closest_hit = hit
                continue

            # Case B: Internal Node (Test Children)
            # We want to push the FAR child first, so we pop the CLOSE child later
            # This ensures we check the closest boxes first (Front-to-Back ordering)
            
            d_left = float('inf')
            d_right = float('inf')
            
            if node.left:
                if stats: stats.aabb_tests += 1
                d_left = node.left.box.intersect(ray)
                
            if node.right:
                if stats: stats.aabb_tests += 1
                d_right = node.right.box.intersect(ray)

            # Push valid children to stack
            # Push the furthest one first, so the closest is at top of stack
            if d_left != float('inf') and d_right != float('inf'):
                if d_left < d_right:
                    stack.append((node.right, d_right))
                    stack.append((node.left, d_left))
                else:
                    stack.append((node.left, d_left))
                    stack.append((node.right, d_right))
            elif d_left != float('inf'):
                stack.append((node.left, d_left))
            elif d_right != float('inf'):
                stack.append((node.right, d_right))

        if not closest_hit.hit and stats:
            stats.missed_rays += 1
            
        return closest_hit

    def _build_bvh(self, objects: list[VObject]) -> BVHNode:
        """
        Recursively splits the object list to build the tree.
        """
        node = BVHNode([])
        
        # 1. Calculate Bounds for all objects in this list
        # We cache AABBs for performance
        object_bounds = [(obj, AABB.from_object(obj)) for obj in objects]
        
        # Calculate Union of all bounds for this node
        if not object_bounds:
            node.box = AABB(np.zeros(3), np.zeros(3))
            return node

        first_box = object_bounds[0][1]
        node_min = first_box.min_point.copy()
        node_max = first_box.max_point.copy()
        
        for _, box in object_bounds:
            node_min = np.minimum(node_min, box.min_point)
            node_max = np.maximum(node_max, box.max_point)
        
        node.box = AABB(node_min, node_max)

        # 2. Leaf Condition
        # If few objects, stop splitting
        if len(objects) <= 2:
            node.objects = objects
            return node

        # 3. Split Strategy: Longest Axis
        # Find the widest dimension of the node's box
        extent = node_max - node_min
        axis = np.argmax(extent) # 0=x, 1=y, 2=z
        
        # Sort objects by their center along that axis
        object_bounds.sort(key=lambda item: (item[1].min_point[axis] + item[1].max_point[axis]) * 0.5)
        
        mid = len(objects) // 2
        
        sorted_objs = [item[0] for item in object_bounds]
        
        node.left = self._build_bvh(sorted_objs[:mid])
        node.right = self._build_bvh(sorted_objs[mid:])
        
        return node

    def _test_single_object(self, obj: VObject, ray: TracingRay, stats: Optional["TracingStats"]) -> HitInfo:
        """
        Hybrid Intersection:
        1. Transforms the World Ray into Object Local Space.
        2. Performs a small Ray March loop against the unit_signed_distance.
        3. Transforms the result back to World Space.
        """
        # --- 1. Transform Ray to Local Space ---
        # We move the ray, not the object. This lets us assume the object is 
        # always at (0,0,0) with identity rotation/scale.
        local_origin = obj.transform.inverse_transform_point(ray.origin)
        local_dir_raw = obj.transform.inverse_transform_direction(ray.orientation)
        
        # Normalize direction for correct SDF stepping
        dir_len = np.linalg.norm(local_dir_raw)
        if dir_len == 0: return HitInfo.miss()
        local_dir = local_dir_raw / dir_len

        # --- 2. Local Ray Marching ---
        t = 0.0
        max_local_march = 20.0 # Unit shapes are usually size ~1-2, so 20 is plenty
        local_epsilon = 1e-4   # Precision threshold
        
        # Optimization: Check if we are even close to the unit sphere/box 
        # before starting the march loop.
        # Simple sphere check: Is the ray passing within sqrt(3) distance of origin?
        # (Skipping for brevity, but recommended for production)

        # Handle "Inside" logic (if ray is inside the object, distance is negative)
        sign_modifier = -1.0 if getattr(ray, "is_inside", False) else 1.0

        for _ in range(64): # 64 steps is robust for convex shapes (Sphere/Cube)
            p = local_origin + (local_dir * t)
            
            if stats: stats.triangle_tests += 1

            # EVALUATE SDF
            dist = obj.shape.unit_signed_distance(p) * sign_modifier
            
            # HIT FOUND
            if dist < local_epsilon:
                # --- 3. Transform Back to World Space ---
                p_local = p
                
                # A. Calculate Normal in Local Space (Gradient)
                local_normal = self._calc_local_gradient(obj.shape, p_local)
                if ray.is_inside: 
                    local_normal = -local_normal # Flip normal if exiting
                
                # B. Transform Results
                p_world = obj.transform.transform_point(p_local)
                normal_world = obj.transform.transform_normal(local_normal)
                
                # C. Recalculate true world distance
                # (Surer than scaling 't' because of non-uniform scaling)
                dist_world = np.linalg.norm(p_world - ray.origin)
                
                return HitInfo(
                    did_hit=True,
                    distance=dist_world,
                    point=p_world,
                    normal=normal_world,
                    obj=obj
                )
            
            # STEP
            # Note: If the object has non-uniform scale (e.g. stretched sphere),
            # this step might be too large/small. For perfect accuracy, we should
            # divide 'dist' by the max scale factor, but direct stepping is usually visually fine.
            t += dist
            
            if t > max_local_march:
                break
                
        return HitInfo.miss()

    def _calc_local_gradient(self, shape: Shape, p: np.ndarray) -> np.ndarray:
        """
        Calculates surface normal using central differences on the SDF.
        """
        h = 1e-4
        dx = np.array([h, 0, 0])
        dy = np.array([0, h, 0])
        dz = np.array([0, 0, h])
        
        grad = np.array([
            shape.unit_signed_distance(p + dx) - shape.unit_signed_distance(p - dx),
            shape.unit_signed_distance(p + dy) - shape.unit_signed_distance(p - dy),
            shape.unit_signed_distance(p + dz) - shape.unit_signed_distance(p - dz)
        ])
        
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 0 else np.array([0.0, 1.0, 0.0])

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
            allow_passthrough: bool = False,   # If True, ray continues straight through
            opacity: float = 1.0,              # 1.0 = Solid, 0.0 = Invisible
        ):
        self.allow_passthrough = allow_passthrough
        self.opacity = max(0.0, min(opacity, 1.0)) # Clamp 0-1

    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Optional["TracingRay"]:
        
        # Case 1: Standard Terminal Interaction (Solid Wall)
        if not self.allow_passthrough:
            return None

        # Case 2: Passthrough / Ghost Mode
        # We handle opacity stochastically (Russian Roulette).
        # This is standard for path tracers to avoid branching rays.
        # If opacity is 0.5, there is a 50% chance the ray is absorbed (Hit),
        # and a 50% chance it passes through (Miss).
        if self.opacity > 0.0:
            # Generate a random number [0.0, 1.0)
            if sampler.random_float() < self.opacity:
                # Ray is absorbed by the "smoke/ghost"
                return None

        # If we survive the opacity check (or opacity is 0), the ray passes through.
        # It continues in the EXACT same direction.
        next_origin = hit_info.point + (ray.orientation * bias)
        
        if stats is not None:
            stats.rays_transparency += 1

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
        stats: Optional["TracingStats"] = None
    ) -> Optional["TracingRay"]:
        # 1. Stochastic Opacity Check (Russian Roulette)
        # If opacity is 0.5, 50% of rays are blocked, 50% pass through.
        # This simulates semi-transparency without splitting rays.
        if self.opacity_cutoff > 0.0:
            if sampler.random_float() < self.opacity_cutoff:
                return None # Ray is absorbed/blocked by the "smoke"

        # 2. Passthrough Logic
        # We spawn a new ray continuing in the exact same direction.
        next_origin = hit_info.point + (ray.orientation * bias)
        
        if stats is not None:
            stats.rays_transparency += 1

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
        scene_ior: float = REFRACTIVE_INDICES["air"],
    ):
        self.scene_ior = scene_ior

    def interact(
        self, 
        ray: TracingRay, 
        hit_info: HitInfo, 
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Optional[TracingRay]:
        """
        Decide between reflection and refraction (Russian roulette) and generate the next ray.
        Updates ray depth/is_inside and conservative tracing stats counters.
        """
        material: PBRMaterial = getattr(hit_info.obj, "material", None) or getattr(hit_info.obj.shape, "material", None) if hasattr(hit_info.obj, "shape") else None
        if not material:
            return None

        # Surface normal (fall back to up)
        normal = getattr(hit_info, "normal", np.array([0.0, 0.0, 1.0]))

        # Indices of refraction (fallback to 1.0)
        n1 = self.scene_ior
        n2 = getattr(material.data, "ior", self.scene_ior + 1e-2)

        # Fresnel-based decision
        reflection_chance = calculate_fresnel_ratio(
            ray.orientation, normal, n1, n2
        )

        do_refract = sampler.random_float() > reflection_chance

        new_ray = None
        pdf = 0.0

        # Try refraction first if decision says so
        if do_refract:
            new_ray, pdf = material.calculate_microfacet_refraction_ray(
                direction=ray.orientation,
                surface_normal=normal,
                new_origin=hit_info.point + normal * bias,
                sampler=sampler,
                ior_incident=n1,
                ior_transmitted=n2
            )
            # Total internal reflection: fall back to reflection
            if new_ray is None:
                do_refract = False

        # Reflection path (or fallback)
        if not do_refract:
            new_ray, pdf = material.calculate_microfacet_reflection_ray(
                direction=ray.orientation,
                surface_normal=normal,
                new_origin=hit_info.point + normal * bias,
                sampler=sampler
            )

        if new_ray is not None:
            print(ray)
            new_ray.depth = ray.depth + 1

            # Toggle inside flag on refraction
            if do_refract:
                new_ray.is_inside = not getattr(ray, "is_inside", False)
                if stats is not None:
                    stats.rays_refraction += 1
            else:
                new_ray.is_inside = getattr(ray, "is_inside", False)
                if stats is not None:
                    stats.rays_reflection += 1

            # Reset throughput for a fresh path (material handles weighting)
            new_ray.throughput = Color(1.0, 1.0, 1.0)
            new_ray.pdf = pdf

        return new_ray

# Shading implementations
class BasicLambertShading(ShadingStrategy):
    """
    Simple Lambertian shader used by unit tests and quick checks.
    """
    def shade(
            self,
            scene: "Scene",
            ray: "TracingRay",
            hit_info: "HitInfo",
            current_depth: int,
            trace_function: Callable[[Scene, TracingRay, int, Sampler], Color], 
            intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], HitInfo],
            interaction_function: Callable[[TracingRay, HitInfo, Sampler, Optional[float], Optional["TracingStats"]], Optional[TracingRay]],
            sampler: Sampler,
            bias: float = 1e-4,
            stats: Optional["TracingStats"] = None
        ) -> Color:
        if not (hit_info.hit or hasattr(hit_info, "obj")):
            return Color(0.0, 1.0, 1.0) # Cyan for missing object
        
        v_obj: VObject = hit_info.obj

        mat = getattr(v_obj, 'material', None)
        if mat is None:
            return Color(1.0, 0.0, 1.0) # Pink for missing material
            
        # Create a minimal color using ambient only
        ambient: Color = getattr(scene, 'ambient_color', Color(0.03, 0.03, 0.03))
        intensity: float = getattr(scene, 'ambient_intensity', 0.1)
            
        albedo: Color = getattr(mat, 'albedo', Color(1.0, 1.0, 1.0)) if mat is not None else Color(1.0, 1.0, 1.0)
        return ambient * intensity * albedo

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
            trace_function: Callable[[Scene, TracingRay, int, Sampler], Color], 
            intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], HitInfo],
            interaction_function: Callable[[TracingRay, HitInfo, Sampler, Optional[float], Optional["TracingStats"]], Optional[TracingRay]],
            sampler: Sampler,
            bias: float = 1e-4,
            stats: Optional["TracingStats"] = None
        ) -> Color:
        if not (hit_info.hit or hasattr(hit_info, "obj")):
            return Color(0.0, 1.0, 1.0) # Cyan for missing object

        v_object: VObject = hit_info.obj
        material: PBRMaterial = v_object.material
        if material is None:
            return Color(1.0, 0.0, 1.0) # Pink for missing material
        
        # --- 1. Emission (The Glow) ---
        # Added first so even if we stop recursing, we see the light.
        final_color = Color(0.0, 0.0, 0.0)
        if material.type == MaterialType.EMISSIVE:
            final_color += material.get_emissive_component()
            # Emissive materials usually don't reflect/receive shadow, so we can return early
            return final_color

        # --- 2. Direct Lighting (Next Event Estimation) ---
        normal = unit(hit_info.normal)
        view_dir = -unit(ray.orientation)
        
        # We assume 'evaluate_direct_light' handles the loop over lights & shadow checks
        def visibility_checker(p1, p2):
             if not self.enable_shadows: return 1.0
             if stats is not None:
                 stats.lights_sampled += 1
                 stats.rays_shadow += 1
             return 0.0 if scene.is_occluded(p1, p2, self.shadow_bias) else 1.0

        direct_light = material.evaluate_direct_light(
            scene.get_lights(), hit_info, view_dir, visibility_checker
        )

        # --- 3. Indirect Lighting (Recursion) ---
        indirect_light = Color(0.0, 0.0, 0.0)

        # The 'interaction_function' calculates the physics of the bounce.
        # It performs Russian Roulette (Reflect OR Refract) and sets 'new_ray.throughput'.
        if current_depth > 0:
            probe_color = Color(1.0, 1.0, 1.0)

            new_ray = interaction_function(ray, hit_info, sampler, bias, stats)

            if new_ray is not None:
                if stats is not None:
                    stats.max_depth_reached = max(stats.max_depth_reached, getattr(new_ray, "depth", 0))

                incoming_light = trace_function(scene, new_ray, current_depth - 1, sampler)
                
                indirect_light: Color = incoming_light * new_ray.throughput
            else:
                indirect_light = probe_color

        final_color = direct_light + indirect_light

        if self.ambient_enabled:
            final_color += material.get_ambient_color(scene.ambient_color, scene.ambient_intensity)
        
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
        trace_function: Callable[[Scene, TracingRay, int, Sampler], Color], 
        intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], HitInfo],
        interaction_function: Callable[[TracingRay, HitInfo, Sampler, Optional[float], Optional["TracingStats"]], Optional[TracingRay]],
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Color:
        # 1. Setup Vectors
        # Normal pointing OUT of the surface
        normal = hit_info.normal
        view_dir = unit(-ray.orientation)

        # 2. Calculate Thickness (The "Through" Ray)
        # We push the origin slightly INSIDE the object (opposite to normal) to avoid self-intersection at the entry.
        # Note: If geometry is single-sided planes, this might fail. Assumes closed volume.
        inside_origin = hit_info.point - (normal * bias) 
        
        inside_ray = TracingRay(
            origin=inside_origin, 
            orientation=ray.orientation,
            depth=ray.depth, # Keep depth to prevent infinite recursion bugs
            is_inside=True      # Mark this ray as originating INSIDE the object so the intersector treats it as an exit ray
        )

        # Find where the ray leaves the object (pass stats along)
        exit_hit = intersection_function(scene, inside_ray, stats)

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

# Stats for ray tracing
@dataclass(slots=True)
class TracingStats(RenderStats):
    # --- Basic Counters ---
    rays_primary: int = 0
    rays_shadow: int = 0
    rays_reflection: int = 0
    rays_refraction: int = 0
    rays_transparency: int = 0  # NEW: Rays passing through alpha cutouts
    missed_rays: int = 0
    
    # --- Intersection Performance (BVH Health) ---
    aabb_tests: int = 0         # Box hits
    bvh_nodes_visited: int = 0  # NEW: Total tree nodes traversed
    triangle_tests: int = 0     # Actual triangle math
    
    # --- Path Tracing Diagnosis ---
    max_depth_reached: int = 0
    roulette_kills: int = 0
    lights_sampled: int = 0
    
    # --- Logic & Debugging ---
    pixels_processed: int = 0
    nan_errors: int = 0      

    @property
    def total_rays(self) -> int:
        return (self.rays_primary + self.rays_shadow + 
                self.rays_reflection + self.rays_refraction + 
                self.rays_transparency)

    @property
    def intersections_per_ray(self) -> float:
        """
        Efficiency Metric: A lower number is better.
        If this is > 50-100, your BVH might be broken.
        """
        if self.total_rays == 0: return 0.0
        return (self.aabb_tests + self.triangle_tests) / self.total_rays

    @property
    def culling_efficiency(self) -> float:
        """
        NEW: Measures how well the BVH protects us from triangle tests.
        Ratio of Box Tests to Triangle Tests.
        High = Good (we test many cheap boxes to find few expensive triangles).
        Low (near 1.0) = Bad (We are testing triangles for every box we hit).
        """
        if self.triangle_tests == 0: return 0.0
        return self.aabb_tests / self.triangle_tests

    def reset_ray_counter(self):
        # Reset all counters (useful for multi-pass rendering)
        for field_name in self.__slots__:
            if isinstance(getattr(self, field_name), (int, float)):
                setattr(self, field_name, 0)

    def __iadd__(self, other: 'TracingStats') -> 'TracingStats':
        # Accumulate all integer fields automatically
        # This prevents missing a field when you add new metrics later
        for s in self.__slots__:
            val_self = getattr(self, s)
            val_other = getattr(other, s)
            if isinstance(val_self, int) and isinstance(val_other, int):
                setattr(self, s, val_self + val_other)
        
        # Handle manual updates for non-sum fields
        self.time_taken_seconds += other.time_taken_seconds
        self.memory_usage = max(self.memory_usage, other.memory_usage)
        self.max_depth_reached = max(self.max_depth_reached, other.max_depth_reached)
        return self

    def print_verbose_report(self):
        print(f"\n=== Extended Tracing Stats ===")
        print(f"Time: {self.time_taken_seconds:.3f}s | Mem: {self.memory_usage:.2f}MB")
        print(f"-------------------------")
        print(f"Ray Traffic:")
        print(f"  - Total:       {self.total_rays:,}")
        print(f"  - Primary:     {self.rays_primary:<10,} ({self.rays_primary/max(1,self.total_rays)*100:.1f}%)")
        print(f"  - Shadow:      {self.rays_shadow:<10,} (Lights used: {self.lights_sampled:,})")
        print(f"  - Bounce:      {(self.rays_reflection+self.rays_refraction):<10,}")
        print(f"-------------------------")
        print(f"BVH Health:")
        print(f"  - AABB Tests:      {self.aabb_tests:,}")
        print(f"  - Tri Tests:       {self.triangle_tests:,}")
        print(f"  - Nodes Visited:   {self.bvh_nodes_visited:,}")
        print(f"  - Cost/Ray:        {self.intersections_per_ray:.2f} (Target: < 50)")
        print(f"  - Culling Ratio:   {self.culling_efficiency:.2f} (Target: > 2.0)")
        print(f"-------------------------")
        print(f"Path Diagnostics:")
        print(f"  - Max Depth Hit:   {self.max_depth_reached}")
        print(f"  - Roulette Kills:  {self.roulette_kills:,}")
        print(f"  - NaN Errors:      {self.nan_errors}")

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

    def _trace_ray(self, scene: Scene, ray: TracingRay, depth: int, sampler: Sampler) -> Color:
        """
        The recursive engine. It takes a ray, finds what it hits, and calculates the color.
        If the surface is reflective, this function will be called again by the shader.
        """
        if depth < 0:
            return Color(1.0, 0.0, 0.0) # Red for depth error

        hit_info = self.intersector.find_hit(scene, ray, self.stats)

        self.stats = update_memory_stats(self.stats)

        if hit_info.hit and hit_info.obj is not None:
            return self.shader.shade(
                scene=scene, 
                ray=ray, 
                hit_info=hit_info,
                current_depth=depth,
                trace_function=self._trace_ray,
                intersection_function=self.intersector.find_hit,
                interaction_function=self.interactor.interact,
                sampler=sampler,
                stats=self.stats
            )

        # The ray missed all scene objects
        if self.enable_scene_background:
            return self.custom_background
        return scene.get_background_color(ray.orientation)

    def render(
        self,
        scene: Scene,
        sampler: Optional[Sampler] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        tile_size: Optional[int] = None,
    ) -> List[Color]:
        self.stats.reset_ray_counter()
        self.stats.start_timer()

        camera = scene.camera
        if camera is None: raise ValueError("No camera provided in Scene")
        cam_width, cam_height = camera.width, camera.height

        if region:
            rx, ry, rw, rh = region
        else:
            rx, ry, rw, rh = 0, 0, cam_width, cam_height

        # Initialize output image and sample counter
        full_image_pixels = [Color(0.0, 0.0, 0.0) for _ in range(rw * rh)]
        self.sample_settings.width = rw
        self.sample_settings.height = rh
        pixels_processed = 0

        # Handle tile_size parameter
        if tile_size is None:
            ts = 64
        else:
            ts = tile_size

        # Create default sampler if not provided
        if sampler is None:
            sampler = RandomSampler(self.sample_settings)

        for tile_y in range(ry, ry + rh, ts):
            for tile_x in range(rx, rx + rw, ts):
                # Calculate current tile dimensions (handle edges)
                current_w = min(ts, (rx + rw) - tile_x)
                current_h = min(ts, (ry + rh) - tile_y)
                
                # Define tile region: (x, y, w, h)
                tile_region = (tile_x, tile_y, current_w, current_h)
                
                # 1. Generate Rays for this Tile ONLY
                rays = self.ray_generator.generate(camera=camera, region=tile_region, sampler=sampler)
                self.stats.rays_primary += len(rays)

                # 2. Local Storage for this Tile
                # Map: (local_tile_index) -> List[(Sample, Color)]
                tile_samples = [[] for _ in range(current_w * current_h)]

                # 3. Trace Rays
                for ray in rays:
                    if ray is None: continue
                    
                    # Global Coordinates
                    px, py = ray.pixel_x, ray.pixel_y
                    
                    # Convert to Local Tile Coordinates
                    local_x = px - tile_x
                    local_y = py - tile_y

                    # Safety Check
                    if not (0 <= local_x < current_w and 0 <= local_y < current_h):
                        continue

                    # Trace
                    pixel_color = self._trace_ray(scene, ray, self.max_depth, sampler)

                    # Create Sample Object
                    s_u = getattr(ray, "sample_u", (px + 0.5) / cam_width)
                    s_v = getattr(ray, "sample_v", (py + 0.5) / cam_height)
                    sample = Sample(s_u, s_v, 1.0) # weight 1.0

                    # Store in Local Tile Buffer
                    local_idx = local_y * current_w + local_x
                    tile_samples[local_idx].append((sample, pixel_color))
                    pixels_processed += 1

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
                    # Convert Color objects to RGBA arrays
                    colors = []
                    for sc in samples_and_colors:
                        color_obj: Color = sc[1]
                        color_array = color_obj.to_np_array(include_alpha=True)
                        colors.append(color_array)
                    
                    rec_rgb = reconstruct_pixel(global_x, global_y, samples, colors, self.sample_settings)
                    final_color = Color(*rec_rgb)

                    # Write to Final Image Buffer
                    # Calculate index in the *region* buffer
                    final_idx = (global_y - ry) * rw + (global_x - rx)
                    full_image_pixels[final_idx] = final_color

        self.stats.pixels_processed = pixels_processed
        self.stats.stop_timer()

        return full_image_pixels
