import numpy as np
import math
from typing import Optional, List, Tuple, Callable, cast
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from CommonUtils import unit, attenuate_distance_exponential
from PrimaryStructures import Transform, TracingRay, HitInfo
from Scene import Scene
from Camera import VCamera
from Geometry import Shape, VObject, AABB, BVHNode 
from Refractions import REFRACTIVE_INDICES
from Luminance import Color, PBRMaterial, MaterialType, LightSource, calculate_fresnel_ratio
from RenderingAlgorithms import Algorithm, RenderStats, register_algorithm, update_memory_stats
from Sampling import SamplingManager, SampleSettings, Sampler, Sample, reconstruct_pixel, RandomSampler

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
        interaction_function: Callable[[TracingRay, HitInfo, Sampler, float, Optional["TracingStats"]], Optional[TracingRay]],
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
                surface_normal = np.array([0.0, 0.0, 1.0])
                closest_object_shape = getattr(closest_object, "shape", None) 
                if closest_object_shape is not None:
                    surface_normal = closest_object_shape.get_normal(point)

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
            cam: VCamera = cast(VCamera,getattr(scene, "camera", None))
            closest_object_transform = getattr(closest_object, "transform", Transform(np.zeros(3), np.zeros(3), np.ones(3)))
            if cam is not None:
                far_plane_distance = np.linalg.norm(cam.transform.position - closest_object_transform.position)
                if distance_traveled >= self.max_distance or far_plane_distance >= cam.far:
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
            hit_obj = getattr(hit_info, "obj", None)
            if hit_obj is None:
                break

            cam = getattr(scene, "camera", None)
            if cam is None:
                break

            far_plane_distance = np.linalg.norm(cam.transform.position - hit_obj.transform.position)
            if hit_info.hit and (hit_info.distance < closest_hit.distance or hit_info.distance < far_plane_distance) :
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
        obj_shape = getattr(obj, "shape", None)
        if obj_shape is None:
            return HitInfo.miss()

        # --- 2. Raymarch Loop ---
        t = 0.0
        
        # Check for "Inside-Out" logic (for X-ray/Dielectrics)
        # If we are inside, we treat negative distance as empty space (flip sign)
        sign_modifier = -1.0 if ray.is_inside else 1.0
        
        for _ in range(self.max_steps):
            # 1. Calculate World Point
            p = ray.point_at(t)
            
            # 2. Sample the Object's SDF (It handles the transform internally)
            raw_dist = obj_shape.signed_distance(p)
            
            if stats is not None:
                stats.triangle_tests += 1

            # Apply Modifier (flips distance if inside)
            dist = raw_dist * sign_modifier
            
            # 3. Hit Check
            if dist < self.epsilon:
                normal = self._calc_local_gradient(obj_shape, p)
                
                if ray.is_inside:
                    normal = -normal
                    
                return HitInfo(
                    did_hit=True,
                    distance=t,
                    point=p,
                    normal=normal,
                    obj=obj
                )
            
            # 4. Step
            t += dist * self.step_relaxation
            
            if t > self.max_distance:
                break
                
        return HitInfo.miss()

    def _calc_local_gradient(self, shape: Shape, p: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """
        Calculates the normal in Local Space using central differences.
        """
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
        return np.array([0.0, 0.0, 1.0]) # Fallback
    
class BVHIntersection(IntersectionStrategy):
    
    def __init__(
            self,
            epsilon: float = 1e-4,
            max_steps: int = 64,
            max_distance: float = 50
        ):
        super().__init__(epsilon, max_steps, max_distance)
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
                l_box = getattr(node.left, "box", None)
                if l_box is None:
                    continue
                d_left = l_box.intersect(ray)
                
            if node.right:
                if stats: stats.aabb_tests += 1
                r_box = getattr(node.right, "box", None)
                if r_box is None:
                    continue
                d_right = r_box.intersect(ray)


            l_node = getattr(node, "left", None)
            r_node = getattr(node, "right", None)

            if l_node is None or r_node is None:
                continue

            # Push valid children to stack
            # Push the furthest one first, so the closest is at top of stack
            if d_left != float('inf') and d_right != float('inf'):
                if d_left < d_right:
                    stack.append((r_node, d_right))
                    stack.append((l_node, d_left))
                else:
                    stack.append((l_node, d_left))
                    stack.append((r_node, d_right))
            elif d_left != float('inf'):
                stack.append((l_node, d_left))
            elif d_right != float('inf'):
                stack.append((r_node, d_right))

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
        obj_shape = getattr(obj, "shape", None)
        if obj_shape is None:
            return HitInfo.miss()
        
        # 1. Transform Ray to Local Space
        obj_transform = cast(Transform, getattr(obj, "transform", Transform(np.zeros(3), np.zeros(3), np.ones(3))))
        local_ray = obj_transform.inverse_transform_ray(ray)

        # 2. Get Minimum Scale Factor for safe stepping
        # If we are in local space, a distance of 1.0 might mean 0.1 in world space (if scaled down).
        # We must multiply the local distance by the smallest scale to avoid overstepping.
        scale = obj_transform.scale
        min_scale = min(abs(scale[0]), abs(scale[1]), abs(scale[2]))

        t = 0.0
        sign_modifier = -1.0 if getattr(ray, "is_inside", False) else 1.0

        for _ in range(self.max_steps): 
            p = local_ray.point_at(t)
            
            if stats: stats.triangle_tests += 1

            # FIX: Use unit_signed_distance (Local Logic)
            dist_local = obj_shape.unit_signed_distance(p) * sign_modifier
            
            # Check convergence
            if dist_local < self.epsilon:
                # --- Hit Found ---
                
                # A. Transform Point to World
                p_world = obj_transform.transform_point(p)
                
                # B. Calculate Normal (Local Gradient -> World Normal)
                local_normal = self._calc_local_gradient(obj_shape, p)
                if ray.is_inside: 
                    local_normal = -local_normal
                normal_world = obj_transform.transform_normal(local_normal)
                
                # C. True World Distance
                distance_world = np.linalg.norm(p_world - ray.origin)
                
                return HitInfo(
                    did_hit=True,
                    distance=float(distance_world),
                    point=p_world,
                    normal=normal_world,
                    obj=obj
                )
            
            t += dist_local / min_scale
            
            if t >= self.max_distance:
                break
                
        return HitInfo.miss()

    def _calc_local_gradient(self, shape: Shape, p: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """
        Calculates surface normal using central differences on the SDF.
        """
        dx = np.array([h, 0, 0])
        dy = np.array([0, h, 0])
        dz = np.array([0, 0, h])
        
        grad = np.array([
            shape.unit_signed_distance(p + dx) - shape.unit_signed_distance(p - dx),
            shape.unit_signed_distance(p + dy) - shape.unit_signed_distance(p - dy),
            shape.unit_signed_distance(p + dz) - shape.unit_signed_distance(p - dz)
        ])
        
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

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
        hit_point = getattr(hit_info, "point", None)
        if hit_point is None:
            return None
        
        next_origin = hit_point + (ray.orientation * bias)
        
        if stats is not None:
            stats.rays_transparency += 1

        return replace(ray, origin=next_origin)

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
        hit_point = getattr(hit_info, "point", None)
        if hit_point is None:
            return None
        
        next_origin = hit_point + (ray.orientation * bias)
        
        if stats is not None:
            stats.rays_transparency += 1

        return replace(ray, origin=next_origin)

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
        v_obj = hit_info.obj
        material = getattr(v_obj, "material", None)
        if material is None:
            return None

        # Surface normal (fall back to up)
        normal = getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0]))
        hit_point = getattr(hit_info, "point", None)
        if hit_point is None:
            return None

        # Indices of refraction (fallback to 1.0)
        n1 = self.scene_ior
        n2 = getattr(material.data, "ior", self.scene_ior + 1e-2)

        # Fresnel-based decision
        reflection_chance = calculate_fresnel_ratio(
            ray.orientation, normal, n1, n2
        )

        do_refract = sampler.random_float() > reflection_chance

        new_ray = None

        # Try refraction first if decision says so
        if do_refract:
            new_ray = material.calculate_microfacet_refraction_ray(
                direction=ray.orientation,
                surface_normal=normal,
                new_origin=hit_point - normal * bias,
                sampler=sampler,
                ior_incident=n1,
                ior_transmitted=n2
            )
            # Total internal reflection: fall back to reflection
            if new_ray is None:
                do_refract = False

        # Reflection path (or fallback)
        if not do_refract:
            new_ray = material.calculate_microfacet_reflection_ray(
                direction=ray.orientation,
                surface_normal=normal,
                new_origin=hit_point + normal * bias,
                sampler=sampler
            )

        if new_ray is not None:
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
            new_ray.throughput = [1.0, 1.0, 1.0]

        return new_ray

# Shading implementations
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
    def shade(
            self,
            scene: Scene,
            ray: TracingRay,
            hit_info: HitInfo,
            current_depth: int,
            trace_function: Callable, 
            intersection_function: Callable,
            interaction_function: Callable,
            sampler: Sampler,
            bias: float = 1e-4,
            stats: Optional["TracingStats"] = None
        ) -> Color:
        
        # If we missed, return black (or background)
        if not (hit_info.hit and hit_info.obj):
            return Color(0.0, 0.0, 0.0)
            
        # 1. Get Normal
        normal = getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0]))
        
        # 2. Map from range [-1, 1] to [0, 1] for color display
        # Normal (0,0,0) becomes Gray (0.5, 0.5, 0.5)
        r = (normal[0] + 1.0) * 0.5
        g = (normal[1] + 1.0) * 0.5
        b = (normal[2] + 1.0) * 0.5
        
        return Color(r, g, b)


class DepthShading(ShadingStrategy):
    """
    Debug shader that visualizes the distance from the camera to the object.
    
    Objects closer than 'min_depth' are White.
    Objects further than 'max_depth' are Black.
    Gradient in between.
    """
    def __init__(self, min_depth: float = 0.0, max_depth: float = 20.0):
        super().__init__()
        self.min_dist = min_depth
        self.max_dist = max_depth

    def shade(
            self,
            scene: Scene,
            ray: TracingRay,
            hit_info: HitInfo,
            current_depth: int,
            *args, **kwargs
        ) -> Color:
        
        if not hit_info.hit:
            return Color(0.0, 0.0, 0.0) # Background is "infinite" depth (black)

        dist = hit_info.distance
        
        # Normalize distance to 0.0 - 1.0
        # Formula: (dist - min) / (max - min)
        range_dist = self.max_dist - self.min_dist
        if range_dist == 0: range_dist = 1.0
        
        normalized = (dist - self.min_dist) / range_dist
        
        # Clamp between 0 and 1
        normalized = max(0.0, min(1.0, normalized))
        
        # Invert so Close = Bright, Far = Dark
        val = 1.0 - normalized
        
        return Color(val, val, val)

class FlatShading(ShadingStrategy):
    """
    Renders objects with their raw Albedo color only. 
    No lighting, no shadows, no recursion. 
    Fastest possible render mode.
    """
    def shade(
            self,
            scene: Scene,
            ray: TracingRay,
            hit_info: HitInfo,
            current_depth: int,
            *args, **kwargs
        ) -> Color:
        
        if not (hit_info.hit and hit_info.obj):
            orientation = getattr(ray, "orientation", [0.0, 0.0, -1.0])
            return scene.get_background_color(orientation)

        mat = hit_info.obj.material
        if mat is None:
            return Color(1.0, 0.0, 1.0) # Error Pink

        # Just return the base color (Albedo)
        return getattr(mat, 'albedo', Color(1.0, 1.0, 1.0))

class BasicLambertShading(ShadingStrategy):
    """
    Simple Lambertian shader.
    """
    def shade(
            self,
            scene: "Scene",
            ray: "TracingRay",
            hit_info: "HitInfo",
            current_depth: int,
            trace_function: Callable[[Scene, TracingRay, int, Sampler], Color], 
            intersection_function: Callable[[Scene, TracingRay, Optional["TracingStats"]], HitInfo],
            interaction_function: Callable[[TracingRay, HitInfo, Sampler, float, Optional["TracingStats"]], Optional[TracingRay]],
            sampler: Sampler,
            bias: float = 1e-4,
            stats: Optional["TracingStats"] = None
        ) -> Color:
        if not (hit_info.hit or hasattr(hit_info, "obj")):
            return Color(0.0, 0.0, 0.0)
        
        mat = getattr(hit_info.obj, 'material', None)
        if mat is None:
            return Color(1.0, 0.0, 1.0) # Pink for missing material
            
        # Create a minimal color using ambient only
        ambient: Color = getattr(scene, 'ambient_color', Color(0.03, 0.03, 0.03))
        intensity: float = getattr(scene, 'ambient_intensity', 0.1)
            
        albedo: Color = getattr(mat, 'albedo', Color(1.0, 1.0, 1.0)) if mat is not None else Color(1.0, 1.0, 1.0)
        return ambient * intensity * albedo

    def _calculate_shadow_visibility(self, scene: Scene, point: np.ndarray, light: LightSource, light_dir: np.ndarray, sampler: Sampler):
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
                    sample_pos = self._random_point_on_disc(light.position, -light_dir, float(radius), sampler)
                    if not scene.is_occluded(point, sample_pos, bias=self.shadow_bias):
                        visible_count += 1
                visibility = visible_count / float(self.shadow_samples)

        return visibility

class RecursiveLambertShading(BasicLambertShading):
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
            interaction_function: Callable[[TracingRay, HitInfo, Sampler, float, Optional["TracingStats"]], Optional[TracingRay]],
            sampler: Sampler,
            bias: float = 1e-4,
            stats: Optional["TracingStats"] = None
        ) -> Color:
        if not (hit_info.hit or hasattr(hit_info, "obj")):
            return Color(0.0, 1.0, 1.0) # Cyan for missing object

        material = getattr(hit_info.obj, "material", None)
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
        view_dir = -unit(ray.orientation)
        
        def visibility_function(point: np.ndarray, light: LightSource):
            return self._calculate_shadow_visibility(scene, point, light, light.get_light_direction(point), sampler)

        direct_light = material.evaluate_direct_light(
            scene.get_lights(), hit_info, view_dir, visibility_function
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
                
                indirect_light = incoming_light * Color(*new_ray.throughput)
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
        interaction_function: Callable[[TracingRay, HitInfo, Sampler, float, Optional["TracingStats"]], Optional[TracingRay]],
        sampler: Sampler,
        bias: float = 1e-4,
        stats: Optional["TracingStats"] = None
    ) -> Color:
        # 1. Setup Vectors
        # Normal pointing OUT of the surface
        normal = getattr(hit_info, "normal", np.array([0.0, 1.0, 0.0]))
        hit_point = getattr(hit_info, "point", None)
        if hit_point is None:
            return Color(0.0, 0.0, 0.0)
        
        view_dir = unit(-ray.orientation)

        # 2. Calculate Thickness (The "Through" Ray)
        # We push the origin slightly INSIDE the object (opposite to normal) to avoid self-intersection at the entry.
        # Note: If geometry is single-sided planes, this might fail. Assumes closed volume.
        inside_origin = hit_point - (normal * bias) 
        
        inside_ray = replace(ray, origin=inside_origin, is_inside=True)      # Mark this ray as originating INSIDE the object so the intersector treats it as an exit ray

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

    def format_report(self) -> str:
        """
        Generates a formatted string report suitable for saving to a .txt file.
        """
        lines = []
        lines.append(f"=== Tracing Stats ===")
        lines.append(f"Time: {self.time_taken_seconds:.3f}s | Mem: {self.memory_usage:.2f}MB")
        lines.append(f"-------------------------")
        lines.append(f"Ray Traffic:")
        lines.append(f"  - Total:       {self.total_rays:,}")
        lines.append(f"  - Primary:     {self.rays_primary:<10,} ({self.rays_primary/max(1,self.total_rays)*100:.1f}%)")
        lines.append(f"  - Shadow:      {self.rays_shadow:<10,} (Lights used: {self.lights_sampled:,})")
        lines.append(f"  - Bounce:      {(self.rays_reflection+self.rays_refraction):<10,}")
        lines.append(f"-------------------------")
        lines.append(f"BVH Health:")
        lines.append(f"  - AABB Tests:      {self.aabb_tests:,}")
        lines.append(f"  - Tri Tests:       {self.triangle_tests:,}")
        lines.append(f"  - Nodes Visited:   {self.bvh_nodes_visited:,}")
        lines.append(f"  - Cost/Ray:        {self.intersections_per_ray:.2f} (Target: < 50)")
        lines.append(f"  - Culling Ratio:   {self.culling_efficiency:.2f} (Target: > 2.0)")
        lines.append(f"-------------------------")
        lines.append(f"Path Diagnostics:")
        lines.append(f"  - Max Depth Hit:   {self.max_depth_reached}")
        lines.append(f"  - Roulette Kills:  {self.roulette_kills:,}")
        lines.append(f"  - NaN Errors:      {self.nan_errors}")
        
        return "\n".join(lines)
    
    def print_verbose_report(self):
        print()
        print(self.format_report())

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
            return self.custom_background if self.custom_background is not None else Color(0.0, 0.0, 0.0)
        orientation = getattr(ray, "orientation", [0.0, 0.0, -1.0])
        return scene.get_background_color(orientation)

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
        ts = tile_size if tile_size is not None else 64

        # Create default sampler if not provided
        if sampler is None:
            sampler = RandomSampler(self.sample_settings)

        # Optional: Print total tiles for progress tracking
        # total_tiles = ((rw + ts - 1) // ts) * ((rh + ts - 1) // ts)
        # tile_count = 0

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


                self.stats = update_memory_stats(self.stats)

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
