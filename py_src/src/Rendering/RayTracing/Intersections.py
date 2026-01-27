from __future__ import annotations
from token import OP
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Tuple, cast
from dataclasses import dataclass, field, replace

from src.Data.Transform import Transform
from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from src.Geometry.BVH import BVHNode, build_bvh_tree
from src.Geometry.SDF import SignedDistanceShape
from src.Utilities.Common import unit
from src.Data.Scene import Scene

if TYPE_CHECKING:
    from src.Data.Scene import SceneNode
    from src.Data.Context import LightContext, SDFContext, MeshContext
    from .Core import TracingStats

@dataclass
class IntersectionSettings:
    epsilon: float = 1e-4          # Hit Threshold
    max_steps: int = 128           # Performance Cap
    max_distance: float = 1000.0   # Far Clip Plane
    step_relaxation: float = 0.9

    def __post_init__(self):
        if self.epsilon <= 0.0:
            raise ValueError("Epsilon must be positive and non-zero.")
        if self.max_steps <= 0:
            raise ValueError("Max steps must be a positive integer.")
        if self.max_distance <= 0.0:
            raise ValueError("Max distance must be positive and non-zero.")
        if not (0.0 < self.step_relaxation <= 1.0):
            raise ValueError("Step relaxation must be in the range (0.0, 1.0].")    
    
    always_rebuild_bvh: bool = False # If true, forces BVH to rebuild on next use

class IntersectionStrategy(ABC):
    def __init__(self, settings: Optional[IntersectionSettings] = None):
        self.settings = settings or IntersectionSettings()
    
    @abstractmethod
    def find_hit(
        self,
        scene: Scene,
        ray: TracingRay,
        stats: Optional["TracingStats"] = None,
    ) -> HitInfo:
        """
        Calculates the closest intersection between a ray and the scene geometry.
        
        :param scene: The scene containing all objects and lights to test against.
        :type scene: Scene
        :param ray: The ray being cast into the scene (Camera Ray, Reflection Ray, etc).
        :type ray: TracingRay
        :param stats: Optional tracker for performance metrics (e.g., intersection counts).
        :type stats: Optional["TracingStats"]
        :return: A HitInfo object detailing the closest intersection point, normal, and distance.
        :rtype: HitInfo
        """
        ...
    
    def _resolve_normal(self, world_point: np.ndarray, world_transform: Transform, local_shape: SignedDistanceShape) -> np.ndarray:
        """
        Calculates the World Normal by transforming the point to local space,
        getting the local normal, and transforming it back.
        
        :param world_point: The point of intersection in World Space coordinates.
        :type world_point: np.ndarray
        :param world_transform: The transform component of the object being hit.
        :type world_transform: Transform
        :param local_shape: The geometric definition (SDF) of the object.
        :type local_shape: SignedDistanceShape
        :return: The normalized surface normal vector in World Space.
        :rtype: np.ndarray
        """
        # 1. World Point -> Local Point
        local_point = world_transform.inverse_transform_point(world_point)
        
        # 2. Local Point -> Local Normal
        local_normal = local_shape.get_normal(local_point)
        
        # 3. Local Normal -> World Normal (Inverse Transpose)
        world_normal = world_transform.transform_normal(local_normal)
        
        return unit(world_normal)

    def _intersect_sdf_object(
            self,
            obj: SceneNode,
            ray: TracingRay,
            stats: Optional["TracingStats"] = None
        ) -> HitInfo:
        """
        Shared logic for Ray vs SDF Object intersection using signed distance fields.
        
        :param obj: The specific object SceneNode to test intersection against.
        :type obj: SceneNode
        :param ray: The ray to march through the object's distance field.
        :type ray: TracingRay
        :param stats: Optional stats collector for profiling intersection costs.
        :type stats: Optional["TracingStats"]
        :return: HitInfo indicating if and where the ray intersected the object.
        :rtype: HitInfo
        """
        local_shape = getattr(obj.context, "shape", None)
        if local_shape is None:
            return HitInfo.miss()
        safe_shape = cast(SignedDistanceShape, local_shape)

        # 1. Transform Ray to Local Space
        # We assume world_transform is up to date
        safe_transform = cast(Transform, getattr(obj, 'world_transform', obj.transform))
        local_ray = safe_transform.inverse_transform_ray(ray)
        local_ray.orientation = unit(local_ray.orientation)

        # 2. Safety for Non-Uniform Scales
        # Convert world max distance to local space
        # We divide by the SMALLEST scale to ensure we cover the full world distance
        max_dist_local = self.settings.max_distance / min(*safe_transform.scale)

        t = 0.0
        sign_modifier = -1.0 if ray.is_inside else 1.0

        for _ in range(self.settings.max_steps):
            local_point = local_ray.point_at(t)

            if stats: stats.triangle_tests += 1

            # Get unscaled distance from SignedDistanceShape
            local_dist = safe_shape.get_distance(local_point) * sign_modifier

            # Hit Condition
            if local_dist < self.settings.epsilon:
                # A. Transform Local Point -> World Point
                world_point = safe_transform.transform_point(local_point)

                # B. Resolve Surface Normal
                surface_normal = self._resolve_normal(world_point, safe_transform, safe_shape)

                # C. True World Distance
                distance_world = np.linalg.norm(world_point - ray.origin)

                return HitInfo(
                    did_hit=True,
                    distance=float(distance_world),
                    point=world_point,
                    direction=ray.orientation,
                    normal=surface_normal,
                    obj=obj
                )

            # Step Forward
            t += local_dist * self.settings.step_relaxation

            # Boundary Check
            if t > max_dist_local:
                break
        
        return HitInfo.miss()
    
    def is_point_occluded(
            self,
            point_1: np.ndarray,
            point_2: np.ndarray,
            objects: List[SceneNode],
            bias: float = 1e-4,
            exclude_obj: Optional[SceneNode] = None,
            stats: Optional["TracingStats"] = None
        ) -> bool:
        """
        Return True if there's any object from `objects` except `exclude_object` in the vector pointing from `point_1` to `point_2`.
        
        :param point_1: The starting point of the shadow ray (usually the surface hit point).
        :type point_1: np.ndarray
        :param point_2: The target point (usually the light source position).
        :type point_2: np.ndarray
        :param objects: A list of SceneNodes in the scene to check for occlusion.
        :type objects: List[SceneNode]
        :param bias: A small offset applied to the origin to avoid surface acne.
        :type bias: float
        :param exclude_obj: The object the ray originated from (to prevent self-shadowing).
        :type exclude_obj: Optional[SceneNode]
        :return: True if an object blocks the path between point_1 and point_2, False otherwise.
        :param stats: Optional stats collector for profiling intersection costs.
        :type stats: Optional["TracingStats"]
        :rtype: bool
        """
        direction = np.array(point_2, dtype=float) - np.array(point_1, dtype=float)
        distance = np.linalg.norm(direction)
        
        if distance <= 0.0:
            return False
            
        unit_direction = direction / distance
        new_origin = np.array(point_1, dtype=float) + unit_direction * bias
        
        # Create a Shadow Ray
        shadow_ray = TracingRay(origin=new_origin, orientation=unit_direction)

        if stats is not None:
            stats.rays_shadow += 1
        
        # Check every object in the provided list
        # Note: In a BVH strategy, you would typically traverse the tree here rather than iterating a list.
        # However, since this signature accepts a specific list of objects, we test them directly.
        for obj in objects:
            if obj is exclude_obj:
                continue
                
            # Use the shared intersection logic
            hit = self._intersect_sdf_object(obj, shadow_ray)
            
            # If we hit something, and that hit is CLOSER than the light (point_2)
            if hit.hit and hit.distance < (distance - self.settings.epsilon):
                return True

        return False

class RayMarchingIntersection(IntersectionStrategy):
    """
    Find hits between rays and objects using a simple ray marching method involving distance estimation.
    """
    def find_hit(
            self,
            scene: Scene,
            ray: TracingRay,
            stats: Optional["TracingStats"] = None
        ) -> HitInfo:
        distance_world = 0.0

        for _ in range(self.settings.max_steps):
            if stats is not None:
                stats.aabb_tests += 1

            world_point = ray.point_at(distance_world)

            safe_objects = scene._cache_objects or scene.objects
            closest_object, distance_to_closest = self._distance_estimator(safe_objects, world_point, ray=ray)

            # Optimization: If we marched into the void
            if closest_object is None:
                break
            
            # Hit Check
            if distance_to_closest <= self.settings.epsilon:
                surface_normal = np.array([0.0, 0.0, 1.0])
                
                if closest_object is not None:
                    SignedDistanceShape = getattr(closest_object, "SignedDistanceShape", None)

                    if SignedDistanceShape is not None:
                        safe_shape = cast(SignedDistanceShape, SignedDistanceShape)
                        safe_transform = getattr(closest_object, 'world_transform', closest_object.transform)

                        # 3. Rotate normal back to world space
                        surface_normal = self._resolve_normal(world_point, safe_transform, safe_shape)

                if stats is not None:
                    stats.triangle_tests += 1

                return HitInfo(
                    did_hit=True,
                    distance=float(distance_world),
                    point=world_point,
                    direction=ray.orientation,
                    normal=surface_normal,
                    obj=closest_object,
                )
            
            # Advance
            distance_world += distance_to_closest * self.settings.step_relaxation
            
            # Frustum/Far Plane checks
            if scene.camera:
                obj_pos = getattr(closest_object, 'world_transform', Transform.Identity()).position
                far_plane_dist = np.linalg.norm(scene.camera.transform.position - obj_pos)
                if distance_world >= self.settings.max_distance or far_plane_dist >= scene.camera.far:
                    break

        if stats is not None:
            stats.rays_missed += 1

        return HitInfo.miss()
    
    def _distance_estimator(self, objects: List[SceneNode], point: np.ndarray, ray: Optional[TracingRay] = None, exclude_obj: Optional[SceneNode] = None) -> Tuple[Optional[SceneNode], float]:
        """
        Evaluates the Scene SDF to find the closest object and the distance to it.
        This relies on obj.context.shape.signed_distance() correctly handling Local->World conversion.
        
        :param objects: List of SceneNode objects in the scene.
        :type objects: List[SceneNode]
        :param point: The world-space point to evaluate distance from.
        :type point: np.ndarray
        :param ray: Optional ray for handling internal marching (is_inside flag).
        :type ray: Optional[TracingRay]
        :param exclude_obj: Object to exclude from distance calculations.
        :type exclude_obj: Optional[SceneNode]
        :return: Tuple of (closest_object, distance_to_closest)
        :rtype: Tuple[SceneNode | None, float]
        """
        min_dist = float("inf")
        closest_object = None
        sign_modifier = -1.0 if (ray and ray.is_inside) else 1.0

        for obj in objects:
            # 1. Skip exclusion (Self-Shadowing fix)
            if exclude_obj is not None and obj is exclude_obj:
                continue

            # Skip lights
            if isinstance(obj.context, LightContext):
                continue
            
            # Skip meshes for now
            if isinstance(obj.context, MeshContext):
                continue

            # 2. Check for SignedDistanceShape
            shape = getattr(obj.context, "shape", None)
            if shape is None:
                continue
            safe_shape = cast(SignedDistanceShape, shape)

            # 3. Calculate Distance
            # Use the WORLD transform so hierarchical/parented objects are handled properly
            safe_transform = getattr(obj, 'world_transform', obj.transform)
            local_point = safe_transform.inverse_transform_point(point)

            try:
                local_dist = float(safe_shape.get_distance(local_point))
                
                # Apply sign modifier for internal marching
                world_dist = local_dist * min(*safe_transform.scale) * sign_modifier
            except Exception:
                continue

            if world_dist < min_dist:
                min_dist = world_dist
                closest_object = obj
        
        return (closest_object, float(min_dist))

class InverseSDFIntersection(IntersectionStrategy):
    """
    Find hits between rays and objects using the Inverse SDF method.
    the SDF defines a mathematical function that erturns a vector/point based on a distance.
    Inverting the SDF allows for the point to calculate the distance
    """
    def __init__(self, settings: Optional[IntersectionSettings] = None, use_bounding_box: bool = True):
        super().__init__(settings)
        self.use_bounding_box = use_bounding_box

    def find_hit(
            self,
            scene: Scene,
            ray: TracingRay,
            stats: Optional["TracingStats"] = None
        ) -> HitInfo:
        closest_hit = HitInfo.miss()
        
        for obj in scene.get_objects_flat():
            # Optional: AABB Culling
            if self.use_bounding_box:
                box = obj.get_bounds()
                t_box = box.intersect(ray, self.settings.max_distance)
                if stats: stats.aabb_tests += 1
                if t_box == float('inf'):
                    continue

            # Intersect
            hit = self._intersect_sdf_object(obj, ray, stats)

            if hit.hit:
                if not closest_hit.hit or hit.distance < closest_hit.distance:
                    closest_hit = hit

        if not closest_hit.hit and stats:
            stats.rays_missed += 1

        return closest_hit

class BVHIntersection(IntersectionStrategy):
    """
    Find hits between rays and objects using a BVH data structure.
    """
    def __init__(self, settings: Optional[IntersectionSettings] = None, max_depth: int = 512):
        super().__init__(settings)
        self.max_depth = max_depth
        self._cached_bvh_root: Optional[BVHNode] = None
        self._cached_scene_version: Optional[int] = None

    def find_hit(self, scene: Scene, ray: TracingRay, stats: Optional["TracingStats"] = None) -> HitInfo:
        if (self._cached_bvh_root is None or scene.version != self._cached_scene_version) or self.settings.always_rebuild_bvh:
            print(f" < Building Hierarchy for scene objects...")
            
            # Update world matrices for all root objects
            for obj in scene.objects:
                obj.update_matrices()
                scene.update_version() # Ensure scene version increments if transforms changed
            
            # Get all objects including children
            all_objects = scene.get_objects_flat()
            
            self._cached_bvh_root = build_bvh_tree(all_objects)
            self._cached_scene_version = scene.version
            print(f" < Hierarchy build Complete for {len(all_objects)} objects.")

        closest_hit = HitInfo.miss()
        stack = [(self._cached_bvh_root, 0.0)]
        
        while stack:
            node, t_enter = stack.pop()
            
            # Optimization: If we already found a hit closer than this box, skip it
            current_max = closest_hit.distance if closest_hit.hit else float('inf')
            if t_enter >= current_max:
                continue

            # LEAF: Test Objects
            if node.objects:
                for obj in node.objects:
                    hit = self._intersect_sdf_object(obj, ray, stats)
                    if hit.hit:
                        if not closest_hit.hit or hit.distance < closest_hit.distance:
                            closest_hit = hit
                continue
            
            # INTERNAL: Use Helper
            self._push_valid_children(stack, node, ray, min(current_max, self.settings.max_distance))
                
        return closest_hit
    
    def is_point_occluded(
        self, 
        point_1: np.ndarray, 
        point_2: np.ndarray, 
        objects: List[SceneNode], 
        bias: float = 1e-4,
        exclude_obj: Optional[SceneNode] = None,
        stats: Optional["TracingStats"] = None
    ) -> bool:
        """
        BVH-Optimized Shadow Ray.
        Returns True immediately upon finding ANY valid intersection.
        """
        # 1. Setup Shadow Ray
        direction = np.array(point_2, dtype=float) - np.array(point_1, dtype=float)
        distance = float(np.linalg.norm(direction))
        
        if distance <= 1e-6:
            return False
        unit_direction = direction / distance

        new_origin = np.array(point_1, dtype=float) + unit_direction * bias

        shadow_ray = TracingRay(origin=new_origin, orientation=unit_direction)
        if stats is not None:
            stats.rays_shadow += 1

        # 2. Fallback if BVH is not built (Safety check)
        if self._cached_bvh_root is None:
            # Fallback to the slow O(N) loop from the base class
            return super().is_point_occluded(point_1, point_2, objects, bias, exclude_obj, stats)

        # 3. "Any Hit" Traversal
        # We don't need to sort children strictly because any valid hit is sufficient.
        # However, checking the closest box first often yields an exit sooner.
        stack = [(self._cached_bvh_root, 0.0)]

        while stack:
            node, t_enter = stack.pop()

            # Optimization: If the box is further away than the light, 
            # nothing inside it can block the light.
            if t_enter >= distance:
                continue

            # LEAF: Check exact object intersections
            if node.objects:
                for obj in node.objects:
                    if obj is exclude_obj:
                        continue
                    
                    # Use shared SDF intersection logic
                    hit = self._intersect_sdf_object(obj, shadow_ray)
                    
                    # EARLY EXIT: We found a blocker!
                    # We don't care if it's the closest one, just that it exists.
                    if hit.hit and hit.distance < (distance - self.settings.epsilon):
                        return True
                continue

            # INTERNAL: Check children
            self._push_valid_children(stack, node, shadow_ray, distance)

        # If stack empties without returning True, the path is clear.
        return False
    
    def _push_valid_children(
        self, 
        stack: List[Tuple[BVHNode, float]], 
        node: BVHNode, 
        ray: TracingRay, 
        limit_dist: float
    ):
        """
        Checks children AABBs and pushes them to the stack.
        Ensures the CLOSER child is popped first (by pushing the FURTHEST first).
        """
        # Check intersections with child boxes (returns float('inf') if miss or no child)
        t_left = node.left.box.intersect(ray, limit_dist) if node.left and node.left.box else float('inf')
        t_right = node.right.box.intersect(ray, limit_dist) if node.right and node.right.box else float('inf')

        # Optimization: Push the FURTHEST valid node first, so we pop the CLOSEST node first.
        # This maximizes the chance of finding a closer hit early and shrinking the limit_dist.
        if t_left < t_right:
            if t_right < limit_dist: 
                stack.append((node.right, t_right))
            if t_left < limit_dist:  
                stack.append((node.left, t_left))
        else:
            if t_left < limit_dist:  
                stack.append((node.left, t_left))
            if t_right < limit_dist: 
                stack.append((node.right, t_right))

class AnalyticalIntersection(IntersectionStrategy):
    """
    Finds hits using exact analytical equations (e.g. Ray-Sphere, Ray-Box).
    Supports object hierarchies.
    """
    def find_hit(self, scene: Scene, ray: TracingRay, stats: Optional["TracingStats"] = None) -> HitInfo:
        closest_object = None
        closest_point = None
        min_dist = float("inf")
        
        safe_objects = scene._cache_objects or scene.objects
        sign_modifier = -1.0 if ray.is_inside else 1.0

        for obj in safe_objects:
            local_shape = getattr(obj.context, "shape", None)
            if local_shape is None:
                return HitInfo.miss()
            safe_shape = cast(SignedDistanceShape, local_shape)
            
            # --- 1. Transform Ray to Local Space ---
            # We use the cached 'world_transform' if available, otherwise calculate it
            transform = getattr(obj, 'world_transform', obj.transform)
            local_ray = transform.inverse_transform_ray(ray)

            # --- 2. Get Intersections (Analytical) ---
            # SignedDistanceShape returns local points (e.g., (0,0,1) for a unit sphere)
            local_hits = safe_shape.ray_intersect(local_ray, self.settings.max_distance * sign_modifier)
            
            if not local_hits:
                continue
                
            if stats: stats.triangle_tests += 1

            # --- 3. Transform Hits to World Space & Check Distance ---
            # We transform points back to world space to compare distances correctly
            for local_p in local_hits:
                world_p = transform.transform_point(local_p)
                dist = np.linalg.norm(world_p - ray.origin)
                
                # Respect is_inside flag: for internal rays, invert sign of distance
                signed_dist = dist * sign_modifier
                
                if self.settings.epsilon < signed_dist < min_dist:
                    min_dist = signed_dist
                    closest_object = obj
                    closest_point = world_p

        # --- 4. Resolve Result ---
        if closest_object is None or closest_point is None:
            if stats: stats.rays_missed += 1
            return HitInfo.miss()

        # Calculate Normal for the closest hit
        local_shape = getattr(closest_object.context, "shape", None)
        if local_shape is None:
            return HitInfo.miss()
        safe_shape = cast(SignedDistanceShape, local_shape)
        safe_transform = getattr(closest_object, 'world_transform', closest_object.transform)

        surface_normal = self._resolve_normal(closest_point, safe_transform, safe_shape)

        # Convert back to unsigned distance for hit result
        unsigned_distance = float(abs(min_dist))

        return HitInfo(
            did_hit=True,
            point=closest_point,
            direction=ray.orientation,
            normal=surface_normal,
            distance=unsigned_distance,
            obj=closest_object
        )