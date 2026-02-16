from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Tuple, cast
from dataclasses import dataclass

from src.Data.Transform import Transform
from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from src.Data.Scene import SceneNode
from src.Data.Context import Mesh_Material
from src.Geometry.BVH import BVHNode, build_bvh_tree
from src.Geometry.AABB import AABB
from src.Geometry.SDF import SignedDistanceShape
from src.Lighting.Core import Light
from src.Utilities.Common import unit
from src.Data.Scene import Scene

if TYPE_CHECKING:
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
    
    use_aabb_bounding_box: bool = True
    bounding_box_bias = 1e-7
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
        
        :param scene: The scene containing all nodes and lights to test against.
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
        local_point = world_transform.local_transform_point(world_point)
        
        # 2. Local Point -> Local Normal
        local_normal = local_shape.get_normal(local_point)
        
        # 3. Local Normal -> World Normal (Inverse Transpose)
        world_normal = world_transform.world_transform_normal(local_normal)
        
        return unit(world_normal)

    def _intersect_sdf_object(
            self,
            node: SceneNode,
            ray: TracingRay,
            stats: Optional["TracingStats"] = None
        ) -> HitInfo:
        """
        Shared logic for Ray vs SDF Object intersection using signed distance fields.
        
        :param node: The specific object SceneNode to test intersection against.
        :type node: SceneNode
        :param ray: The ray to march through the object's distance field.
        :type ray: TracingRay
        :param stats: Optional stats collector for profiling intersection costs.
        :type stats: Optional["TracingStats"]
        :return: HitInfo indicating if and where the ray intersected the object.
        :rtype: HitInfo
        """
        local_shape = getattr(node.context, "shape", None)
        if local_shape is None:
            return HitInfo.miss()
        safe_shape = cast(SignedDistanceShape, local_shape)

        # 1. Transform Ray to Local Space
        # We assume world_transform is up to date
        local_ray = node.world_transform.local_transform_ray(ray)
        local_ray.orientation = unit(local_ray.orientation)

        # 2. Safety for Non-Uniform Scales
        # Convert world max distance to local space
        # We divide by the SMALLEST scale to ensure we cover the full world distance
        max_dist_local = self.settings.max_distance / min(*node.world_transform.scale)

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
                world_point = node.world_transform.world_transform_point(local_point)

                # B. Resolve Surface Normal
                surface_normal = self._resolve_normal(world_point, node.world_transform, safe_shape)

                # C. True World Distance
                distance_world = np.linalg.norm(world_point - ray.origin)

                return HitInfo(
                    did_hit=True,
                    distance=float(distance_world),
                    point=world_point,
                    direction=ray.orientation,
                    normal=surface_normal,
                    obj=node
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
            nodes: List[SceneNode],
            bias: float = 1e-4,
            exclude_obj: Optional[SceneNode] = None,
            stats: Optional["TracingStats"] = None
        ) -> bool:
        """
        Return True if there's any object from `nodes` except `exclude_object` in the vector pointing from `point_1` to `point_2`.
        
        :param point_1: The starting point of the shadow ray (usually the surface hit point).
        :type point_1: np.ndarray
        :param point_2: The target point (usually the light source position).
        :type point_2: np.ndarray
        :param nodes: A list of SceneNodes in the scene to check for occlusion.
        :type nodes: List[SceneNode]
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
        # However, since this signature accepts a specific list of nodes, we test them directly.
        for node in nodes:
            if node is exclude_obj:
                continue
                
            # Use the shared intersection logic
            hit = self._intersect_sdf_object(node, shadow_ray)
            
            # If we hit something, and that hit is CLOSER than the light (point_2)
            if hit.hit and hit.distance < (distance - self.settings.epsilon):
                return True

        return False

class RayMarchingIntersection(IntersectionStrategy):
    """
    Find hits between rays and nodes using a simple ray marching method involving distance estimation.
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

            safe_objects = scene._cache_nodes or scene.nodes
            closest_object, distance_to_closest = self._distance_estimator(safe_objects, world_point, ray=ray)

            # Optimization: If we marched into the void
            if closest_object is None:
                break
            
            # Hit Check
            if distance_to_closest <= self.settings.epsilon:
                surface_normal = np.array([0.0, 0.0, 1.0])
                
                if closest_object is not None:
                    shape = getattr(closest_object.context, "shape", None)

                    if shape is not None:
                        safe_shape = cast(SignedDistanceShape, shape)
                        surface_normal = self._resolve_normal(world_point, closest_object.world_transform, safe_shape)

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

        if stats is not None:
            stats.rays_missed += 1

        return HitInfo.miss()
    
    def _distance_estimator(
            self,
            nodes: List[SceneNode],
            point: np.ndarray,
            ray: Optional[TracingRay] = None,
            exclude_obj: Optional[SceneNode] = None,
            stats: Optional["TracingStats"] = None
        ) -> Tuple[Optional[SceneNode], float]:
        """
        Evaluates the Scene SDF to find the closest object and the distance to it.
        This relies on node.context.shape.signed_distance() correctly handling Local->World conversion.
        
        :param nodes: List of SceneNode nodes in the scene.
        :type nodes: List[SceneNode]
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

        for node in nodes:
            # 1. Skip exclusion (Self-Shadowing fix)
            if exclude_obj is not None and node is exclude_obj:
                continue

            if not node.active:
                continue
            
            # Skip meshes for now
            if isinstance(node.context, Mesh_Material):
                continue

            # 2. Check for SignedDistanceShape
            shape = getattr(node.context, "shape", None)
            if shape is None:
                continue
            safe_shape = cast(SignedDistanceShape, shape)
            
            if self.settings.use_aabb_bounding_box:
                bounds = node.get_global_bounds()
                t_box = AABB.from_bounds(bounds).intersect(ray, self.settings.max_distance, self.settings.bounding_box_bias)
                if stats: stats.aabb_tests += 1
                if t_box is None:
                    continue

            # 3. Calculate Distance
            # Use the WORLD transform so hierarchical/parented nodes are handled properly
            safe_transform = getattr(node, 'world_transform', node.transform)
            local_point = safe_transform.local_transform_point(point)

            try:
                local_dist = float(safe_shape.get_distance(local_point))
                
                # Apply sign modifier for internal marching
                world_dist = local_dist * min(*safe_transform.scale) * sign_modifier
            except Exception:
                continue

            if world_dist < min_dist:
                min_dist = world_dist
                closest_object = node
        
        return (closest_object, float(min_dist))

class InverseSDFIntersection(IntersectionStrategy):
    """
    Find hits between rays and nodes using the Inverse SDF method.
    the SDF defines a mathematical function that erturns a vector/point based on a distance.
    Inverting the SDF allows for the point to calculate the distance
    """
    def find_hit(
            self,
            scene: Scene,
            ray: TracingRay,
            stats: Optional["TracingStats"] = None
        ) -> HitInfo:
        closest_hit = HitInfo.miss()
        
        for node in scene.cache_scene_nodes_flat():
            if not node.active:
                continue

            # Optional: AABB Culling
            if self.settings.use_aabb_bounding_box:
                bounds = node.get_global_bounds()
                t_box = AABB.from_bounds(bounds).intersect(ray, self.settings.max_distance, self.settings.bounding_box_bias)
                if stats: stats.aabb_tests += 1
                if t_box is None:
                    continue

            # Intersect
            hit = self._intersect_sdf_object(node, ray, stats)

            if hit.hit:
                if not closest_hit.hit or hit.distance < closest_hit.distance:
                    closest_hit = hit


        if not closest_hit.hit and stats:
            stats.rays_missed += 1

        return closest_hit

class BVHIntersection(IntersectionStrategy):
    """
    Find hits between rays and nodes using a BVH data structure.
    """
    def __init__(self, settings: Optional[IntersectionSettings] = None, max_depth: int = 512):
        super().__init__(settings)
        self.max_depth = max_depth
        self._cached_bvh_root: Optional[BVHNode] = None
        self._cached_scene_version: Optional[int] = None

    def find_hit(self, scene: Scene, ray: TracingRay, stats: Optional["TracingStats"] = None) -> HitInfo:
        """Build BVH if needed, then traverse with optimized recursion."""
        if self._cached_bvh_root is None or scene.version != self._cached_scene_version or self.settings.always_rebuild_bvh:
            print(f" < Building Hierarchy for scene nodes...")
            
            for node in scene.cache_scene_nodes_flat():
                node.update_matrices()
            
            all_objects = scene.cache_scene_nodes_flat()
            self._cached_bvh_root = build_bvh_tree(all_objects)
            self._cached_scene_version = scene.version
            print(f" < Hierarchy build Complete for {len(all_objects)} nodes.")

        if self._cached_bvh_root is None:
            return HitInfo.miss()

        self._closest_hit = HitInfo.miss()  # Track best hit across recursion
        self._scene = scene  # Cache for access in recurse function
        self._ray = ray
        self._stats = stats

        def recurse(b_node: BVHNode):
            """Recursively traverse BVH with ordered child visiting."""
            if b_node is None or b_node.box is None:
                return

            # Prune if box is beyond current best hit
            current_max = self._closest_hit.distance if self._closest_hit.hit else float('inf')
            t_enter = b_node.box.intersect(self._ray, current_max, self.settings.bounding_box_bias)
            
            if t_enter == None or (self._closest_hit.hit and t_enter >= self._closest_hit.distance):
                return

            if self._stats:
                self._stats.aabb_tests += 1

            # LEAF: Test all nodes
            if b_node.objects:
                for node in b_node.objects:
                    if self.settings.use_aabb_bounding_box:
                        box = node.get_transformed_aabb()
                        if box is None:
                            continue
                        t_box = box.intersect(self._ray, self.settings.max_distance, self.settings.bounding_box_bias)
                        if self._stats:
                            self._stats.aabb_tests += 1
                        if t_box is None:
                            continue

                    hit = self._intersect_sdf_object(node, self._ray, self._stats)
                    if hit.hit:
                        # Respect far plane
                        if self._scene.camera:
                            obj_pos = hit.obj.world_transform.position
                            far_plane_dist = np.linalg.norm(self._scene.camera.transform.position - obj_pos)
                            if far_plane_dist >= self._scene.camera.far:
                                continue

                        if not self._closest_hit.hit or hit.distance < self._closest_hit.distance:
                            self._closest_hit = hit
                return

            self._traverse_ordered_children(recurse, b_node)

        recurse(self._cached_bvh_root)
        return self._closest_hit

    def _traverse_ordered_children(self, recurse_func, node: BVHNode):
        """Visit children in order: closer child first."""
        left_dist = self._get_box_distance(node.left)
        right_dist = self._get_box_distance(node.right)

        if left_dist < right_dist:
            recurse_func(node.left)
            recurse_func(node.right)
        else:
            recurse_func(node.right)
            recurse_func(node.left)

    def _get_box_distance(self, node: Optional[BVHNode]) -> float:
        """Calculate distance from ray origin to box center."""
        if node is None or node.box is None:
            return float('inf')
        center = node.box.center
        return float(np.linalg.norm(center - self._ray.origin))
    
    def is_point_occluded(
        self, 
        point_1: np.ndarray, 
        point_2: np.ndarray, 
        nodes: List[SceneNode], 
        bias: float = 1e-6,
        exclude_obj: Optional[SceneNode] = None,
        stats: Optional["TracingStats"] = None
    ) -> bool:
        """BVH shadow ray with early termination."""
        direction = np.array(point_2, dtype=float) - np.array(point_1, dtype=float)
        distance = float(np.linalg.norm(direction))
        
        if distance <= 1e-6:
            return False
        
        unit_direction = direction / distance
        new_origin = np.array(point_1, dtype=float) + unit_direction * bias
        shadow_ray = TracingRay(origin=new_origin, orientation=unit_direction)
        
        if stats is not None:
            stats.rays_shadow += 1

        if self._cached_bvh_root is None:
            return super().is_point_occluded(point_1, point_2, nodes, bias, exclude_obj, stats)

        # Ordered recursion for shadow rays ("any hit" = early exit)
        def recurse(b_node: BVHNode) -> bool:
            """Returns True immediately if ANY valid blocker found."""
            if b_node is None or b_node.box is None:
                return False

            t_enter = b_node.box.intersect(shadow_ray, distance, 1e-7)
            if t_enter is None or t_enter >= distance:
                return False

            if stats:
                stats.aabb_tests += 1

            # LEAF: Check for blockers
            if b_node.objects:
                for node in b_node.objects:
                    if node is exclude_obj or not node.active:
                        continue

                    hit = self._intersect_sdf_object(node, shadow_ray)
                    
                    # EARLY EXIT: Found blocker!
                    if hit.hit and hit.distance < (distance - self.settings.epsilon):
                        return True
                return False

            # INTERNAL: Check closer child first (better early exit)
            left_dist = np.linalg.norm(b_node.left.box.center - shadow_ray.origin) if b_node.left else float('inf')
            right_dist = np.linalg.norm(b_node.right.box.center - shadow_ray.origin) if b_node.right else float('inf')

            if left_dist < right_dist:
                return recurse(b_node.left) or recurse(b_node.right)
            else:
                return recurse(b_node.right) or recurse(b_node.left)

        return recurse(self._cached_bvh_root)
    
    def _push_valid_children(
        self, 
        stack: List[Tuple[BVHNode, float]], 
        b_node: BVHNode, 
        ray: TracingRay, 
        limit_dist: float,
        bias: float = 1e-6 
    ):
        """
        Checks children AABBs and pushes them to the stack.
        Ensures the CLOSER child is popped first (by pushing the FURTHEST first).
        """
        # Check intersections with child boxes (returns float('inf') if miss or no child)
        t_left = b_node.left.box.intersect(ray, limit_dist, bias) if b_node.left and b_node.left.box else float('inf')
        t_right = b_node.right.box.intersect(ray, limit_dist, bias) if b_node.right and b_node.right.box else float('inf')

        # Optimization: Push the FURTHEST valid node first, so we pop the CLOSEST node first.
        # This maximizes the chance of finding a closer hit early and shrinking the limit_dist.
        if t_left < t_right:
            if t_right < limit_dist: 
                stack.append((b_node.right, t_right))
            if t_left < limit_dist:  
                stack.append((b_node.left, t_left))
        else:
            if t_left < limit_dist:  
                stack.append((b_node.left, t_left))
            if t_right < limit_dist: 
                stack.append((b_node.right, t_right))

class AnalyticalIntersection(IntersectionStrategy):
    """
    Finds hits using exact analytical equations (e.g. Ray-Sphere, Ray-Box).
    Supports object hierarchies.
    """
    def find_hit(self, scene: Scene, ray: TracingRay, stats: Optional["TracingStats"] = None) -> HitInfo:
        closest_object = None
        closest_point = None
        min_dist = float("inf")
        
        safe_objects = scene.cache_scene_nodes_flat()
        sign_modifier = -1.0 if ray.is_inside else 1.0

        for node in safe_objects:
            local_shape = getattr(node.context, "shape", None)
            if local_shape is None:
                continue
            safe_shape = cast(SignedDistanceShape, local_shape)
            
            # --- 1. Transform Ray to Local Space ---
            # We use the cached 'world_transform' if available, otherwise calculate it
            transform = getattr(node, 'world_transform', node.transform)
            local_ray = transform.local_transform_ray(ray)

            # --- 2. Get Intersections (Analytical) ---
            # SignedDistanceShape returns local points (e.g., (0,0,1) for a unit sphere)
            local_hits = safe_shape.ray_intersect(local_ray, self.settings.max_distance * sign_modifier)
            
            if not local_hits:
                continue
                
            if stats: stats.triangle_tests += 1

            # --- 3. Transform Hits to World Space & Check Distance ---
            # We transform points back to world space to compare distances correctly
            for local_p in local_hits:
                world_p = transform.world_transform_point(local_p)
                dist = np.linalg.norm(world_p - ray.origin)
                
                # Respect is_inside flag: for internal rays, invert sign of distance
                signed_dist = dist * sign_modifier
                
                if self.settings.epsilon < signed_dist < min_dist:
                    min_dist = signed_dist
                    closest_object = node
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