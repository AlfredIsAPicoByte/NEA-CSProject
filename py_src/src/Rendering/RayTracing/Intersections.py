from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, cast

from src.Data.Transform import Transform
from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from src.Data.AABB import AABB
from ..Core import TracingStats
from src.Geometry.Core import Shape
from src.Geometry.BVH import BVHNode, BVHSplitMode
from src.Geometry.Primitive import Primitive
from src.Utilities.Common import unit
from src.Data.Scene import Scene

class IntersectionStrategy(ABC):
    def __init__(
            self,
            epsilon: float = 1e-4,          # Hit Threshold
            max_steps: int = 128,           # Performance Cap
            max_distance: float = 1000.0,   # Far Clip Plane
            step_relaxation: float = 0.9
        ):
        self.epsilon = epsilon
        self.max_distance = max_distance
        self.max_steps = max_steps
        self.step_relaxation = step_relaxation
    
    @abstractmethod
    def find_hit(
        self,
        scene: Scene,
        ray: TracingRay,
        stats: Optional["TracingStats"] = None,
    ) -> HitInfo:
        ...

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
        distance_traveled = 0.0

        for _ in range(self.max_steps):
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
                
                if closest_object is not None:
                    shape = getattr(closest_object, "shape", None)
                    transform = closest_object.transform

                    if shape is not None:
                        shape = cast(Shape, shape)
                        
                        # 1. Convert hit point to local space
                        local_hit_point = transform.inverse_transform_point(point)
                        
                        # 2. Calculate normal in local space (e.g., via finite difference)
                        local_normal = shape.get_normal(local_hit_point)
                        
                        # 3. Rotate normal back to world space
                        world_normal = transform.transform_normal(local_normal)
                        
                        # Normalize to be safe
                        surface_normal = unit(world_normal)

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
            distance_traveled += distance_to_closest * self.step_relaxation
            
            # Frustum/Far Plane checks
            if scene.camera:
                obj_pos = getattr(closest_object, 'world_transform', Transform.identity()).position
                far_plane_dist = np.linalg.norm(scene.camera.transform.position - obj_pos)
                if distance_traveled >= self.max_distance or far_plane_dist >= scene.camera.far:
                    break

        if stats is not None:
            stats.missed_rays += 1

        return HitInfo.miss()

class InverseSDFIntersection(IntersectionStrategy):
    """
    Find hits between rays and objects using the Inverse SDF method.
    the SDF defines a mathematical function that erturns a vector/point based on a distance.
    Inverting the SDF allows for the point to calculate the distance
    """
    def __init__(
            self,
            epsilon: float = 1e-4,
            max_steps: int = 128,
            max_distance: float = 1000,
            step_relaxation: float = 0.9,   # Step Safety Factor
            use_bounding_box: bool = True   # Optimization Flag
        ):
        super().__init__(epsilon, max_steps, max_distance, step_relaxation)
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

            if self.use_bounding_box:
                bounds = getattr(obj, "bounds", None)
                if bounds is None:
                    continue
                
                bounds = cast(AABB, bounds)
                if bounds.intersect(ray) == float('inf'):
                    continue

            # Attempt to intersect this specific object
            hit_info = self._intersect_object(obj, ray, scene.camera.far, scene.camera.transform.position, stats)
            
            # Keep track of the closest hit only
            hit_obj = getattr(hit_info, "obj", None)
            if hit_obj is None:
                break
            
            if scene.camera is None:
                break

            far_plane_distance = np.linalg.norm(scene.camera.transform.position - hit_obj.world_transform.position)
            if hit_info.hit and (hit_info.distance < closest_hit.distance or hit_info.distance < far_plane_distance) :
                closest_hit = hit_info
                
        if not closest_hit.hit and stats is not None:
            stats.missed_rays += 1

        return closest_hit

    def _intersect_object(
            self,
            obj: Primitive,
            ray: TracingRay,
            far_plane: float,
            cam_pos: np.ndarray,
            stats: Optional["TracingStats"] = None
        ) -> "HitInfo":
        obj_shape = getattr(obj, "shape", None)
        if obj_shape is None:
            return HitInfo.miss()

        # Get object transform
        obj_transform = getattr(obj, 'world_transform', Transform.identity())
        
        # Transform the ray to local space for marching
        local_ray = obj_transform.inverse_transform_ray(ray)

        # Calculate the scale factor for converting world distances to local distances
        # This is needed because we're stepping in local space but max_distance is in world space
        scale = obj_transform.scale
        min_scale = min(abs(scale[0]), abs(scale[1]), abs(scale[2]))
        
        # Handle near-zero scales: clamp to prevent infinite marching distances
        # If min_scale is very small (< 1e-6), use a reasonable minimum instead of dividing
        safe_min_scale = max(min_scale, 1e-6)
        max_distance_local = min(self.max_distance / safe_min_scale, self.max_distance * 1e4)

        # Check for "Inside-Out" logic (for X-ray/Dielectrics)
        # If we are inside, we treat negative distance as empty space (flip sign)
        sign_modifier = -1.0 if ray.is_inside else 1.0
        
        # --- Raymarch Loop in Local Space ---
        t = 0.0
        
        for _ in range(self.max_steps):
            # 1. Calculate point in local space
            local_p = local_ray.point_at(t)

            # 2. Sample the Object's SDF in LOCAL space (unscaled)
            dist_local = obj_shape.signed_distance(local_p)

            if stats is not None:
                stats.triangle_tests += 1

            # Apply sign modifier for inside-out logic
            dist = dist_local * sign_modifier

            # 3. Hit Check (using pure local distance)
            if dist < self.epsilon:
                # Compute local normal using the local point
                local_normal = self._calc_local_gradient(obj_shape, local_p)
                if ray.is_inside:
                    local_normal = -local_normal

                # Transform normal to world space (inverse transpose handles non-uniform scale)
                world_normal = obj_transform.transform_normal(local_normal)
                surface_normal = unit(world_normal)

                # Compute accurate world-space hit point and distance
                p_world = obj_transform.transform_point(local_p)
                distance_world = np.linalg.norm(p_world - ray.origin)

                return HitInfo(
                    did_hit=True,
                    distance=float(distance_world),
                    point=p_world,
                    normal=surface_normal,
                    obj=obj
                )
            
            # 4. Step in local space (unscaled)
            t += (dist * self.step_relaxation)
            
            # Frustum/Far Plane checks (convert max_distance to local space)
            obj_pos = obj_transform.position
            far_plane_dist = np.linalg.norm(cam_pos - obj_pos)
            if t >= max_distance_local or far_plane_dist >= far_plane:
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
    """
    Find hits between rays and objects using a BVH data structure.
    """
    def __init__(
            self,
            epsilon: float = 1e-4,
            max_steps: int = 64,
            max_distance: float = 50,
            step_relaxation: float = 0.9,
            max_depth: int = 512,
            split_mode: BVHSplitMode = BVHSplitMode.LONGEST_AXIS
        ):
        super().__init__(epsilon, max_steps, max_distance, step_relaxation)

        self.max_depth = max_depth
        self.split_mode = split_mode
        self._cached_bvh_root: Optional[BVHNode] = None
        self._cached_scene_id: Optional[int] = None

    def find_hit(self, scene: Scene, ray: TracingRay, stats: Optional["TracingStats"] = None) -> HitInfo:
        # 1. Check if we need to build/rebuild the BVH
        # (We use id(scene.objects) as a cheap way to detect if the list changed)
        current_scene_id = id(scene.objects)
        if self._cached_bvh_root is None or current_scene_id != self._cached_scene_id:
            print(f"[BVH] Building Hierarchy for {len(scene.objects)} objects...")
            self._cached_bvh_root = self._build_bvh(scene.objects, 0)
            self._cached_scene_id = current_scene_id
            print(f"[BVH] Built Hiearchy for {len(scene.objects)} objects.")

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
                    hit = self._test_single_object(obj, ray, scene.camera.far, scene.camera.transform.position, stats)
                    
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
                if l_box is not None:
                    d_left = l_box.intersect(ray)
                
            if node.right:
                if stats: stats.aabb_tests += 1
                r_box = getattr(node.right, "box", None)
                if r_box is not None:
                    d_right = r_box.intersect(ray)

            l_node = getattr(node, "left", None)
            r_node = getattr(node, "right", None)

            # Push valid children to stack
            # Push the furthest one first, so the closest is at top of stack
            if d_left != float('inf') and d_right != float('inf'):
                if d_left < d_right:
                    stack.append((r_node, d_right))
                    stack.append((l_node, d_left))
                else:
                    stack.append((l_node, d_left))
                    stack.append((r_node, d_right))
            elif d_left != float('inf') and l_node is not None:
                stack.append((l_node, d_left))
            elif d_right != float('inf') and r_node is not None:
                stack.append((r_node, d_right))

        if not closest_hit.hit and stats:
            stats.missed_rays += 1
            
        return closest_hit

    def _build_bvh(self, objects: list[Primitive], build_depth: int) -> BVHNode:
        """
        Recursively splits the object list to build the tree.
        """
        # 0. Base Case
        if build_depth >= self.max_depth or not objects:
            leaf_node = BVHNode(objects)
            return leaf_node

        # Create Node
        node = BVHNode([])
        
        # 1. Calculate Bounds for all objects in this list
        # We cache AABBs for performance
        object_bounds = [(obj, AABB.from_transform_shape(getattr(obj, 'world_transform', obj.transform), obj.shape)) for obj in objects]
        
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
        
        print(node.objects)

        # 3. Split Strategy: Longest Axis
        if self.split_mode == BVHSplitMode.LONGEST_AXIS:
            # Find longest axis
            extent = node_max - node_min
            axis = np.argmax(extent) # 0=x, 1=y, 2=z

            # Sort objects by their center along the chosen axis
            object_bounds = [(obj, obj.get_aabb()) for obj in node.objects]
            sorted_objs = sorted(object_bounds, key=lambda item: (item[1].min_point[axis] + item[1].max_point[axis]) / 2)
            mid = len(sorted_objs) // 2

            # Create child nodes
            left_objs = [item[0] for item in sorted_objs[:mid]]
            right_objs = [item[0] for item in sorted_objs[mid:]]

            node.left = self._build_bvh(left_objs, build_depth + 1)
            node.right = self._build_bvh(right_objs, build_depth + 1)

        elif self.split_mode == BVHSplitMode.BALANCED:
            # Sort objects by their center along all axes (average)
            object_bounds = [(obj, obj.get_aabb()) for obj in node.objects]
            object_bounds.sort(key=lambda item: np.mean([item[1].min_point, item[1].max_point]))

            mid = len(object_bounds) // 2

            # Create child nodes
            left_objs = [item[0] for item in sorted_objs[:mid]]
            right_objs = [item[0] for item in sorted_objs[mid:]]

            node.left = self._build_bvh(left_objs, build_depth + 1)
            node.right = self._build_bvh(right_objs, build_depth + 1)

        return node

    def _test_single_object(
            self,
            obj: Primitive,
            ray: TracingRay,
            far_plane: float,
            cam_pos: np.ndarray,
            stats: Optional["TracingStats"]) -> HitInfo:
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
        obj_transform = cast(Transform, getattr(obj, 'world_transform', Transform.identity()))
        local_ray = obj_transform.inverse_transform_ray(ray)
        
        local_dir_len = np.linalg.norm(local_ray.orientation)
        if local_dir_len > 0:
            local_ray.orientation /= local_dir_len

        # 2. Get an minimum Scale Factor for safe stepping. Imagine the object is uniform in size
        scale = obj_transform.scale
        min_scale = min(abs(scale[0]), abs(scale[1]), abs(scale[2]))
        
        # Handle near-zero scales: clamp to prevent infinite marching distances
        # If min_scale is very small (< 1e-6), use a reasonable minimum instead of dividing
        safe_min_scale = max(min_scale, 1e-6)
        max_distance_local = min(self.max_distance / safe_min_scale, self.max_distance * 1e4)

        t = 0.0
        sign_modifier = -1.0 if getattr(ray, "is_inside", False) else 1.0

        for _ in range(self.max_steps): 
            p = local_ray.point_at(t)
            
            if stats: stats.triangle_tests += 1

            # Sample the unit SDF (unscaled)
            dist_local = obj_shape.unit_signed_distance(p) * sign_modifier
            
            # Check convergence (using pure local distance)
            if dist_local < self.epsilon:
                # --- Hit Found ---
                
                # A. Transform Point to World
                p_world = obj_transform.transform_point(p)
                
                # B. Calculate Normal (Local Gradient -> World Normal)
                local_normal = self._calc_local_gradient(obj_shape, p)
                if ray.is_inside: 
                    local_normal = -local_normal
                normal_world = obj_transform.transform_normal(local_normal)
                surface_normal = unit(normal_world)
                
                # C. True World Distance
                distance_world = np.linalg.norm(p_world - ray.origin)
                
                return HitInfo(
                    did_hit=True,
                    distance=float(distance_world),
                    point=p_world,
                    normal=surface_normal,
                    obj=obj
                )
            
            # Step in local space (unscaled)
            t += dist_local * self.step_relaxation
            
            # Frustum/Far Plane checks (convert max_distance to local space)
            obj_pos = getattr(obj, 'world_transform', Transform.identity()).position
            far_plane_dist = np.linalg.norm(cam_pos - obj_pos)
            if t >= max_distance_local or far_plane_dist >= far_plane:
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
        return grad / norm if norm > 0 else np.array([0.0, 1.0, 0.0])