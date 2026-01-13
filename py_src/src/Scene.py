from typing import Callable, List, Tuple, Optional, cast
import numpy as np
from math import atan2, asin, pi, floor

from CommonUtils import safe_norm
from PrimaryStructures import Ray, HitInfo, Transform
from Camera import VCamera
from Geometry import Shape, VObject
from Luminance import LightSource, Color

class Scene:
    def __init__(self, name: str = "Scene", camera: Optional[VCamera] = None, **kwargs):
        self.name = name
        self.objects: List[VObject] = []
        self.lights: List[LightSource] = []
        self.camera: VCamera = camera if camera is not None else VCamera(Transform.identity())

        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_camera(self, camera: VCamera):
        self.camera = camera
    
    def add_object(self, obj: VObject):
        self.objects.append(obj)

    def add_light(self, light: LightSource):
        self.lights.append(light)

    def get_lights(self):
        return list(self.lights)

    def distance_estimator(self, point: np.ndarray, exclude_obj: Optional[VObject] = None) -> Tuple[Optional[VObject], float]:
        """
        Evaluates the Scene SDF to find the closest object and the distance to it.
        This relies on obj.shape.signed_distance() correctly handling World->Local conversion.
        """
        min_d = float("inf")
        closest = None
        
        all_object_flattened: set[VObject] = get_all_objects_flattened(self.objects)

        for obj in all_object_flattened:
            # 1. Skip exclusion (Self-Shadowing fix)
            if exclude_obj is not None and obj is exclude_obj:
                continue
            
            # 2. Check for Shape
            shape = getattr(obj, "shape", None)
            if shape is None:
                continue
            shape = cast(Shape, shape)

            # 3. Calculate Distance
            # The Shape is responsible for transforming the world 'point' to its local space.
            transform = obj.world_transform
            local_point = transform.inverse_transform_point(point)

            try:
                # 3. Get Local Distance (SDF assumes object is at 0,0,0)
                d_local = float(shape.signed_distance(local_point))
                
                # 4. Scale Distance back to World Space
                # If the object is scaled down (0.1), a local distance of 1.0 is actually 0.1 in world space.
                # We multiply by the smallest scale component to avoid overstepping (overshooting).
                scale = transform.scale
                min_scale = min(abs(scale[0]), abs(scale[1]), abs(scale[2]))
                
                d_world = d_local * min_scale
            except Exception:
                continue

            if d_world < min_d:
                min_d = d_world
                closest = obj
        
        return (closest, float(min_d))
    
    def get_closest_intersection(self, ray: Ray) -> HitInfo:
        """
        Analytical Intersection (Ray-Sphere, Ray-Box).
        Returns a `HitInfo` describing the closest valid intersection.
        """
        closest_obj = None
        closest_point = None
        min_distance = float("inf")
        
        # Threshold to ignore self-intersections (Shadow Acne)
        epsilon = 1e-4

        # Iterate over all objects including children to ensure we test child shapes
        for obj in get_all_objects_flattened(self.objects):
            shape = getattr(obj, "shape", None)
            if shape is None:
                continue

            # Get all intersection points (World Space)
            # The Shape class handles the Ray transformation (World->Local) internally
            # and returns the points transformed back to World Space.
            transform = obj.world_transform

            # 1. Transform Ray -> Local Space
            # "Shoot the ray as if the camera moved relative to the object"
            local_ray = transform.inverse_transform_ray(ray)

            # 2. Get Intersections in Local Space
            # Shape returns points like (0, 0, 1) assuming it is centered at origin
            local_hits = shape.get_ray_intersections(local_ray)
            
            if not local_hits:
                continue

            # 3. Transform Hits Local -> World
            # Move the hit points to where the object actually is in the scene
            world_hits = [transform.transform_point(p) for p in local_hits]

            for hit_point in world_hits:
                dist = np.linalg.norm(hit_point - ray.origin)
                
                if epsilon < dist < min_distance:
                    min_distance = dist
                    closest_obj = obj
                    closest_point = hit_point

        if closest_obj is None or closest_point is None:
            return HitInfo.miss()

        # Resolve Normal
        normal = np.array([0.0, 1.0, 0.0])
        
        # Recalculate normal correctly using the object's transform logic
        shape = getattr(closest_obj, "shape", None)
        if shape is not None:
            # We need to manually do the normal transform pipeline:
            # World Point -> Local Point -> Local Gradient -> World Normal
            
            # Use the object's WORLD transform so the normal is transformed
            # consistently with how the hit point was computed (handles parents)
            transform = getattr(closest_obj, 'world_transform', Transform.identity())
            local_pt = transform.inverse_transform_point(closest_point)
            
            # Calculate Local Normal (Gradient)
            # Note: 'get_normal' usually expects a local point if it's a pure shape
            local_normal = shape.get_normal(local_pt) 
            
            # Transform Normal to World (Inverse Transpose logic handles non-uniform scale)
            normal = transform.transform_normal(local_normal)

        return HitInfo(
            did_hit=True, 
            point=closest_point, 
            direction=ray.orientation, 
            normal=normal, 
            distance=float(min_distance), 
            obj=closest_obj 
        )
    
    def clear_objects(self):
        self.objects.clear()
        self.lights.clear()
        self.camera = VCamera(Transform.identity())
    
    def is_occluded(self, point: np.ndarray, light_pos: np.ndarray, bias: float = 1e-4, epsilon: float = 1e-4, max_steps: int = 256, exclude_obj = None) -> bool:
        """
        Return True if there's an occluder between `point` and `light_pos`.
        Uses geometry intersection (get_closest_intersection) if available; otherwise
        falls back to sampling with distance_estimator (SDF ray-march).

        Args:
            exclude_obj: Optional object instance to ignore during occlusion (useful to avoid self-shadowing).
        """
        dir_vec = np.array(light_pos, dtype=float) - np.array(point, dtype=float)
        dist_to_light = np.linalg.norm(dir_vec)
        if dist_to_light <= 0.0:
            return False
        dir_norm = dir_vec / dist_to_light

        origin = np.array(point, dtype=float) + dir_norm * bias

        # --- 1) Try geometry intersection path if available ---
        try:
            from PrimaryStructures import Ray

            ray = Ray(origin, dir_norm)
            hit_info = self.get_closest_intersection(ray)
            v_object = getattr(hit_info, "obj", None)
            if hit_info.hit and v_object is not None and hit_info.point is not None:
                # Ignore hits on the excluded object to prevent self-shadowing
                if v_object is not exclude_obj:
                    hit_dist = np.linalg.norm(hit_info.point - origin)
                    if hit_dist < (dist_to_light - epsilon):
                        return True
        except Exception:
            # geometry intersection may not be supported; fall back to distance estimator below
            pass

        # --- 2) If SDF-based distance estimator exists, use ray-march ---
        if hasattr(self, "distance_estimator") and callable(self.distance_estimator):
            distance_traveled = 0.0
            for _ in range(max_steps):
                sample_point = origin + dir_norm * distance_traveled
                v_object, dist = self.distance_estimator(sample_point)

                if v_object is not None:
                    if dist <= epsilon:
                        # hit something before reaching the light -> occluded
                        return True
                    distance_traveled += dist
                    if distance_traveled >= dist_to_light:
                        # reached light without hitting anything
                        return False
            # exceeded max steps without reaching the light - treat as occluded
            return True

        # If no approach worked, conservatively say not occluded
        return False

def get_all_objects_flattened(root_objects: list[VObject]) -> set[VObject]:
    """
    Flattens a hierarchy of objects into a single list, 
    preventing infinite loops from circular references.
    """
    flat_list = []
    
    # We store the ID (memory address) of visited objects
    visited_ids = set() 
    
    # Initialize stack with the top-level objects
    stack = list(root_objects)

    while stack:
        current_obj = stack.pop()

        # 1. Cycle Detection: Have we seen this specific object instance before?
        if id(current_obj) in visited_ids:
            continue
            
        # 2. Mark as visited
        visited_ids.add(id(current_obj))
        
        # 3. Add to our result list
        flat_list.append(current_obj)

        # 4. Add children to the stack to be processed next
        # (Check if the object actually has children first)
        if hasattr(current_obj, 'children') and current_obj.children:
            stack.extend(current_obj.children)
            
    return set(flat_list)