from typing import Callable, List, Tuple, Optional
import numpy as np
from math import atan2, asin, pi, floor

from PrimaryStructures import Ray, HitInfo
from Camera import VCamera
from Geometry import VObject
from Luminance import LightSource, Color, ColorGradient

class Scene:
    def __init__(self, name: str = "Scene", camera: Optional[VCamera] = None, **kwargs):
        self.name = name
        self.objects: List[VObject] = []
        self.lights: List[LightSource] = []
        self.camera: Optional[VCamera] = camera

        # Ambient lighting defaults (can be overridden via kwargs)
        # small neutral ambient to avoid complete black shadows by default
        self.ambient_color: Color = Color(0.03, 0.03, 0.03)
        self.ambient_intensity: float = 0.1

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
        
        Args:
            point: The 3D point to check.
            exclude_obj: (Optional) An object to ignore. 
                         Crucial for preventing self-shadowing (shadow acne).
        """
        min_d = float("inf")
        closest = None
        
        for obj in self.objects:
            # 1. Skip the object we are starting from (Self-Shadowing fix)
            if exclude_obj is not None and obj is exclude_obj:
                continue
                
            d = float("inf")
            
            try:
                # 2. Check if object has a Signed Distance Function
                # Support several common method namings used across shapes
                sdf_fn = None
                if hasattr(obj.shape, "signed_distance") and callable(getattr(obj.shape, "signed_distance")):
                    sdf_fn: Callable[[np.ndarray], float] = getattr(obj.shape, "signed_distance")
                    d = float(sdf_fn(point))

            except Exception:
                # If math fails on one object, don't crash the whole renderer
                continue
            
            # 4. Update Closest
            if d < min_d:
                min_d = d
                closest = obj
        
        return (closest, min_d)
    
    def get_closest_intersection(self, ray: Ray) -> HitInfo:
        """
        Analytical Intersection (Ray-Sphere, Ray-Plane).
        Returns a `HitInfo` describing the closest intersection.
        """
        closest_obj = None
        closest_hit = None
        min_distance = float("inf")
        
        # CRITICAL: Threshold to ignore self-intersections.
        # Any hit closer than this is considered a numerical error.
        epsilon = 1e-5

        for obj in self.objects:
            # 1. Safety Check: Only call intersect on analytical shapes
            if not hasattr(obj.shape, 'intersect'):
                continue

            try:
                hit_point = obj.shape.intersect(ray)
            except Exception:
                hit_point = None

            if hit_point is not None:
                # 2. Calculate Distance
                # Note: It is faster if your intersect() returns 't' (distance) directly,
                # but calculating norm here works fine for now.
                dist_vec = hit_point - ray.origin
                distance = np.linalg.norm(dist_vec)
                
                # 3. Check bounds (Closest valid hit)
                # MUST check distance > epsilon to prevent "Shadow Acne"
                if distance > epsilon and distance < min_distance:
                    min_distance = distance
                    closest_obj = obj
                    closest_hit = hit_point

        if closest_obj is None or closest_hit is None:
            return HitInfo.miss()

        # 4. Resolve Normal
        normal = np.array([0.0, 1.0, 0.0]) # Default up fallback
        if hasattr(closest_obj.shape, 'GetNormal') and callable(closest_obj.shape.GetNormal):
            normal = closest_obj.shape.GetNormal(closest_hit)
            
            # Normalize safely
            nm = np.linalg.norm(normal)
            if nm > 1e-6:
                normal = normal / nm

        # 5. Return HitInfo
        # Naming 'object' matches the access pattern in 'is_occluded'
        return HitInfo(
            hit=True, 
            point=closest_hit, 
            direction=ray.orientation, 
            normal=normal, 
            distance=min_distance, 
            object=closest_obj 
        )
    
    def get_background_color(self, direction) -> Color:
        """
        Return the background color based on the ray's direction vector.
        Handles Solid Color, ColorGradient (Skybox), or Texture Map safely.
        """
        # 1. Safe Access to Background
        bg = getattr(self, 'background_color', None)
        if bg is None:
            return Color(0.0, 0.0, 0.0)

        # 2. Get the Class Name (Avoids NameError if classes aren't imported)
        bg_type_name = type(bg).__name__

        # 3. Handle ColorGradient (Skybox)
        if bg_type_name == 'ColorGradient':
            # Resolve Direction
            try:
                dir_vec = np.array(direction, dtype=float)
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-6:
                    dir_vec = dir_vec / norm
                else:
                    dir_vec = np.array([0.0, 1.0, 0.0])
            except (ValueError, TypeError):
                dir_vec = np.array([0.0, 1.0, 0.0])

            # Map Y [-1, 1] to [0, 1]
            t = 0.5 * (dir_vec[1] + 1.0)
            return bg.get_color(t)

        # 4. Handle Texture Map (Numpy Array)
        elif isinstance(bg, np.ndarray):
            # Resolve Direction (Reuse logic or recalculate)
            dir_vec = np.array(direction, dtype=float)
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6: dir_vec /= norm
            
            return self._sample_equirectangular_map(bg, dir_vec)

        # 5. Handle Solid Color (Color Object)
        # We check the name OR the instance to be safe
        elif bg_type_name == 'Color' or isinstance(bg, Color):
            return bg
        # Tuple/List fallback
        elif isinstance(bg, (tuple, list)) and len(bg) == 3:
            return Color(*bg)

        return Color(0.0, 0.0, 0.0)
    
    def _sample_equirectangular_map(self, texture: np.ndarray, direction: np.ndarray) -> Color:
        """
        Samples a 2D texture using Spherical (Equirectangular) mapping.
        Texture is assumed to be a numpy array of shape (H, W, 3).
        """
        # Convert 3D Direction -> 2D UV Coordinates
        # u = atan2(z, x) / 2pi + 0.5
        # v = asin(y) / pi + 0.5
        x, y, z = direction
        
        u = atan2(z, x) / (2 * pi) + 0.5
        v = asin(y) / pi + 0.5
        
        # Map UV to Pixel Coordinates
        height, width, _ = texture.shape
        
        # Clamp coordinates and convert to integer indices
        u_idx = int(floor(u * width)) % width
        v_idx = int(floor(v * height))
        v_idx = max(0, min(height - 1, v_idx)) # Clamp vertical to avoid out of bounds
        
        # Retrieve pixel (assume float 0-1 or uint8 0-255)
        pixel = texture[v_idx, u_idx]
        
        # Normalize if the texture is 0-255 (integers)
        if texture.dtype.kind in 'iu': # int or uint
            pixel = pixel / 255.0
            
        return Color(*pixel)
    
    def clear_objects(self):
        self.objects.clear()
        self.lights.clear()
        self.camera = None
    
    def is_occluded(self, point: np.ndarray, light_pos: np.ndarray, bias: float = 1e-4, epsilon: float = 1e-4, max_steps: int = 256) -> bool:
        """
        Return True if there's an occluder between `point` and `light_pos`.
        Uses geometry intersection (get_closest_intersection) if available; otherwise
        falls back to sampling with distance_estimator (SDF ray-march).
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
            v_object: Optional[VObject] = hit_info.object
            if hit_info.hit and v_object is not None and hit_info.point is not None:
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