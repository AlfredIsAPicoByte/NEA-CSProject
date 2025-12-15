from typing import List, Tuple, Optional
import numpy as np
from math import atan2, asin, pi, floor

from PrimaryStructures import Ray
from Camera import VCamera
from Geometry import VObject
from Luminance import LightSource, Color, ColorGradient

class BackgroundType:
    SOLID_COLOR = 0
    GRADIENT = 1
    SHPERE_MAP = 2
    TEXTURE = 3

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

    def distance_estimator(self, point: np.ndarray) -> Tuple[float, Optional[VObject]]:
        """Return either a scalar distance or (distance, object). RayMarchingIntersection expects either."""
        min_d = float("inf")
        closest = None
        for obj in self.objects:
            try:
                d = obj.shape.SignedDistance(point)
            except Exception:
                continue
            if d < min_d:
                min_d = d
                closest = obj
        return (min_d, closest)
    
    def get_closest_intersection(self, ray: Ray) -> Tuple[Optional[VObject], Optional[np.ndarray]]:
        """Return the closest intersected object and the hit point, or (None, None) if no intersection."""
        closest_obj = None
        closest_hit = None
        min_distance = float("inf")
        
        for obj in self.objects:
            hit_point = obj.shape.intersect(ray)
            if hit_point is not None:
                distance = np.linalg.norm(hit_point - ray.origin)
                if distance < min_distance:
                    min_distance = distance
                    closest_obj = obj
                    closest_hit = hit_point
        
        return closest_obj, closest_hit
    
    def get_background_color(self, direction) -> Color:
        """
        Return the background color based on the ray's direction vector.
        Handles Solid Color, ColorGradient (Skybox), or Texture Map (Equirectangular).
        """
        # 1. Safe Access to Background Property
        bg = getattr(self, 'background_color', None)
        if bg is None:
            return Color(0.0, 0.0, 0.0) # Default Black

        # 2. Resolve Direction Vector
        # We need a normalized numpy array direction vector
        try:
            dir_vec = np.array(direction, dtype=float)
            if dir_vec.shape != (3,):
                 raise ValueError("Invalid shape")
        except (ValueError, TypeError):
            # Fallback: If direction is invalid (e.g. None), use Camera Forward
            camera = getattr(self, "camera", None)
            if camera and hasattr(camera, "transform"):
                dir_vec = camera.transform.forward
            else:
                dir_vec = np.array([0.0, 0.0, 1.0]) # Absolute Z forward fallback

        # Normalize logic (safeguard against zero-length vectors)
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-6:
            dir_vec = dir_vec / norm
        else:
            dir_vec = np.array([0.0, 1.0, 0.0]) # Default UP if zero vector

        # 3. Handle ColorGradient (Skybox / Vertical Gradient)
        # Using type name string check avoids circular import issues if they exist
        if type(bg).__name__ == 'ColorGradient' or hasattr(bg, 'get_color'):
            # Calculate 't' for vertical gradient mapping
            # Map Y (Up) from [-1, 1] to [0, 1]
            t = 0.5 * (dir_vec[1] + 1.0)
            return bg.get_color(t)

        # 4. Handle Texture Map (Environment / HDRI Map)
        # Checks if it's a numpy array (image data)
        elif isinstance(bg, np.ndarray):
            return self._sample_equirectangular_map(bg, dir_vec)

        # 5. Handle Solid Color (Color Object)
        elif isinstance(bg, Color):
            return bg

        # 6. Handle Solid Color (Tuple/List fallback)
        elif isinstance(bg, (tuple, list)) and len(bg) == 3:
            return Color(bg[0], bg[1], bg[2])

        # Default fallback
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
            
        return Color(pixel[0], pixel[1], pixel[2])
    
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
            closest_obj, hit_point = self.get_closest_intersection(ray)
            if closest_obj is not None and hit_point is not None:
                hit_dist = np.linalg.norm(hit_point - origin)
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
                de = self.distance_estimator(sample_point)
                # distance_estimator may return (dist, obj) or dist only
                if isinstance(de, tuple):
                    d, _ = de
                else:
                    d = de
                if d <= epsilon:
                    # hit something before reaching the light -> occluded
                    return True
                distance_traveled += d
                if distance_traveled >= dist_to_light:
                    # reached light without hitting anything
                    return False
            # exceeded max steps without reaching the light - treat as occluded
            return True

        # If no approach worked, conservatively say not occluded
        return False