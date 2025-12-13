from typing import List, Tuple, Optional
import numpy as np
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
            Accepts:
              - direction: sequence/np.ndarray of length 3 (direction vector or euler rotation)
            """
            if not hasattr(self, 'background_color'):
                return Color()  # Default to black if no background_color defined

            # Normalize/resolve direction input: accept np arrays, lists, or Euler rotation fallback
            try:
                dir_arr = np.asarray(direction, dtype=float)
            except Exception:
                # fallback to camera forward if available
                dir_arr = None

            if dir_arr is None or dir_arr.size != 3:
                # If caller passed rotation Euler angles or invalid data, try camera forward
                dir_vec = getattr(getattr(self, "camera", None), "transform", None)
                if dir_vec is not None:
                    dir_vec = getattr(self.camera.transform, "forward", np.array([0.0, 0.0, 1.0]))
                else:
                    dir_vec = np.array([0.0, 0.0, 1.0])
            else:
                dir_vec = dir_arr

            # ensure unit length and safe numeric handling
            if np.linalg.norm(dir_vec) > 0:
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
            else:
                dir_vec = np.array([0.0, 0.0, 1.0])

            if not hasattr(self, 'background_gradient_type'):
                # choose a reasonable default; if background_color is a ColorGradient prefer GRADIENT
                if isinstance(self.background_color, ColorGradient):
                    self.background_gradient_type = BackgroundType.GRADIENT
                else:
                    self.background_gradient_type = BackgroundType.SOLID_COLOR

            # --- 1. Determine the 't' value based on the gradient type ---
            if isinstance(self.background_color, ColorGradient):
                # Determine parameter 't' based on chosen gradient type (default=vertical)
                if self.background_gradient_type == BackgroundType.GRADIENT:
                    value = dir_vec[1]  # vertical (Y)
                elif self.background_gradient_type == BackgroundType.SHPERE_MAP:
                    value = dir_vec[1]  # simple dome mapping; can be enhanced
                else:
                    value = dir_vec[1]
                value = np.clip(value, -1.0, 1.0)
                t = 0.5 * (value + 1.0)
                return self.background_color.get_color(t)
 
            # --- 2. Handle Array-Like Texture/Environment Map ---
            elif isinstance(self.background_color, np.ndarray) or (isinstance(self.background_color, (list, tuple)) and isinstance(self.background_color[0], (list, tuple, np.ndarray))):
                self.background_gradient_type = BackgroundType.TEXTURE
                try:
                    color = self.sample_texture(self.background_color, dir_vec)
                    return color
                except Exception as e:
                    print(f"Error sampling background texture: {e}. Defaulting to black.")
                    return Color()
 
            # --- 3. Handle Solid Color / Simple Tuple Fallback ---
            elif isinstance(self.background_color, Color):
                self.background_gradient_type = BackgroundType.SOLID_COLOR
                return self.background_color
            elif isinstance(self.background_color, (tuple, list)) and len(self.background_color) == 3:
                self.background_gradient_type = BackgroundType.SOLID_COLOR
                r, g, b = self.background_color
                return Color(r, g, b)
            else:
                self.background_gradient_type = BackgroundType.SOLID_COLOR
                return Color(1.0, 0.0, 1.0)
    
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