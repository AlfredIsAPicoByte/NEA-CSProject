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
    
    def get_background_color(self, direction: float) -> Color:
            """
            Return the background color based on the ray's direction vector.
            
            Args:
                direction: The unit vector (Dx, Dy, Dz) pointing from the camera 
                            to the background.
            """
            if not hasattr(self, 'background_color'):
                return Color()  # Default to black if no background_color defined
            
            if not hasattr(self, 'background_gradient_type'):
                self.background_gradient_type = BackgroundType.SOLID_COLOR  # Default type

            # --- 1. Determine the 't' value based on the gradient type ---
            if isinstance(self.background_color, ColorGradient) and not self.background_gradient_type == BackgroundType.SOLID_COLOR:
                # Use the direction vector components (Dx, Dy, Dz)
                Dx, Dy, Dz = direction[0], direction[1], direction[2]
                
                if self.background_gradient_type == BackgroundType.GRADIENT:
                    # Standard sky gradient: dependent on the UP/DOWN component (Y-axis)
                    value = Dy
                elif self.background_gradient_type == BackgroundType.SHPERE_MAP:
                    # Simulates a sky dome. Dependent on the vertical angle (theta).
                    value = Dy # TODO: More complex mapping can be applied here
                else:
                    # Default to vertical if type is unrecognized
                    value = Dy
                
                # Ensures 'value' is strictly [-1, 1] (though ray_direction should be unit length)
                value = np.clip(value, -1.0, 1.0)
                
                t = 0.5 * (value + 1.0)
                
                return self.background_color.get_color(t)

            # --- 2. Handle Array-Like Texture/Environment Map ---
            elif isinstance(self.background_color, np.ndarray) or (isinstance(self.background_color, (list, tuple)) and isinstance(self.background_color[0], (list, tuple, np.ndarray))):
                
                # If the structure is array-like (a list of lists or a 2D/3D NumPy array)
                self.background_gradient_type = BackgroundType.TEXTURE
                
                # The core mechanism for texture/map lookups:
                try:
                    # The 'sample_texture' function must map the 3D 'direction' vector 
                    # to 2D UV coordinates to read the color from the texture array.
                    # 
                    color = self.sample_texture(self.background_color, direction)
                    return color
                except Exception as e:
                    # Fallback if texture sampling fails
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
                print("Setting background color to black due to unknown value type.")
                return Color()
    
    def clear_objects(self):
        self.objects.clear()
        self.lights.clear()
        self.camera = None