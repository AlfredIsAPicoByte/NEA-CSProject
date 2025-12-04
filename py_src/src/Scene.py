from typing import List, Tuple, Optional
import numpy as np
from Geometry import VObject, Shape
from Luminance import LightSource
from Camera import VCamera
from PrimaryStructures import Ray


class Scene:
    def __init__(self, name: str = "Scene", camera: Optional[VCamera] = None):
        self.name = name
        self.objects: List[VObject] = []
        self.lights: List[LightSource] = []
        self.camera: Optional[VCamera] = camera
        self.background_color = np.array([0.5, 0.7, 1.0])
    
    def set_camera(self, camera: VCamera):
        self.camera = camera
    
    def add_object(self, obj: VObject):
        self.objects.append(obj)

    def add_light(self, light: LightSource):
        self.lights.append(light)

    def get_lights(self):
        return list(self.lights)

    def distance_estimator(self, point: np.ndarray):
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
    
    def get_background_color(self, direction: np.ndarray) -> Tuple[float, float, float]:
        """Return the background color as an RGB tuple based on the direction vector."""
        # Simple gradient based on the y-component of the direction
        t = 0.5 * (direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
    
    def clear(self):
        self.objects.clear()
        self.lights.clear()

    def render(self):
        """Placeholder for rendering logic."""
        pass
