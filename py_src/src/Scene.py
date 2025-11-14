import numpy as np
from enum import Enum

from PrimaryStructures import Transform, Ratio
from Geometry import VObject
from Camera import VCamera
from Luminance import LightSource

"""

"""

class Scene:
    def __init__(self, dimension: int = 2, name: str = "Scene"):
        if dimension not in (2, 3):
            raise AttributeError("Scene dimension must be either 2 or 3")
        self.dimension = dimension

        self.objects: list[VObject] = []
        self.lights: list[LightSource] = []
        self.cam: VCamera = None
        self.name = name
    
    def set_camera(self, camera: VCamera):
        self.cam = camera
    
    def add_object(self, obj: VObject):
        self.objects.append(obj)
    
    def add_light(self, light: LightSource):
        self.lights.append(light)

    def distance_estimator(self, point: np.ndarray) -> tuple[float, VObject]:
        """Returns the minimum distance from the point to any object in the scene."""

        min_dist = float("inf")
        closest_object = None

        for vobject in self.objects:
            # Placeholder for actual distance calculation
            dist = np.linalg.norm(point - vobject.transform.position)

            if dist < min_dist:
                min_dist = dist
                closest_object = vobject
        
        if closest_object is not None:
            min_dist = np.linalg.norm(point - vobject.shape.GetClosestPoint(point))
            return min_dist, closest_object
        else:
            return float("inf"), None

    def __repr__(self):
        return f"Scene(camera={self.cam}, objects={len(self.objects)}, lights={len(self.lights)})"
