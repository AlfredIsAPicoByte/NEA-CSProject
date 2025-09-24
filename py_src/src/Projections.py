from src.Shapes import *
from src.Camera import *
from src.Basic import Ray
import numpy as np

class Projection:
    objects = []

    def __init__(self, camera: Camera):
        self.camera = camera

    def AddObject(self, obj: Shape|Ray):
        self.objects.append(obj)
    
    def Display(self):
        pass

    def Project(self, point: np.ndarray) -> np.ndarray:
        pass

    def DistanceEstimator(self, point: np.ndarray) -> float:
        min_distance = float('inf')

        for obj in self.objects:
            if isinstance(obj, Shape):
                if obj.CheckPoint(point):
                    distance = 0  # Compute actual distance to the shape if needed
                    min_distance = min(min_distance, distance)
                else:
                    continue # Skip if point is not near the shape
            elif isinstance(obj, Ray):
                # Handle ray-specific distance estimation if needed
                pass
        
        return min_distance

    def __repr__(self):
        return f"Projection(camera={self.camera}, objects={self.objects})"
    
class OrthographicProjection(Projection):
    def __init__(self, camera: Camera):
        super().__init__(camera)

    def Project(self, point: Vector) -> Vector:
        # Orthographic projection simply drops the z-coordinate
        return Vector(point.x, point.y, 0)
