from src.PrimaryStructures import Ray
import numpy as np

"""

"""

class Scene:
    objects = []

    def __init__(self, camera: CameraObject):
        self.camera = camera

    def AddObject(self, obj):
        self.objects.append(obj)

    def NearestObject(self, point: np.ndarray) -> tuple[np.ndarray, object]:
        min_distance = float('inf')
        closest_object = None

        for obj in self.objects:
            if isinstance(obj, Shape):
                distance = 0  # Compute actual distance to the shape if needed
                min_distance = min(min_distance, distance)

                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj
            elif isinstance(obj, Ray):
                # Handle ray-specific distance estimation if needed
                position = point - obj.origin
                direction = obj.direction

                t = np.dot(position, direction) / np.dot(direction, direction)
                if t < 0:
                    distance = np.linalg.norm(position)
                else:
                    closest_point = obj.point_at(t)
                    distance = np.linalg.norm(point - closest_point)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj
        
        return (min_distance, closest_object)

    def GetObjectOfType(self, obj_type):
        return [obj for obj in self.objects if isinstance(obj, obj_type)]

    def __repr__(self):
        return f"Projection(camera={self.camera}"
