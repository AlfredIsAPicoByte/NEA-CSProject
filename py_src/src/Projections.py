from src.Shapes import *
from src.Camera import *
from src.Basic import Ray
import numpy as np

class Projection:
    objects = []

    def __init__(self, camera: Camera):
        self.camera = camera

    def DrawShape(self, shape: Shape) -> Shape:
        """Project a shape using the camera's parameters."""
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def DrawRay(self, ray: Ray) -> Ray:
        """Project a ray using the camera's parameters."""
        raise NotImplementedError("This method should be overridden by subclasses")

class OrthographicProjection(Projection):
    def __init__(self, camera: Camera):
        super().__init__(camera)
        camera.type = CameraType.ORTHOGRAPHIC
    
    def DrawShape(self, shape: Shape) -> Shape:
        # For orthographic projection, we can ignore the z-coordinate
        drawn_shape = None

        if isinstance(shape, Circle):
            projected_center = np.ndarray([shape.center[0] / self.camera.fov, shape.center[1] / self.camera.fov, shape.center[2] / self.camera.fov])
            projected_radius = shape.radius / self.camera.fov
            drawn_shape = Circle(projected_center, projected_radius)
        
        if isinstance(shape, Triangle):
            projected_bottom_left = np.ndarray([shape.bottom_left[0], shape.bottom_left[1], 0])
            drawn_shape = Triangle(projected_bottom_left, shape.width, shape.height)

        if drawn_shape:
            self.objects.append(drawn_shape)
            return drawn_shape

        raise NotImplementedError("Orthographic projection not implemented for this shape type")
    
    def DrawRay(self, ray: Ray) -> Ray:
        projected_origin = np.ndarray([ray.origin[0] / self.camera.fov, ray.origin[1] / self.camera.fov, ray.origin[2] / self.camera.fov])
        projected_direction = ray.direction.Normalize()
        drawn_ray = Ray(projected_origin, projected_direction)
        self.objects.append(drawn_ray)
        return drawn_ray

    def __repr__(self):
        return f"OrthographicProjection with {len(self.objects)} objects"
    
class PerspectiveProjection(Projection):
    def __init__(self, camera: Camera):
        super().__init__(camera)
        camera.type = CameraType.PERSPECTIVE
    
    def DrawShape(self, shape: Shape) -> Shape:
        # For perspective projection, we need to account for depth
        drawn_shape = None

        if isinstance(shape, Circle):
            raise NotImplementedError("Perspective projection not implemented for Circle shape")
        
        if isinstance(shape, Triangle):
            raise NotImplementedError("Perspective projection not implemented for Triangle shape")

        if drawn_shape:
            self.objects.append(drawn_shape)
            return drawn_shape

        raise NotImplementedError("Perspective projection not implemented for this shape type")
    
    def DrawRay(self, ray: Ray) -> Ray:
        if ray.origin[2] == 0:
            raise ValueError("Ray origin cannot be at z=0 for perspective projection")
        projected_origin = np.ndarray([(ray.origin[0] * self.camera.fov) / ray.origin[2],
                                  (ray.origin[1] * self.camera.fov) / ray.origin[2],
                                  0])
        projected_direction = ray.direction.normalize()
        drawn_ray = Ray(projected_origin, projected_direction)
        self.objects.append(drawn_ray)
        return drawn_ray

    def __repr__(self):
        return f"PerspectiveProjection with {len(self.objects)} objects"