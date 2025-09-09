from src.Shapes import *
from src.Camera import *
from src.Basic import Ray

class Projection2D:
    def __init__(self, camera: Camera):
        self.camera = camera

    def DrawShape(self, shape: Shape) -> Shape:
        """Project a shape using the camera's parameters."""
        raise NotImplementedError("This method should be overridden by subclasses")

    def ProjectIntersections(self, ray: 'Ray', shape: Shape):
        """Compute intersection between a ray and a shape."""
        raise NotImplementedError("This method should be overridden by subclasses")

class OrthographicProjection2D(Projection2D):
    def __init__(self, camera: Camera):
        super().__init__(camera)
        camera.type = CameraMode.ORTHOGRAPHIC
    
    def DrawShape(self, shape: Shape) -> Shape:
        # For orthographic projection, we can ignore the z-coordinate
        if isinstance(shape, Circle):
            projected_center = Vector(shape.center[0], shape.center[0], 0)
            return Circle(projected_center, shape.radius)
        else:
            raise NotImplementedError("Orthographic projection not implemented for this shape type")
    
    def ProjectIntersections(self, ray: Ray, shape: Shape):
        if isinstance(shape, Circle):
            # For orthographic projection, we can ignore the z-component of the ray
            projected_ray = Ray(Vector(ray.origin[0], ray.origin[1], 0), Vector(ray.direction[0], ray.direction[1], 0).Normalize())
            return shape.CheckIntersection(projected_ray)
        else:
            raise NotImplementedError("Orthographic projection intersection not implemented for this shape type")
        
class PerspectiveProjection2D(Projection2D):
    def __init__(self, camera: Camera):
        super().__init__(camera)
        camera.mode = CameraMode.FIRST_PERSON
    
    def DrawShape(self, shape: Shape) -> Shape:
        # For perspective projection, we need to apply the camera's projection matrix
        if isinstance(shape, Circle):
            # Transform the center of the circle
            projected_center = self.camera.get_camera_matrix() * Vector(shape.center[0], shape.center[1], shape.center[2], 1)
            # Perspective divide
            if projected_center.w != 0:
                projected_center = Vector(projected_center.x / projected_center.w, projected_center.y / projected_center.w, projected_center.z / projected_center.w)
            return Circle(projected_center, shape.radius)  # Note: radius may need adjustment based on depth
        else:
            raise NotImplementedError("Perspective projection not implemented for this shape type")
    
    def ProjectIntersections(self, ray: Ray, shape: Shape):
        if isinstance(shape, Circle):
            return shape.CheckIntersection(ray)
        else:
            raise NotImplementedError("Perspective projection intersection not implemented for this shape type")