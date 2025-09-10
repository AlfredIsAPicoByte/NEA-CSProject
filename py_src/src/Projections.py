from src.Shapes import *
from src.Camera import *
from src.Basic import Vector, Ray

class Projection:
    objects = []
    
    def __init__(self, camera: Camera):
        self.camera = camera

    def DrawShape(self, shape: Shape) -> Shape:
        """Project a shape using the camera's parameters."""
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def EraseShape(self, **kwargs) -> bool:
        """Erase a shape from the projection."""
        raise NotImplementedError("This method should be overridden by subclasses")

class OrthographicProjection(Projection):
    def __init__(self, camera: Camera):
        super().__init__(camera)
        camera.type = CameraType.ORTHOGRAPHIC
    
    def DrawShape(self, shape: Shape) -> Shape:
        # For orthographic projection, we can ignore the z-coordinate
        drawn_shape = None

        if isinstance(shape, Circle):
            projected_center = Vector(shape.center[0] / self.camera.fov, shape.center[1] / self.camera.fov, shape.center[2] / self.camera.fov)
            projected_radius = shape.radius / self.camera.fov
            drawn_shape = Circle(projected_center, projected_radius)
        
        if isinstance(shape, Triangle):
            projected_bottom_left = Vector(shape.bottom_left[0], shape.bottom_left[1], 0)
            drawn_shape = Triangle(projected_bottom_left, shape.width, shape.height)

        if drawn_shape:
            self.objects.append(drawn_shape)
            return drawn_shape

        raise NotImplementedError("Orthographic projection not implemented for this shape type")
    
    def DrawRay(self, ray: Ray) -> Ray:
        projected_origin = Vector(ray.origin[0] / self.camera.fov, ray.origin[1] / self.camera.fov, ray.origin[2] / self.camera.fov)
        projected_direction = ray.direction.Normalize()
        drawn_ray = Ray(projected_origin, projected_direction)
        self.objects.append(drawn_ray)
        return drawn_ray

    def EraseObject(self, **kwargs) -> bool:
        if 'shape' in kwargs:
            shape = kwargs['shape']
            if shape in self.objects:
                self.objects.remove(shape)
                return True
            return False
        
        if 'ray' in kwargs:
            ray = kwargs['ray']
            if ray in self.objects:
                self.objects.remove(ray)
                return True
            return False

        if 'index' in kwargs:
            index = kwargs['index']
            if 0 <= index < len(self.objects):
                del self.objects[index]
                return True
            return False
        
        if 'name' in kwargs:
            name = kwargs['name']
            for obj in self.objects:
                if obj.name == name:
                    self.objects.remove(obj)
                    return True
            return False
        
        raise ValueError("Must provide 'shape', 'index', or 'name' to erase a shape")
    
    def EraseAll(self):
        self.objects.clear()
    
    def GetObjects(self):
        return self.objects
    
    def GetObject(self, **kwargs):
        if 'index' in kwargs:
            index = kwargs['index']
            if 0 <= index < len(self.objects):
                return self.objects[index]
            return None
        
        if 'name' in kwargs:
            name = kwargs['name']
            for obj in self.objects:
                if obj.name == name:
                    return obj
            return None

        raise ValueError("Must provide 'index' or 'name' to get a shape")

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
        projected_origin = Vector((ray.origin[0] * self.camera.fov) / ray.origin[2],
                                  (ray.origin[1] * self.camera.fov) / ray.origin[2],
                                  0)
        projected_direction = ray.direction.normalize()
        drawn_ray = Ray(projected_origin, projected_direction)
        self.objects.append(drawn_ray)
        return drawn_ray

    def EraseObject(self, **kwargs) -> bool:
        if 'shape' in kwargs:
            shape = kwargs['shape']
            if shape in self.objects:
                self.objects.remove(shape)
                return True
            return False
        
        if 'ray' in kwargs:
            ray = kwargs['ray']
            if ray in self.objects:
                self.objects.remove(ray)
                return True
            return False
        
        if 'index' in kwargs:
            index = kwargs['index']
            if 0 <= index < len(self.objects):
                del self.objects[index]
                return True
            return False
        
        if 'name' in kwargs:
            name = kwargs['name']
            for obj in self.objects:
                if obj.name == name:
                    self.objects.remove(obj)
                    return True
            return False
        
        raise ValueError("Must provide 'shape', 'index', or 'name' to erase a shape")
    
    def EraseAll(self):
        self.objects.clear()

    def GetObjects(self):
        return self.objects
    
    def GetObject(self, **kwargs):
        if 'index' in kwargs:
            index = kwargs['index']
            if 0 <= index < len(self.objects):
                return self.objects[index]
            return None
        
        if 'name' in kwargs:
            name = kwargs['name']
            for obj in self.objects:
                if obj.name == name:
                    return obj
            return None

        raise ValueError("Must provide 'index' or 'name' to get a shape")
    
    def __repr__(self):
        return f"PerspectiveProjection with {len(self.objects)} objects"