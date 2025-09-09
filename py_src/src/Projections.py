from src.Shapes import *
from src.Camera import *
from src.Basic import Ray

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
        camera.type = CameraMode.ORTHOGRAPHIC
    
    def DrawShape(self, shape: Shape) -> Shape:
        # For orthographic projection, we can ignore the z-coordinate
        drawn_shape = None

        if isinstance(shape, Circle):
            projected_center = Vector(shape.center[0], shape.center[0], 0)
            drawn_shape = Circle(projected_center, shape.radius)
        
        if isinstance(shape, Triangle):
            projected_bottom_left = Vector(shape.vertex1[0], shape.vertex1[1], 0)
            drawn_shape = Triangle(projected_bottom_left, shape.width, shape.height)

        if drawn_shape:
            self.objects.append(drawn_shape)
            return drawn_shape

        raise NotImplementedError("Orthographic projection not implemented for this shape type")
    
    def EraseShape(self, **kwargs) -> bool:
        if 'shape' in kwargs:
            shape = kwargs['shape']
            if shape in self.objects:
                self.objects.remove(shape)
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