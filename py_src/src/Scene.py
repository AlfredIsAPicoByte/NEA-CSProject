from src.Basic import *
from src.Camera import *
from src.Projections import Projection
from src.Lighting import Light


from src.Shapes import Shape
from src.algorithim.Raycast import Ray

class Object:
    def __init__(self, shape: Shape = None, ray: Ray = None, camera: Camera = None, light: Light = None):
        self.shape = shape
        self.ray = ray
        self.camera = camera
        self.light = light
    
    def set_shape(self, shape: Shape):
        self.shape = shape
    def set_ray(self, ray: Ray):
        self.ray = ray
    def set_camera(self, camera: Camera):
        self.camera = camera
    def set_light(self, light: Light):
        self.light = light
    def get_components(self):
        return {
            'shape': self.shape,
            'ray': self.ray,
            'camera': self.camera,
            'light': self.light
        }

class Scene:
    def __init__(self, projection: Projection, defaultCamera: Camera):
        self.projection = projection
        self.objects: list[Object] = [Object(camera = defaultCamera)]
    
    def add_object(self, obj: Shape|Ray|Camera|Light):
        self.objects.append(obj)
    
    def render(self):
        rendered_objects = []
        for obj in self.objects:
            rendered_obj = self.projection.DrawShape(obj)
            rendered_objects.append(rendered_obj)
        return rendered_objects
    
    def transformObjects(self, transform_func):
        for obj in self.objects:
            if obj.shape:
                obj.shape = transform_func(obj.shape)