from src.PrimaryStructures import Transform, Ratio
from src.Camera import CameraObject
from Luminance import LightRay, LightSource
import numpy as np
from enum import Enum

"""

"""


class SceneType(Enum):
    SCENE_2D = 1
    SCENE_3D = 2

class Scene:
    def __init__(self, stype: SceneType):
        self.type = stype
        self.objects = []
        self.lights: LightSource = []
        self.cam: CameraObject = None
    
    def set_camera(self, camera: CameraObject):
        self.cam = camera
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, light):
        self.lights.append(light)

    
