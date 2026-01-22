from typing import List, Optional

from .Camera import Camera
from src.Geometry.Primitive import Primitive
from src.Lighting.Core import LightSource

class Scene:
    """
    A container for objects and lights used in rendering algorithms
    """
    def __init__(self, name: str = "Scene", camera: Optional[Camera] = None, **kwargs):
        self.name = name
        self.camera: Camera = camera or Camera()

        self.objects: List[Primitive] = []
        self.lights: List[LightSource] = []
        
        self._version: int = 1
        self._cache_objects: Optional[List[Primitive]] = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def version(self) -> int:
        """
        A counter for the amount of changes that have occured on the scene since initilization.
        Starts at `1` when initialized.
        
        :return: Description
        :rtype: int
        """
        return self._version
    
    def get_objects_flat(self) -> List[Primitive]:
        """
        Docstring for get_objects_flat
        
        :return: Description
        :rtype: List[Primitive]
        """
        if self._cache_objects is None:
            self.flatten_objects()

        return self._cache_objects or self.objects
    
    def flatten_objects(self):
        """
        Retruns a list of all objects in a flat list array and updated the cache for faster access
        """
        flat_list = []
        
        # We store the ID (memory address) of visited objects
        visited_ids = set() 
        
        # Initialize stack with the top-level objects
        stack = list(self.objects)

        while stack:
            current_obj = stack.pop()

            # 1. Cycle Detection: Have we seen this specific object instance before?
            if id(current_obj) in visited_ids:
                continue
                
            # 2. Mark as visited
            visited_ids.add(id(current_obj))
            
            # 3. Add to our result list
            flat_list.append(current_obj)

            # 4. Add children to the stack to be processed next
            # (Check if the object actually has children first)
            if hasattr(current_obj, 'children') and current_obj.children:
                stack.extend(current_obj.children)
        
        # remove duplicate objects (with the same id)
        self._cache_objects = list(set(flat_list))
    
    def set_camera(self, camera: Camera):
        """
        Change the current camera that is used.
        Doesn't update the version counter.
        """
        self.camera = camera

    def add_object(self, obj: Primitive):
        """
        Adds an object to the scene. 
        """
        self.objects.append(obj)
        self.update_version()

    def add_light(self, light: LightSource):
        """
        Adds a light source to the scene.
        """
        self.lights.append(light)
        self.update_version()

    def clear(self):
        """
        Removes all the objects and light sources from the scene, while updating the version counter.
        """
        self.objects.clear()
        self.lights.clear()
        self.update_version()

    def reset(self):
        """
        Removes all the objects and light sources from the scene, without updating the version counter.
        """
        self.objects.clear()
        self.lights.clear()
    
    def update_version(self):
        """
        Signal a change in the scene.
        """
        self._version += 1