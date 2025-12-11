from typing import List, Tuple, Optional
import re
import numpy as np
from PIL import Image
from Geometry import VObject, Shape
from Luminance import LightSource, Color
from Camera import VCamera
from PrimaryStructures import Ray

class Scene:
    def __init__(self, name: str = "Scene", camera: Optional[VCamera] = None):
        self.name = name
        self.objects: List[VObject] = []
        self.lights: List[LightSource] = []
        self.camera: Optional[VCamera] = camera
        self.background_color = np.array([0.5, 0.7, 1.0])
    
    def set_camera(self, camera: VCamera):
        self.camera = camera
    
    def add_object(self, obj: VObject):
        self.objects.append(obj)

    def add_light(self, light: LightSource):
        self.lights.append(light)

    def get_lights(self):
        return list(self.lights)

    def distance_estimator(self, point: np.ndarray):
        """Return either a scalar distance or (distance, object). RayMarchingIntersection expects either."""
        min_d = float("inf")
        closest = None
        for obj in self.objects:
            try:
                d = obj.shape.SignedDistance(point)
            except Exception:
                continue
            if d < min_d:
                min_d = d
                closest = obj
        return (min_d, closest)
    
    def get_background_color(self, direction: np.ndarray) -> Tuple[float, float, float]:
        """Return the background color as an RGB tuple based on the direction vector."""
        # Simple gradient based on the y-component of the direction
        t = 0.5 * (direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
    
    def clear(self):
        self.objects.clear()
        self.lights.clear()

    def render(self, algorithim, sampler=None) -> Image:
        """Render the scene using the given algorithm and return a PIL Image."""
        pixel_colors = algorithim.render(self, self.camera, sampler=sampler)
        
        W, H = self.camera.width, self.camera.height
        
        # Convert flat pixel_colors list (row-major) to 2D array
        img_array = np.zeros((H, W, 3), dtype=np.uint8)
        for idx, color in enumerate(pixel_colors):
            y = idx // W
            x = idx % W
            
            # Extract RGB from Color object
            if hasattr(color, "rgba"):
                rgb = color.rgba[:3]
            elif hasattr(color, "to_array"):
                rgb = color.to_array()[:3]
            else:
                rgb = [0.0, 0.0, 0.0]
            
            # Clamp to [0, 1] and convert to [0, 255]
            rgb = np.clip(np.asarray(rgb, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
            img_array[y, x, :] = rgb
        
        im = Image.fromarray(img_array, mode="RGB")
        print(f"Rendered image: {W}x{H}, {len(pixel_colors)} pixel colors")
        
        return im
