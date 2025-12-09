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

    def render(self, algorithim, sampler = None) -> Image:
        rays, hits, output_rays = algorithim.render(self, self.camera, sampler = sampler)

        # Ray names are "Camera Ray (x,y) #r"
        rx = re.compile(r'Camera Ray \((\d+),(\d+)\)')

        # Prepare accumulation buffers (float)
        W, H = self.camera.width, self.camera.height
        accum = np.zeros((H, W, 3), dtype=np.float64)
        counts = np.zeros((H, W), dtype=int)
        for r in output_rays:
            x = getattr(r, "pixel_x", None)
            y = getattr(r, "pixel_y", None)
            if x is None or y is None:
                # fallback: parse name as "Camera Ray (x,y) #r"
                m = rx.search(r.name)
                if not m: continue
                x = int(m.group(1)); y = int(m.group(2))
            c: Color = getattr(r, "final_color", None) or getattr(r, "color", None) or getattr(r, "base_color", None)
            if c is None: continue
            arr = np.asarray(c.rgba if hasattr(c, "rgba") else c)
            accum[y, x, :] += arr[:3]
            counts[y, x] += 1
        
        # Avoid division by zero; where no samples, use background color from scene
        mask = counts > 0
        accum[mask] = accum[mask] / counts[mask][:, None]

        # Fill empty pixels with background color
        bg = np.asarray(self.background_color, dtype=np.float64)
        accum[~mask] = bg

        # Convert to uint8 and save
        # Colors are represented in [0.0, 1.0]; scale to 0-255 before clamping/conversion.
        img8 = np.clip(accum * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(img8, mode="RGB")

        print("Number of Rays: " + len(rays).__str__() + ", Number of Hits: " + len(hits).__str__())
        
        return im
