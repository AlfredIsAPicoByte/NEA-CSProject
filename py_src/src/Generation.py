from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

from PrimaryStructures import TracingRay
from Camera import VCamera
from Sampling import Sampler

class RayGenerationStrategy(ABC):
    @abstractmethod
    def generate(
        self,
        camera: VCamera,
        sampler: Sampler,
        region: Optional[Tuple[int, int, int, int]] = None,  # (x1, y1, w, h)
    ) -> List[TracingRay]:
        ...

class RayGenerator(RayGenerationStrategy):
    """
    Generate camera rays
    - Iterates over the requested region.
    - Ask the Sampler for (u,v) offsets for every pixel.
    - Calculates the Ray Origin and Direction based on Camera Type.
    """
    def generate(
        self,
        camera: VCamera,
        sampler: Sampler,
        region: Optional[Tuple[int, int, int, int]] = None, # (x, y, w, h)
    ) -> List[TracingRay]:
        cam_width, cam_height = camera.width, camera.height
        
        # 1. Resolve Region
        if region is None:
            x_start, y_start, region_w, region_h = 0, 0, cam_width, cam_height
        else:
            x_start, y_start, req_w, req_h = region
            # Clamp to image bounds
            x_start = max(0, min(x_start, cam_width))
            y_start = max(0, min(y_start, cam_height))
            region_w = max(0, min(req_w, cam_width - x_start))
            region_h = max(0, min(req_h, cam_height - y_start))

        rays: List[TracingRay] = []

        # 2. Iterate over pixels
        for y in range(y_start, y_start + region_h):
            for x in range(x_start, x_start + region_w):
                # 3. Get Samples
                # The SamplingManager returns a list of Sample objects (offsets 0.0-1.0)
                # matching the configured Samples Per Pixel (SPP).

                pixel_samples = sampler.get_samples_per_pixel(x, y)
                for i, sample in enumerate(pixel_samples):
                    # Normalize sample coordinates to 0..1 for camera functions
                    screen_x = (x + sample.u) / float(cam_width)
                    screen_y = (y + sample.v) / float(cam_height)
                    
                    # 4. Calculate Ray Geometry
                    ray_origin, ray_orientation = camera.get_camera_ray(screen_x, screen_y)

                    # 5. Build Ray
                    ray = TracingRay(
                        origin=ray_origin,
                        orientation=ray_orientation,
                        pixel_x=x,
                        pixel_y=y,
                        sample_u=sample.u,
                        sample_v=sample.v,
                        name=f"ray#{i}_({x},{y})",
                    )
                    rays.append(ray)
        return rays