import numpy as np
from typing import Optional, Tuple, Any

from src.Data.Ray import TracingRay, RayPool
from src.Data.Color import Color
from ..Core import Algorithm, TracingStats, register_algorithm, update_memory_stats
from . import Intersections
from . import Shading
from src.Image.Film import Film
from src.Data.Sampling import SamplingManager, SampleSettings, Sampler, RandomSampler
from src.Data.Scene import Scene

# TODO: Pool tracing rays and hit info to reduce memory useage at runtime

# RayTracer using strategies
@register_algorithm("ray-tracer")
class RayTracer(Algorithm):
    def __init__(
        self,
        max_recursions: int = 4,
        tile_size: int = 64,
        sampling_manager: Optional[SamplingManager] = None,
        sample_settings: Optional[SampleSettings] = None,
        intersection_strategy: Optional[Intersections.IntersectionStrategy] = None,
        shading_strategy: Optional[Shading.ShadingStrategy] = None,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.sampling_manager = sampling_manager
        self.sample_settings = sample_settings or SampleSettings()

        self.intersector: Intersections.IntersectionStrategy = intersection_strategy or Intersections.InverseSDFIntersection()
        self.shader: Shading.ShadingStrategy = shading_strategy or Shading.LambertShading()
        
        self.max_recursions = max_recursions
        self.tile_size = tile_size

        self.stats: TracingStats = TracingStats()
        self.debug_mode = debug_mode

        # NaN logging: tracks when we last emitted a NaN warning and threshold to avoid spam
        self._last_nan_logged: int = 0
        self._nan_log_threshold: int = 10  # Emit a log every N new NaN events

    def _record_nan(self, reason: str = "", ray: Optional[TracingRay] = None) -> None:
        """Record a NaN event and emit a warning if we've crossed the reporting threshold."""
        self.stats.nan_errors += 1
        # Call logger if threshold reached
        if (self.stats.nan_errors - self._last_nan_logged) >= self._nan_log_threshold:
            self._maybe_log_nan(ray=ray, reason=reason)

    def _maybe_log_nan(self, ray: Optional[TracingRay] = None, reason: str = "") -> None:
        """Emit a short warning about NaN events and advance the last-logged counter.
        Kept minimal to avoid spamming logs; updates `_last_nan_logged` when emitted."""
        delta = self.stats.nan_errors - self._last_nan_logged
        if delta < self._nan_log_threshold:
            return

        loc = f" (ray={ray.name})" if ray is not None else ""
        print(f" NaN events: {self.stats.nan_errors} total{loc}. Reason: {reason}. Emitting summary log.")
        self._last_nan_logged = self.stats.nan_errors

    def _sanitize_color(self, c: Color) -> Color:
        """Turn arbitrary shader output into a finite Color and record NaN events if needed."""
        # 1. Handle None or Invalid types quickly
        if c is None: 
            return np.array([0.0, 0.0, 0.0])
            
        # 2. Extract values
        # Accessing slots directly (c.r) is faster than methods
        vals = np.array([c.r, c.g, c.b], dtype=np.float32)
        
        # 3. Check Finite (Vectorized)
        if not np.isfinite(vals).all():
            self.stats.nan_errors += 1
            return Color(*np.nan_to_num(vals, nan=0.0, posinf=1.0))
            
        return Color(*vals)

    def _trace_ray(self, scene: Scene, ray: TracingRay, recursions_left: int, sampler: Sampler) -> Color:
        # 1. Base Case
        if recursions_left < 0:
            return Color(0.0, 0.0, 0.0)

        # 2. Geometry Intersection
        # The stats object is passed down for internal counters
        hit_info = self.intersector.find_hit(scene, ray, self.stats)

        # 3. Miss -> Background
        if not hit_info.hit:
            # Assuming background settings handle their own safety
            return self.shader.background_settings.get_background_color(ray.orientation)

        # Only do expensive error checking if debugging
        if self.debug_mode:
            # Validate hit_info (point and normal must be finite and present)
            if getattr(hit_info, 'point', None) is None or getattr(hit_info, 'normal', None) is None:
                self._record_nan(reason='hit_info missing point/normal', ray=ray)
                return Color(0.0, 0.0, 0.0)

            if not (np.all(np.isfinite(np.asarray(hit_info.point))) and np.all(np.isfinite(np.asarray(hit_info.normal)))):
                self._record_nan(reason='hit_info point/normal not finite', ray=ray)
                return Color(0.0, 0.0, 0.0)

        # 4. Shading & Recursion
        # The Shader is responsible for casting secondary rays via the 'trace_function' callback
        try:
            color = self.shader.shade(
                scene=scene,
                ray=ray,
                hit_info=hit_info,
                recursions_left=recursions_left,
                trace_function=self._trace_ray,
                sampler=sampler,
                stats=self.stats
            )
        except ArithmeticError as e:
            # Catch divide-by-zero or overflow in shading math
            if self.debug_mode: self._record_nan(reason=f'Arithmetic error: {e}', ray=ray)
            return Color(0.0, 0.0, 0.0)

        if self.debug_mode:
            if color is None:
                self._record_nan(reason='shader returned None', ray=ray)
                return Color(0.0, 0.0, 0.0)

        return color

    def render(
        self,
        scene: Scene,
        sampler: Optional[Sampler] = None,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Film:
        self.stats.reset_ray_counter()
        self.stats.start_timer()

        camera = scene.camera
        if camera is None: raise ValueError("No camera provided in Scene")
        cam_width, cam_height = camera.width, camera.height

        # Create a film for sample_color buffer
        film = Film(cam_width, cam_height)

        rx, ry, rw, rh = region if region else (0, 0, cam_width, cam_height)

        # Initialize output image and sample counter
        self.sample_settings.width = rw
        self.sample_settings.height = rh
        pixels_processed = 0

        # Create default sampler if not provided
        if sampler is None:
            sampler = RandomSampler(self.sample_settings)

        # Optional: Print total tiles for progress tracking
        # total_tiles = ((rw + ts - 1) // ts) * ((rh + ts - 1) // ts)
        # tile_count = 0

        for tile_y in range(ry, ry + rh, self.tile_size):
            for tile_x in range(rx, rx + rw, self.tile_size):
                # Calculate current tile dimensions (handle edges)
                current_w = min(self.tile_size, (rx + rw) - tile_x)
                current_h = min(self.tile_size, (ry + rh) - tile_y)
                
                # Define tile region: (x, y, w, h)
                tile_region = (tile_x, tile_y, current_w, current_h)
                
                # Generate Rays for this Tile ONLY
                rays = camera.generate_screen_rays(region=tile_region, sampler=sampler)
                self.stats.rays_primary += len(rays)

                for ray in rays:
                    if ray is None: continue
                    
                    # Trace Ray
                    pixel_color = self._trace_ray(scene, ray, self.max_recursions, sampler)

                    pixel_color = self._sanitize_color(pixel_color)

                    film.add_pixel_batch(
                        ray.pixel_x,
                        ray.pixel_y,
                        pixel_color.to_np_array(),
                        1.0
                    )

                    # Store in Film Sample Buffer
                    pixels_processed += 1
                
                self.stats = update_memory_stats(self.stats)

        self.stats.pixels_processed = pixels_processed
        self.stats.stop_timer()

        return film