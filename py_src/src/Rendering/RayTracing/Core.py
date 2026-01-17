import numpy as np
from typing import Optional, Tuple, Any

from src.Data.Ray import TracingRay, RayPool
from src.Data.Color import Color
from ..Core import Algorithm, TracingStats, register_algorithm, update_memory_stats
from . import Intersections
from . import Shading
from . import Interactions
from src.Image.Film import Film
from src.Data.Sampling import SamplingManager, SampleSettings, Sampler, RandomSampler
from src.Data.Scene import Scene

# TODO: Pool tracing rays and hit info to reduce memory useage at runtime

# Raytracer using strategies
@register_algorithm("ray-tracer")
class RayTracer(Algorithm):
    def __init__(
        self,
        max_recursions: int = 4,
        sampling_manager: Optional[SamplingManager] = None,
        intersection_strategy: Optional[Intersections.IntersectionStrategy] = None,
        interaction_strategy: Optional[Interactions.InteractionStrategy] = None,
        shading_strategy: Optional[Shading.ShadingStrategy] = None,
        sample_settings: Optional[SampleSettings] = None,
    ):
        super().__init__()
        self.sampling_manager = sampling_manager
        self.sample_settings = sample_settings or SampleSettings()

        self.intersector: Intersections.IntersectionStrategy = intersection_strategy if intersection_strategy is not None else Intersections.RayMarchingIntersection()
        self.shader: Shading.ShadingStrategy = shading_strategy if shading_strategy is not None else Shading.LambertShading()
        self.interactor: Interactions.InteractionStrategy = interaction_strategy if interaction_strategy is not None else Interactions.StandardInteraction()
        
        self.max_recursions = max_recursions

        self.stats: TracingStats = TracingStats()

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

    def _sanitize_color(self, c: Any, ray: Optional[TracingRay] = None, reason: str = "") -> Color:
        """Turn arbitrary shader output into a finite Color and record NaN events if needed."""
        if c is None:
            self._record_nan(reason=reason, ray=ray)
            return Color(0.0, 0.0, 0.0)
        try:
            vals = np.array([c.r, c.g, c.b], dtype=float)
        except Exception:
            self._record_nan(reason=reason, ray=ray)
            return Color(0.0, 0.0, 0.0)

        if not np.all(np.isfinite(vals)):
            # Record NaN and clamp values
            self._record_nan(reason=reason, ray=ray)
            vals = np.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)
        return Color(float(vals[0]), float(vals[1]), float(vals[2]), getattr(c, 'a', 1.0))

    def _trace_ray(self, scene: Scene, ray: TracingRay, recursions_left: int, sampler: Sampler) -> Color:
        # 1. Base Case
        if recursions_left < 0:
            self._record_nan(reason='negative recursion depth', ray=ray)
            return Color(0.0, 0.0, 0.0)

        # 2. Geometry Intersection
        hit_info = self.intersector.find_hit(scene, ray, self.stats)

        if not hit_info.hit:
            return self.shader.background_settings.get_background_color(ray.orientation)

        # Validate hit_info (point and normal must be finite and present)
        if getattr(hit_info, 'point', None) is None or getattr(hit_info, 'normal', None) is None:
            self._record_nan(reason='hit_info missing point/normal', ray=ray)
            return Color(0.0, 0.0, 0.0)

        if not (np.all(np.isfinite(np.asarray(hit_info.point))) and np.all(np.isfinite(np.asarray(hit_info.normal)))):
            self._record_nan(reason='hit_info point/normal not finite', ray=ray)
            return Color(0.0, 0.0, 0.0)

        # 3. Direct Lighting / Local Shading
        # (Assuming this calculates direct light or calls recursion for mirror reflections)
        shaded_color = self.shader.shade(
            scene=scene, 
            ray=ray, 
            hit_info=hit_info,
            recursions_left=recursions_left - 1,
            trace_function=self._trace_ray,
            sampler=sampler,
            stats=self.stats
        )

        # Sanitize shaded color; track NaNs
        if shaded_color is None:
            self._record_nan(reason='shader returned None', ray=ray)
            shaded_color = Color(0.0, 0.0, 0.0)
        shaded_color = self._sanitize_color(shaded_color, ray=ray, reason='shaded output')

        return shaded_color

    def render(
        self,
        scene: Scene,
        sampler: Optional[Sampler] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        tile_size: Optional[int] = None,
    ) -> Film:
        self.stats.reset_ray_counter()
        self.stats.start_timer()

        camera = scene.camera
        if camera is None: raise ValueError("No camera provided in Scene")
        cam_width, cam_height = camera.width, camera.height

        # Ensure object world transforms are up-to-date so BVH & intersections use correct positions
        try:
            for obj in scene.objects:
                obj.update_world_matrices()
        except Exception:
            # If scene contains non-Primitive items, ignore and proceed
            pass

        # Create a film for sample_color buffer
        film = Film(cam_width, cam_height)

        if region:
            rx, ry, rw, rh = region
        else:
            rx, ry, rw, rh = 0, 0, cam_width, cam_height

        # Initialize output image and sample counter
        self.sample_settings.width = rw
        self.sample_settings.height = rh
        pixels_processed = 0

        # Handle tile_size parameter
        ts = tile_size if tile_size is not None else 64

        # Create default sampler if not provided
        if sampler is None:
            sampler = RandomSampler(self.sample_settings)

        # Optional: Print total tiles for progress tracking
        # total_tiles = ((rw + ts - 1) // ts) * ((rh + ts - 1) // ts)
        # tile_count = 0

        for tile_y in range(ry, ry + rh, ts):
            for tile_x in range(rx, rx + rw, ts):
                # Calculate current tile dimensions (handle edges)
                current_w = min(ts, (rx + rw) - tile_x)
                current_h = min(ts, (ry + rh) - tile_y)
                
                # Define tile region: (x, y, w, h)
                tile_region = (tile_x, tile_y, current_w, current_h)
                
                # Generate Rays for this Tile ONLY
                rays = camera.generate_screen_rays(region=tile_region, sampler=sampler)
                self.stats.rays_primary += len(rays)

                self.stats = update_memory_stats(self.stats) # type: ignore

                for ray in rays:
                    if ray is None: continue
                    
                    # Trace Ray
                    pixel_color = self._trace_ray(scene, ray, self.max_recursions, sampler)

                    # sample = Sample(ray.sample_u, ray.sample_v, 1.0) # weight 1.0

                    film.add_pixel_batch(
                        ray.pixel_x,
                        ray.pixel_y,
                        pixel_color.to_np_array(),
                        1.0
                    )

                    # Store in Film Sample Buffer
                    pixels_processed += 1

        self.stats.pixels_processed = pixels_processed
        self.stats.stop_timer()

        return film