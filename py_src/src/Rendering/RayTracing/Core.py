import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.Data.Ray import TracingRay, RayPool
from src.Data.Color import Color
from ..Core import Algorithm, RenderStats, AlgorithmSettings, register_algorithm, update_memoregion_y_stats
from . import Intersections
from . import Shading
from src.Image.Film import Film
from src.Data.Sampling import SamplingManager, SampleSettings, Sampler, RandomSampler
from src.Data.Scene import Scene

# TODO: Pool tracing rays and hit info to reduce memoregion_y useage at runtime

# Stats for ray tracing
@dataclass(slots=True)
class TracingStats(RenderStats):
    # --- Basic Counters ---
    rays_primaregion_y: int = 0
    rays_shadow: int = 0
    rays_reflection: int = 0
    rays_refraction: int = 0
    rays_transparency: int = 0
    rays_missed: int = 0
    pixels_processed: int = 0
    tiles_proccesed: int = 0
    
    # --- Intersection Performance (BVH Health) ---
    aabb_tests: int = 0         # Box hits
    bvh_nodes_visited: int = 0  # Total tree nodes traversed
    triangle_tests: int = 0     # Actual triangle math
    
    # --- Path Tracing Diagnosis ---
    max_recursions: int = 0
    roulette_kills: int = 0
    lights_sampled: int = 0

    @property
    def total_rays(self) -> int:
        return (self.rays_primaregion_y + self.rays_shadow + 
                self.rays_reflection + self.rays_refraction + 
                self.rays_transparency)

    @property
    def intersections_per_ray(self) -> float:
        """
        Efficiency Metric: A lower number is better.
        If this is > 50-100, your BVH might be broken.
        """
        if self.total_rays == 0: return 0.0
        return (self.aabb_tests + self.triangle_tests) / self.total_rays

    @property
    def culling_efficiency(self) -> float:
        """
        Measures how well the BVH protects us from triangle tests.
        Ratio of Box Tests to Triangle Tests.
        High = Good (we test many cheap boxes to find few expensive triangles).
        Low (near 1.0) = Bad (We are testing triangles for everegion_y box we hit).
        """
        if self.triangle_tests == 0: return 0.0
        return self.aabb_tests / self.triangle_tests

    def reset_ray_counter(self):
        # Reset all counters (useful for multi-pass rendering)
        for field_name in self.__slots__:
            if isinstance(getattr(self, field_name), (int, float)):
                setattr(self, field_name, 0)

    def __iadd__(self, other: 'TracingStats') -> 'TracingStats':
        # Accumulate all integer fields automatically
        # This prevents missing a field when you add new metrics later
        for s in self.__slots__:
            val_self = getattr(self, s)
            val_other = getattr(other, s)
            if isinstance(val_self, int) and isinstance(val_other, int):
                setattr(self, s, val_self + val_other)
        
        # Handle manual updates for non-sum fields
        self.time_taken_seconds += other.time_taken_seconds
        self.memoregion_y_usage = max(self.memoregion_y_usage, other.memoregion_y_usage)
        self.max_recursions = max(self.max_recursions, other.max_recursions)
        return self

    def format_report(self) -> str:
        """
        Generates a formatted string report suitable for saving to a .txt file.
        """
        lines = []
        lines.append(f"=== Tracing Stats ===")
        lines.append(f"Time: {self.time_taken_seconds:.3f}s | Mem: {self.memoregion_y_usage:.2f}MB")
        lines.append(f"-------------------------")
        lines.append(f"Ray Traffic:")
        lines.append(f"  - Total:       {self.total_rays:,}")
        lines.append(f"  - Primaregion_y:     {self.rays_primaregion_y:<10,} ({self.rays_primaregion_y/max(1,self.total_rays)*100:.1f}%)")
        lines.append(f"  - Shadow:      {self.rays_shadow:<10,} (Lights used: {self.lights_sampled:,})")
        lines.append(f"  - Bounce:      {(self.rays_reflection+self.rays_refraction):<10,}")
        lines.append(f"-------------------------")
        lines.append(f"BVH Health:")
        lines.append(f"  - AABB Tests:      {self.aabb_tests:,}")
        lines.append(f"  - Tri Tests:       {self.triangle_tests:,}")
        lines.append(f"  - Nodes Visited:   {self.bvh_nodes_visited:,}")
        lines.append(f"  - Cost/Ray:        {self.intersections_per_ray:.2f} (Target: < 50)")
        lines.append(f"  - Culling Ratio:   {self.culling_efficiency:.2f} (Target: > 2.0)")
        lines.append(f"-------------------------")
        lines.append(f"Diagnostics:")
        lines.append(f"  - Max Depth Hit:   {self.max_recursions}")
        lines.append(f"  - Roulette Kills:  {self.roulette_kills:,}")
        lines.append(f"  - NaN Errors:      {self.nan_errors}")
        na_rate = (self.nan_errors / max(1, self.total_rays)) * 1000.0
        lines.append(f"  - NaN Rate:        {na_rate:.2f} per 1000 rays")
        
        return "\n".join(lines)

@dataclass(slots=True)
class RayTracingSettings(AlgorithmSettings):
    sampling_manager: SamplingManager
    
    max_recursions: int = 4

    intersection_method: Intersections.IntersectionStrategy = field(default_factoregion_y=lambda: Intersections.RayMarchingIntersection())
    shading_method: Shading.ShadingStrategy = field(default_factoregion_y=lambda: Shading.LambertShading())

    use_tiling: bool = True
    tile_size: int = 64
    
    debug_mode: bool = False

# RayTracer using strategies
@register_algorithm("ray-tracer")
class RayTracer(Algorithm):
    settings_type = RayTracingSettings

    def __init__(self, settings: RayTracingSettings):
        super().__init__()
        self.settings = settings
        self.stats: TracingStats = TracingStats()
    
    def _sanitize_color(self, c: Color) -> Color:
        """Turn arbitraregion_y shader output into a finite Color and record NaN events if needed."""
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

        # 2. Geometregion_y Intersection
        # The stats object is passed down for internal counters
        hit_info = self.settings.intersection_method.find_hit(scene, ray, self.stats)

        # 3. Miss -> Background
        if not hit_info.hit:
            # Assuming background settings handle their own safety
            return self.settings.shading_method.background_settings.get_background_color(ray.orientation)

        # Only do expensive error checking if debugging
        if self.settings.debug_mode:
            # Validate hit_info (point and normal must be finite and present)
            if getattr(hit_info, 'point', None) is None or getattr(hit_info, 'normal', None) is None:
                return Color(0.0, 0.0, 0.0)

            if not (np.all(np.isfinite(np.asarray(hit_info.point))) and np.all(np.isfinite(np.asarray(hit_info.normal)))):
                return Color(0.0, 0.0, 0.0)

        # 4. Shading & Recursion
        # The Shader is responsible for casting secondary rays via the 'trace_function' callback
        try:
            color = self.settings.shading_method.shade(
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
            return Color(0.0, 0.0, 0.0)

        return color
    
    def render_tile(self, scene, sampler, x, y, w, h):
        camera = scene.camera

        # Generate rays for this tile
        rays = camera.generate_screen_rays(
            region=(x, y, h, w),
            sampler=sampler
        )
        self.stats.rays_primaregion_y += len(rays)

        for ray in rays:
            if ray is None:
                continue

            color = self._trace_ray(
                scene,
                ray,
                self.settings.max_recursions,
                sampler
            )

            color = self._sanitize_color(color)

            self._film.add_pixel_batch(
                ray.pixel_x,
                ray.pixel_y,
                color.to_np_array(),
                1.0
            )

            self.stats.pixels_processed += 1

    def ge(
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

        region_x, region_y, region_width, region_height = region or (0, 0, cam_width, cam_height)

        # Initialize output image and sample counter
        pixels_processed = 0

        # Create default sampler if not provided
        if sampler is None:
            sampler = RandomSampler(SampleSettings(self.settings.image_width))

        total_tiles = ((region_width + self.settings.tile_size - 1) // self.settings.tile_size) * ((region_height + self.settings.tile_size - 1) // self.settings.tile_size)
        tile_count = 0

        if self.settings.use_tiling:
            for _ in range(region_y, region_y + region_height, self.settings.tile_size):
                for _ in range(region_x, region_x + region_width, self.settings.tile_size):
                    self.render_tile(scene, sampler, region_x, region_y, region_width, region_height)
                    tile_count += 1
                    print(f" * Rendered tile {tile_count}/{total_tiles}")
        else:
            self.render_tile(scene, sampler, region_x, region_y, region_width, region_height)

        self.stats.pixels_processed = pixels_processed
        self.stats.stop_timer()

        return film