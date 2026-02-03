from random import sample
from xmlrpc.client import Boolean
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from src.Data.Ray import TracingRay, RayPool
from src.Data.Color import Color
from .. import register_algorithm
from ..Core import Algorithm, RenderStats, AlgorithmSettings
from . import Intersections
from . import Shading
from src.Image.Film import Film
from src.Data.Sampling.Core import SamplingManager, SampleSettings, Sampler, Sample, reconstruct_pixel, AdaptiveSampler
from src.Data.Scene import Scene

# TODO: Pool tracing rays and hit info to reduce memory useage at runtime

# Stats for ray tracing
@dataclass(slots=True)
class TracingStats(RenderStats):
    # --- Basic Counters ---
    rays_primary: int = 0
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
        return (self.rays_primary + self.rays_shadow + 
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
        Low (near 1.0) = Bad (We are testing triangles for every box we hit).
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
        self.memory_usage = max(self.memory_usage, other.memory_usage)
        self.max_recursions = max(self.max_recursions, other.max_recursions)
        return self

    def format_report(self) -> str:
        """
        Generates a formatted string report suitable for saving to a .txt file.
        """
        lines = []
        lines.append(f"=== Tracing Stats ===")
        lines.append(f"Time: {self.time_taken_seconds:.3f}s | Mem: {self.memory_usage:.2f}MB")
        lines.append(f"-------------------------")
        lines.append(f"Ray Traffic:")
        lines.append(f"  - Total:       {self.total_rays:,}")
        lines.append(f"  - Primary:     {self.rays_primary:<10,} ({self.rays_primary/max(1,self.total_rays)*100:.1f}%)")
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
    sampling_manager: SamplingManager = SamplingManager(SampleSettings(), "random")
    
    max_recursions: int = 4

    intersection_strategy: Intersections.IntersectionStrategy = field(default_factory=lambda: Intersections.RayMarchingIntersection())
    shading_strategy: Shading.ShadingStrategy = field(default_factory=lambda: Shading.LambertShading())

    use_tiling: bool = True
    tile_size: int = 64
    
    debug_mode: bool = False
    verbose_logging: bool = False

# RayTracer using strategies
@register_algorithm("ray-tracer")
class RayTracer(Algorithm):
    settings_type = RayTracingSettings

    def __init__(self, settings: RayTracingSettings):
        super().__init__(settings)
        self.settings = settings
        self.stats: TracingStats = TracingStats()

    def _trace_ray(self, scene: Scene, ray: TracingRay, recursions_left: int, sampler: Sampler) -> Color:
        # 1. Base Case
        if recursions_left < 0:
            return Color(0.0, 0.0, 0.0)

        # 2. Geometry Intersection
        # The stats object is passed down for internal counters
        hit_info = self.settings.intersection_strategy.find_hit(scene, ray, self.stats)

        # 3. Missed First -> Background (The bounced rays return black only if the environmental contribuion is enabled in the shading_strategy)
        if not hit_info.hit:
            if self.settings.shading_strategy.background_settings.environment_effect_enabled:
                if ray.current_depth == 0:
                    return self.settings.shading_strategy.background_settings.get_background_color(ray.orientation)
                
                return self.settings.shading_strategy.background_settings.get_background_color(ray.orientation) * self.settings.shading_strategy.background_settings.environment_contribution_factor
            elif ray.current_depth == 0:
                return self.settings.shading_strategy.background_settings.get_background_color(ray.orientation)
            else:
                return Color(0.0, 0.0, 0.0)

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
            color = self.settings.shading_strategy.shade(
                scene=scene,
                ray=ray,
                hit_info=hit_info,
                recursions_left=recursions_left,
                trace_function=self._trace_ray,
                intersection_function=self.settings.intersection_strategy.find_hit,
                occlusion_function=self.settings.intersection_strategy.is_point_occluded,
                sampler=sampler,
                stats=self.stats
            )
        except ArithmeticError as e:
            # Catch divide-by-zero or overflow in shading math
            return Color(0.0, 0.0, 0.0)

        return color
    
    def render_tile(
            self,
            scene: Scene,
            sampler: Sampler,
            tile_x: int,
            tile_y: int, 
            width: int,
            height: int
        ) -> None:
        camera = scene.camera

        # Generate rays for this tile
        rays = camera.generate_screen_rays(sampler, region=(tile_x, tile_y, width, height))
        self.stats.rays_primary += len(rays)

        # Map: (local_tile_index) -> List[(Sample, Color)]
        tile_samples = [[] for _ in range(width * height)]

        for ray in rays:
            if ray is None:
                continue

            # Global Coordinates
            px, py = ray.pixel_x, ray.pixel_y
                    
            # Convert to Local Tile Coordinates
            local_x = px - tile_x
            local_y = py - tile_y

            if not (0 <= local_x < width and 0 <= local_y < height):
                continue
            
            pixle_color = self._trace_ray(
                scene,
                ray,
                self.settings.max_recursions,
                sampler
            )
            pixle_color = self._sanitize_color(pixle_color)

            image_w = self.settings.sampling_manager.settings.width
            image_h = self.settings.sampling_manager.settings.height
            s_u = getattr(ray, "sample_u", (px + 0.5) / image_w)
            s_v = getattr(ray, "sample_v", (py + 0.5) / image_h)
            sample = Sample(s_u, s_v, 1.0)
            
            local_idx = local_y * width + local_x
            tile_samples[local_idx].append((sample, pixle_color))

            if isinstance(sampler, AdaptiveSampler):
                pixel_colors_so_far = [sc[1].to_np_array(include_alpha=False) for sc in tile_samples[local_idx]]
                if len(pixel_colors_so_far) >= sampler.settings.min_samples:
                    if sampler.has_converged(pixel_colors_so_far):
                        continue  # skip remaining samples for this pixel

            self.stats.pixels_processed += 1

        # Reconstruct Tile
        for i in range(len(tile_samples)):
                samples_and_colors = tile_samples[i]
                
                if not samples_and_colors: continue
                # Calculate Global Pixel Index
                ty = i // width
                tx = i % width
                    
                px = tile_x + tx
                py = tile_y + ty
                    
                # Reconstruct
                samples = [sc[0] for sc in samples_and_colors]
                
                # Convert Color objects to RGBA arrays
                colors = []
                for sc in samples_and_colors:
                    color_obj: Color = sc[1]
                    colors.append(color_obj.to_np_array(include_alpha=False))
                
                rec_rgb = reconstruct_pixel(px, py, samples, colors, self.settings.sampling_manager.settings)
                final_color = Color(*rec_rgb[:3])
                
                self.settings.film.add_pixel_batch(
                    px, py,
                    final_color.to_np_array(),
                    1.0
                )
        self.stats.tiles_proccesed += 1

    def generate_film(
        self,
        scene: Scene,
        sampler: Optional[Sampler] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.stats.reset_ray_counter()
        self.stats.start_timer()

        if self.settings.verbose_logging:
            print(" * Starting rendering...")

        camera = scene.camera
        if camera is None: raise ValueError("No camera provided in Scene")
        cam_width, cam_height = camera.width, camera.height

        # Create a film for sample_color buffer
        self.settings.film = Film(cam_width, cam_height)

        region_x, region_y, region_width, region_height = region or (0, 0, cam_width, cam_height)

        # Initialize output image and sample counter
        pixels_processed = 0

        # Create default sampler if not provided
        if sampler is None:
            sampler = self.settings.sampling_manager.sampler

        total_tiles = ((region_width + self.settings.tile_size - 1) // self.settings.tile_size) * ((region_height + self.settings.tile_size - 1) // self.settings.tile_size)
        tile_count = 0

        if self.settings.use_tiling:
            for tile_y in range(region_y, region_y + region_height, self.settings.tile_size):
                for tile_x in range(region_x, region_x + region_width, self.settings.tile_size):

                    tile_w = min(self.settings.tile_size, region_x + region_width - tile_x)
                    tile_h = min(self.settings.tile_size, region_y + region_height - tile_y)

                    self.render_tile(scene, sampler, tile_x, tile_y, tile_w, tile_h)
                    tile_count += 1
                    
                    if self.settings.verbose_logging:
                        print(f" * Rendered tile {tile_count}/{total_tiles}")
                    
                    if self.settings.debug_mode:
                        # Save intermediate image for debugging
                        Film.save(self.settings.film.get_image(), "_temp.png")
        else:
            self.render_tile(scene, sampler, region_x, region_y, region_width, region_height)

        self.stats.pixels_processed = pixels_processed
        self.stats.lights_sampled = len(Scene.get_nodes_by_type(scene.cache_scene_nodes_flat(), "Light"))
        self.stats.stop_timer()

        if self.settings.verbose_logging:
            print(" * Rendering complete.")
    
    def _sanitize_color(self, color) -> Color:
        """Turn arbitrary shader output into a finite Color and record NaN events if needed."""
        # 1. Handle None or Invalid types quickly
        if color is None: 
            return np.array([0.0, 0.0, 0.0])
            
        # 2. Extract values
        # Accessing slots directly (c.r) is faster than methods
        vals = np.array([color.r, color.g, color.b], dtype=np.float32)
        
        # 3. Check Finite (Vectorized)
        if not np.isfinite(vals).all():
            self.stats.nan_errors += 1
            return Color(*np.nan_to_num(vals, nan=0.0, posinf=1.0))
            
        return Color(*vals)