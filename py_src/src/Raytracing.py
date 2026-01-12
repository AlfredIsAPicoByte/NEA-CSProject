from typing import Optional, List, Tuple
from dataclasses import dataclass

from PrimaryStructures import TracingRay
from Scene import Scene
from Luminance import Color
from RenderingAlgorithms import Algorithm, RenderStats, register_algorithm, update_memory_stats
from Sampling import SamplingManager, SampleSettings, Sampler, Sample, reconstruct_pixel, RandomSampler
import Generation
import Intersections
import Shading
import Interactions

# TODO: Pool tracing rays and hit info to reduce memory useage at runtime
# TODO: Localise stat updates, dont use global referenced up until the end of the logic
# TODO: Freeup large object that aren't in use
# TODO: Figure out how to simplify memory intensive logic into chunks
    
# Stats for ray tracing
@dataclass(slots=True)
class TracingStats(RenderStats):
    # --- Basic Counters ---
    rays_primary: int = 0
    rays_shadow: int = 0
    rays_reflection: int = 0
    rays_refraction: int = 0
    rays_transparency: int = 0  # NEW: Rays passing through alpha cutouts
    missed_rays: int = 0
    
    # --- Intersection Performance (BVH Health) ---
    aabb_tests: int = 0         # Box hits
    bvh_nodes_visited: int = 0  # Total tree nodes traversed
    triangle_tests: int = 0     # Actual triangle math
    
    # --- Path Tracing Diagnosis ---
    max_recursions: int = 0
    roulette_kills: int = 0
    lights_sampled: int = 0
    
    # --- Logic & Debugging ---
    pixels_processed: int = 0
    nan_errors: int = 0      

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
        NEW: Measures how well the BVH protects us from triangle tests.
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
        lines.append(f"Path Diagnostics:")
        lines.append(f"  - Max Depth Hit:   {self.max_recursions}")
        lines.append(f"  - Roulette Kills:  {self.roulette_kills:,}")
        lines.append(f"  - NaN Errors:      {self.nan_errors}")
        
        return "\n".join(lines)
    
    def print_verbose_report(self):
        print()
        print(self.format_report())

# Raytracer using strategies
@register_algorithm("raytracer")
class Raytracer(Algorithm):
    def __init__(
        self,
        max_recursions: int = 4,
        sampling_manager: Optional[SamplingManager] = None,
        ray_generator: Optional[Generation.RayGenerationStrategy] = None,
        intersection_strategy: Optional[Intersections.IntersectionStrategy] = None,
        interaction_strategy: Optional[Shading.ShadingStrategy] = None,
        shading_strategy: Optional[Interactions.InteractionStrategy] = None,
        sample_settings: Optional[SampleSettings] = None,
        custom_background: Optional[Color] = None,
        enable_scene_background: bool = False
    ):
        super().__init__()
        self.sampling_manager = sampling_manager
        self.sample_settings = sample_settings or SampleSettings()

        self.ray_generator: Generation.RayGenerationStrategy = ray_generator if ray_generator is not None else Generation.RayGenerator()
        self.intersector: Intersections.IntersectionStrategy = intersection_strategy if intersection_strategy is not None else Intersections.RayMarchingIntersection()
        self.shader: Shading.ShadingStrategy = shading_strategy if shading_strategy is not None else Shading.LambertShading()
        self.interactor: Interactions.InteractionStrategy = interaction_strategy if interaction_strategy is not None else Interactions.StandardInteraction()
        
        self.max_recursions = max_recursions

        self.stats: TracingStats = TracingStats()

    def _trace_ray(self, scene: Scene, ray: TracingRay, recursions_left: int, sampler: Sampler) -> Color:
        # 1. Base Case
        if recursions_left < 0:
            return Color(-1.0, 0.0, 0.0) # depth error or background color

        # 2. Geometry Intersection
        hit_info = self.intersector.find_hit(scene, ray, self.stats)
        
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

        # 4. Generate the Secondary Ray (e.g., Diffuse bounce, Refraction)
        # Note: Ensure your interact method returns the 'throughput' or 'attenuation' 
        # (how much color creates the bounce) along with the ray.
        interaction_result = self.interactor.interact(ray, hit_info, sampler, stats=self.stats)
        
        # Check if a ray was actually generated (it might be absorbed)
        if interaction_result and interaction_result.new_ray:
            new_ray = interaction_result.new_ray
            attenuation = interaction_result.attenuation # You likely need this value!

            # --- THE RECURSIVE STEP ---
            incoming_light = self._trace_ray(
                scene=scene,
                ray=new_ray,
                recursions_left=recursions_left - 1,
                sampler=sampler
            )

            # 5. Combine Direct and Indirect Light
            # Standard rendering equation: Out = Emitted + (Incoming * BRDF * cos_theta)
            # Assuming 'attenuation' includes the BRDF * cos_theta part:
            return shaded_color + (incoming_light * attenuation)

        return shaded_color

    def render(
        self,
        scene: Scene,
        sampler: Optional[Sampler] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        tile_size: Optional[int] = None,
    ) -> List[Color]:
        self.stats.reset_ray_counter()
        self.stats.start_timer()

        camera = scene.camera
        if camera is None: raise ValueError("No camera provided in Scene")
        cam_width, cam_height = camera.width, camera.height

        if region:
            rx, ry, rw, rh = region
        else:
            rx, ry, rw, rh = 0, 0, cam_width, cam_height

        # Initialize output image and sample counter
        full_image_pixels = [Color(0.0, 0.0, 0.0) for _ in range(rw * rh)]
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
                
                # 1. Generate Rays for this Tile ONLY
                rays = self.ray_generator.generate(camera=camera, region=tile_region, sampler=sampler)
                self.stats.rays_primary += len(rays)

                # 2. Local Storage for this Tile
                # Map: (local_tile_index) -> List[(Sample, Color)]
                tile_samples = [[] for _ in range(current_w * current_h)]


                self.stats = update_memory_stats(self.stats)

                # 3. Trace Rays
                for ray in rays:
                    if ray is None: continue
                    
                    # Global Coordinates
                    px, py = ray.pixel_x, ray.pixel_y
                    
                    # Convert to Local Tile Coordinates
                    local_x = px - tile_x
                    local_y = py - tile_y

                    # Safety Check
                    if not (0 <= local_x < current_w and 0 <= local_y < current_h):
                        continue

                    # Trace
                    pixel_color = self._trace_ray(scene, ray, self.max_depth, sampler)

                    # Create Sample Object
                    s_u = getattr(ray, "sample_u", (px + 0.5) / cam_width)
                    s_v = getattr(ray, "sample_v", (py + 0.5) / cam_height)
                    sample = Sample(s_u, s_v, 1.0) # weight 1.0

                    # Store in Local Tile Buffer
                    local_idx = local_y * current_w + local_x
                    tile_samples[local_idx].append((sample, pixel_color))
                    pixels_processed += 1

                # 4. Reconstruct Tile (Resolve samples to final color)
                for i in range(len(tile_samples)):
                    samples_and_colors = tile_samples[i]
                    
                    if not samples_and_colors: continue
                    # Calculate Global Pixel Index
                    local_y_in_tile = i // current_w
                    local_x_in_tile = i % current_w
                        
                    global_x = tile_x + local_x_in_tile
                    global_y = tile_y + local_y_in_tile
                        
                    # Reconstruct
                    samples = [sc[0] for sc in samples_and_colors]
                    # Convert Color objects to RGBA arrays
                    colors = []
                    for sc in samples_and_colors:
                        color_obj: Color = sc[1]
                        color_array = color_obj.to_np_array(include_alpha=True)
                        colors.append(color_array)
                    
                    rec_rgb = reconstruct_pixel(global_x, global_y, samples, colors, self.sample_settings)
                    final_color = Color(*rec_rgb)

                    # Write to Final Image Buffer
                    # Calculate index in the *region* buffer
                    final_idx = (global_y - ry) * rw + (global_x - rx)
                    full_image_pixels[final_idx] = final_color

        self.stats.pixels_processed = pixels_processed
        self.stats.stop_timer()

        return full_image_pixels
