from __future__ import annotations
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Type, Any, Tuple

from src.Image.Film import Film
from src.Utilities.Scene import Scene
from src.Utilities.Sampling import Sampler
from src.Utilities.Memory.Core import get_process_id, get_memory_mb

@dataclass
class RenderStats:
    memory_usage: float = 0.0  # in MB
    
    # --- Timing (Internal use) ---
    time_taken_seconds: float = 0.0
    _start_time: float = field(default=0.0, repr=False)

    def start_timer(self):
        self._start_time = time.perf_counter()

    def stop_timer(self):
        self.time_taken_seconds = time.perf_counter() - self._start_time

    def __add__(self, other: "RenderStats") -> "RenderStats":
        new_stats = RenderStats()

        # Max/Avg specific fields
        new_stats.time_taken_seconds = max(self.time_taken_seconds, other.time_taken_seconds)
        new_stats.memory_usage = max(self.memory_usage, other.memory_usage)

        return new_stats

def update_memory_stats(stats: RenderStats) -> RenderStats:
    """
    Returns a NEW TracingStats object with updated memory usage,
    leaving the original object untouched (Immutability).
    """
    from dataclasses import replace
    
    current_mem = get_memory_mb(get_process_id())
    
    return replace(stats, memory_usage=current_mem)

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
        na_rate = (self.nan_errors / max(1, self.total_rays)) * 1000.0
        lines.append(f"  - NaN Rate:        {na_rate:.2f} per 1000 rays")
        
        return "\n".join(lines)
    
    def print_verbose_report(self):
        print()
        print(self.format_report())

class Algorithm(ABC):
    """
    Abstract base for rendering algorithms (ray marcher, path tracer, rasterizer, ...).
    Implementations should be side-effect free where possible and avoid global state.
    """
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
                
        self.stats = RenderStats()

    @abstractmethod
    def render(
            self,
            scene: Scene,
            sampler: Optional[Sampler] = None,
            region: Optional[Tuple[int, int, int, int]] = None,
            tile_size: Optional[int] = None
        ) -> Film:
        """
        High-level render entry point: produce an image/buffer from the scene and camera.
        """
        raise NotImplementedError

    def reset_stats(self) -> None:
        self.stats = RenderStats()

# Lightweight registry/factory for algorithm implementations
_ALGO_REGISTRY: Dict[str, Type[Algorithm]] = { }

def register_algorithm(name: str):
    def _decorator(cls: Type[Algorithm]):
        _ALGO_REGISTRY[name] = cls
        return cls
    return _decorator

def create_algorithm(name: str, **kwargs) -> Algorithm:
    """
    Instantiate a registered algorithm by name.
    Use this to avoid hard imports at call sites.
    """
    if name not in _ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}'. Registered: {list(_ALGO_REGISTRY.keys())}")
    return _ALGO_REGISTRY[name](**kwargs)