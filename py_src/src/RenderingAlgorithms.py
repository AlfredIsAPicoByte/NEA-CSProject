from __future__ import annotations
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Type, Any, Tuple, List

from Scene import Scene
from Sampling import Sampler
from Luminance import Color
from MemoryUtils import get_process_id, get_memory_mb

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
        ) -> List[Color]:
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