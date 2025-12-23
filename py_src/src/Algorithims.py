from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Type, Any, List, Tuple

from Luminance import Color
from Scene import Scene

@dataclass
class ParameterStats:
    image_width: int = 800
    image_height: int = 600
    output_file: str = "render.png"
    max_bounces: int = 5
    max_distance: float = 1000.0
    max_steps: int = 1000
    epsilon: float = 0.005 # Small offset to avoid self-intersection

    rays_traced: int = 0
    hits: int = 0
    misses: int = 0
    bounces: int = 0
    max_depth_reached: int = 0
    time_seconds: float = 0.0
    memory_usage: float = 0.0

class Algorithm(ABC):
    """
    Abstract base for rendering algorithms (ray marcher, path tracer, rasterizer, ...).
    Implementations should be side-effect free where possible and avoid global state.
    """
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.stats = ParameterStats()

    @abstractmethod
    def render(
            self,
            scene: Scene,
            seed: Optional[int] = None,
            tile_size: Optional[Tuple[int,int]] = None
        ) -> List[Color]:
        """
        High-level render entry point: produce an image/buffer from the scene and camera.
        """
        raise NotImplementedError

    def reset_stats(self) -> None:
        self.stats = ParameterStats()

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