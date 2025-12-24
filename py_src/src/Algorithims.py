from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Type, Any, List, Tuple

from Luminance import Color
from Scene import Scene

@dataclass
class RenderStats:
    time_taken_seconds: float = 0.0
    memory_usage: float = 0.0  # in MB

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
            seed: Optional[int] = None,
            region: Optional[Tuple[int,int,int,int]] = None,
            tile_size: Optional[Tuple[int,int]] = None
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