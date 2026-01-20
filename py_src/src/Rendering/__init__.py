from typing import Dict, Type

from .Core import Algorithm, RenderStats

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

__all__ = [
    'Algorithm',
    'RenderStats',
    'register_algorithm',
    'create_algorithm'
]

from .RayTracing.Core import RayTracer, RayTracingSettings
