from typing import Dict, Type, TypeVar

from .Core import Algorithm, AlgorithmSettings, RenderStats

T = TypeVar("T", bound=Algorithm)

_ALGO_REGISTRY: Dict[str, Type[Algorithm]] = {}

def register_algorithm(name: str):
    def _decorator(cls: Type[T]) -> Type[T]:
        _ALGO_REGISTRY[name] = cls
        return cls
    return _decorator

def create_algorithm(name: str, settings: AlgorithmSettings) -> Algorithm:
    """
    Instantiate a registered algorithm by name.
    Use this to avoid hard imports at call sites.
    """
    if name not in _ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. Registered: {list(_ALGO_REGISTRY.keys())}"
        )

    cls = _ALGO_REGISTRY[name]

    if hasattr(cls, "SettingsType") and not isinstance(settings, cls.SettingsType):
        raise TypeError(
            f"{name} expects settings of type {cls.SettingsType.__name__}, "
            f"got {type(settings).__name__}"
        )

    return cls(settings)

def list_algorithms() -> Dict[str, Type[Algorithm]]:
    return dict(_ALGO_REGISTRY)

__all__ = [
    'Algorithm',
    'AlgorithmSettings',
    'RenderStats',
    'register_algorithm',
    'create_algorithm',
    'list_algorithms',
]