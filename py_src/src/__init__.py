"""
src package initializer.

Imports available submodules and registers top-level aliases to maintain
compatibility with modules that use bare imports like `import Luminance`.
"""

import sys
from importlib import import_module

_submodules = [
    "PrimaryStructures",
    "Geometry",
    "Luminance",
    "Camera",
    "Scene",
    "Algorithims",
    "Sampling",
    "Reflections",
    "Refractions",
    "Raytracing",
]

__all__ = []

for name in _submodules:
    full = f"src.{name}"
    try:
        module = import_module(full)
        globals()[name] = module
        __all__.append(name)
        # (register only if not already present)
        if name not in sys.modules:
            sys.modules[name] = module
    
    except Exception:
        # Defer import errors to callers; keep package import robust
        pass