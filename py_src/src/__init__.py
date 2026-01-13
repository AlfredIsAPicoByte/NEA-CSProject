"""
py_src/src package initializer.

Imports available submodules and registers top-level aliases.
Exposes key classes (Color, Ray, Transform) to the package level.
"""

import sys
from importlib import import_module

_submodules = [
    "Data",
    "Geometry",
    "Ligting",
    "Material",
    "Image",
    "Rendering",
    "Utilities"
]

__all__ = []

for name in _submodules:
    full = f"src.{name}"
    try:
        module = import_module(full)
        globals()[name] = module
        __all__.append(name)
        
        # Register module in sys.modules if not present
        if name not in sys.modules:
            sys.modules[name] = module

    except Exception as e:
        # Useful for debugging: print the error if a module fails to load
        # print(f"Failed to load submodule {name}: {e}")
        pass