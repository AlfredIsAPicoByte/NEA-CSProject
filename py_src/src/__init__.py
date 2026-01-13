"""
py_src/src package initializer.

Imports available submodules and registers top-level aliases.
Exposes key classes (Color, Ray, Transform) to the package level.
"""

import sys
from importlib import import_module

_submodules = [
    "PrimaryStructures",
    "Geometry",
    "Reflections",
    "Refractions",
    "Luminance",
    "Camera",
    "Scene",
    "Sampling",
    "Generation",
    "Intersection",
    "Shading",
    "Interaction",
    "RenderingAlgorithms",
    "Raytracing",
    "MemoryUtils",
    "MemoryProfiler"
    "PostProcessing",
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

        # --- EXPOSE MAIN CLASSES ---
        
        # 1. Expose Ray from PrimaryStructures
        if name == "PrimaryStructures":
            Ray = getattr(module, "Ray", None)
            if Ray:
                globals()["Ray"] = Ray
                __all__.append("Ray")

        # 2. Expose Color from Luminance (+ Builtin Hack)
        if name == "Luminance":
            Color = getattr(module, "Color", None)
            if Color:
                globals()["Color"] = Color
                __all__.append("Color")
                
                # Legacy support: inject into builtins
                try:
                    import builtins
                    builtins.Color = Color
                except Exception:
                    pass

    except Exception as e:
        # Useful for debugging: print the error if a module fails to load
        # print(f"Failed to load submodule {name}: {e}")
        pass