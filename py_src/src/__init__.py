"""
src package initializer.

Imports available submodules and registers top-level aliases.
Exposes key classes (Color, Ray, Transform) to the package level.
"""
# Explicit submodule imports
from . import Data
from . import Geometry
from . import Lighting 
from . import Material
from . import Image
from . import Rendering
from . import Utilities

from .Data import Transform, Ray, Color

__all__ = [
    # Modules
    'Data',
    'Geometry',
    'Lighting',
    'Material',
    'Image',
    'Rendering',
    'Utilities',

    # Key classes
    'Transform',
    'Ray',
    'Color',
]