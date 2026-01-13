from math import cos, sin, acos, gcd
import numpy as np
from typing import Optional, Any, List
from dataclasses import dataclass, field

from src.Utilities.Common import unit

@dataclass(slots=True)
class HitInfo:
    """
    Stores information about a ray-object intersection.
    """
    # --- 1. Define ALL storage fields here ---
    
    # Did we hit something?
    hit: bool = False
    
    # Distance along the ray (for Z-buffer/sorting)
    distance: float = float('inf')
    
    # World-space coordinate of intersection
    point: Optional[np.ndarray] = None
    
    # Surface normal at intersection
    normal: Optional[np.ndarray] = None
    
    # The incoming ray direction (useful for shading calculations)
    direction: Optional[np.ndarray] = None
    
    # The object we hit (for material lookup)
    obj: Optional[Any] = None
    
    # Texture coordinates
    uv: Optional[np.ndarray] = None

    # --- 2. Custom Init to handle your specific naming logic ---
    def __init__(
        self,
        did_hit: bool,
        distance: float = float('inf'),
        point: Optional[np.ndarray] = None,
        direction: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
        obj: Optional[Any] = None,
        uv: Optional[np.ndarray] = None
    ):
        object.__setattr__(self, 'hit', bool(did_hit))
        object.__setattr__(self, 'point', point)
        object.__setattr__(self, 'distance', distance)
        object.__setattr__(self, 'obj', obj)
        object.__setattr__(self, 'uv', uv)

        if direction is not None:
            norm_dir = unit(direction)
            object.__setattr__(self, 'direction', norm_dir)
        else:
            object.__setattr__(self, 'direction', None)

        if normal is not None:
            norm_surf = unit(normal)
            object.__setattr__(self, 'normal', norm_surf)
        else:
            object.__setattr__(self, 'normal', None)

    @classmethod
    def miss(cls):
        """Fast helper to create a Miss."""
        # Uses the defaults defined in init arguments
        return cls(did_hit=False)
