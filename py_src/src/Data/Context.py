import numpy as np
from typing import Any, Optional
from dataclasses import dataclass, field

from src.Geometry.SDF import SignedDistanceShape
from src.Geometry.AABB import convert_bounds_to_corners_2d, convert_bounds_to_corners, transform_bounds, transform_corners
from src.Geometry.Mesh import Mesh
from src.Material.Core import PBRMaterial

"""
Define what comon scene node that will be used.
"""

@dataclass
class SDF_Material:
    """
    Contains a signed distance shape for the scene.
    """
    shape: SignedDistanceShape = field(default_factory=SignedDistanceShape)
    material: PBRMaterial = field(default_factory=PBRMaterial)
    
    def __post_init__(self):
        if not isinstance(self.shape, SignedDistanceShape):
            raise TypeError("shape must be an instance of SignedDistanceShape")
        if not isinstance(self.material, PBRMaterial):
            raise TypeError("material must be an instance of Material")
        
        # ──────────────────────────────
    # Geometry access
    # ──────────────────────────────

    def signed_distance(self, point: np.ndarray) -> float:
        """Evaluate signed distance at a world-space point."""
        return self.shape.get_distance(point)

    def normal(self, point: np.ndarray) -> np.ndarray:
        """Surface normal at a world-space point."""
        return self.shape.get_normal(point)

    # ──────────────────────────────
    # Bounds & corners
    # ──────────────────────────────

    def local_corners(self, padding: float = 1e-2) -> Optional[np.ndarray]:
        """Local-space bounding corners."""
        if hasattr(self.shape, 'get_local_corners'):
            return self.shape.get_local_corners(padding)
        
        if callable(getattr(self.shape, 'get_convex_hull')):
            hull = self.shape.get_convex_hull()
            if hull is None or len(hull) == 0:
                return None
            
            min_pt = hull.min(axis=0) - padding
            max_pt = hull.max(axis=0) + padding
            return convert_bounds_to_corners(min_pt, max_pt)
        
        else:
            return None
    
    @property
    def is_2d(self) -> bool:
        return self.shape.dimension == 2

@dataclass
class Mesh_Material:
    """
    Contains a mesh for the scene.
    """
    mesh: Mesh = field(default_factory=Mesh)
    material: PBRMaterial = field(default_factory=PBRMaterial)

    def __post_init__(self):
        if not isinstance(self.mesh, Mesh):
            raise TypeError("mesh must be an instance of Mesh")
        if not isinstance(self.material, PBRMaterial):
            raise TypeError("material must be an instance of Material")
    
    # ──────────────────────────────
    # Geometry access
    # ──────────────────────────────

    def vertices(self) -> np.ndarray:
        return self.mesh.__getattribute__("vertices")

    def triangles(self) -> np.ndarray:
        return self.mesh.__getattribute__("faces")

    # ──────────────────────────────
    # Bounds & corners
    # ──────────────────────────────

    def local_corners(self) -> np.ndarray:
        min_pt, max_pt = self.mesh.get_local_bounds()
        return convert_bounds_to_corners(min_pt, max_pt)