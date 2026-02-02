import numpy as np
from typing import Any
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

    def local_corners(self, padding: float = 1e-2) -> np.ndarray:
        """Local-space bounding corners."""
        return self.shape.get_local_corners(padding)
    
    @property
    def is_2d(self) -> bool:
        return self.shape.dimension == 2

    def world_corners(self, padding: float = 1e-2) -> np.ndarray:
        """World-space bounding corners."""
        return transform_corners(self.local_corners(padding))

    def local_bounds(self, padding: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
        """Local-space (min, max) bounds."""
        c = self.local_corners(padding)
        if self.is_2d:
            c = convert_bounds_to_corners_2d(c)
        return convert_bounds_to_corners(c)

    def world_bounds(self, padding: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
        """World-space (min, max) bounds."""
        c = self.world_corners(padding)
        return c.min(axis=0), c.max(axis=0)

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
        return self.mesh.vertices

    def triangles(self) -> np.ndarray:
        return self.mesh.faces

    # ──────────────────────────────
    # Bounds & corners
    # ──────────────────────────────

    def local_corners(self) -> np.ndarray:
        min_pt, max_pt = self.mesh.get_local_bounds()
        return convert_bounds_to_corners(min_pt, max_pt)

    def world_corners(self) -> np.ndarray:
        return transform_corners(self.local_corners())

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        min_pt = self.mesh.vertices.min(axis=0)
        max_pt = self.mesh.vertices.max(axis=0)
        return min_pt, max_pt

    def world_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.world_corners()
        return c.min(axis=0), c.max(axis=0)