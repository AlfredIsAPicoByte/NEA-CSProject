import numpy as np
from typing import Optional, Dict, List, Tuple, cast
from dataclasses import dataclass, field

from src.Geometry.SDF import SignedDistanceShape
from src.Geometry.Mesh import Mesh
from src.Material.Core import PBRMaterial
from src.Lighting.Core import Light

class ContextBase:
    """
    Base class for different types of scene contexts (e.g., SDF, Mesh, Light).
    """
    ...

@dataclass
class SDFContext(ContextBase):
    """
    Contains a signed distance shape for the scene.
    """
    shape: SignedDistanceShape = field(default_factory=SignedDistanceShape)
    material: Optional[PBRMaterial] = None
    
    def __post_init__(self):
        if not isinstance(self.shape, SignedDistanceShape):
            raise TypeError("shape must be an instance of SignedDistanceShape")
    
    def get_bounds(self, world_matrix: np.ndarray, padding: float = 1e-2):
        return self.shape.get_transformed_aabb(world_matrix, padding)

@dataclass
class MeshContext(ContextBase):
    """
    Contains a mesh for the scene.
    """
    mesh: Mesh = field(default_factory=Mesh)
    material: Optional[PBRMaterial] = None

    def __post_init__(self):
        if not isinstance(self.mesh, Mesh):
            raise TypeError("mesh must be an instance of Mesh")

    def get_bounds(self, world_matrix: np.ndarray, padding: float = 1e-2):
        return self.mesh.get_transformed_aabb(world_matrix, padding)

@dataclass
class LightContext(ContextBase):
    """
    Contains a light source for the scene.
    """
    light: Light = field(default_factory=Light)

    def __post_init__(self):
        if not isinstance(self.light, Light):
            raise TypeError("light must be an instance of Light")