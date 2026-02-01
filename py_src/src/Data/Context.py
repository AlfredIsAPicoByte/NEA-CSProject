import numpy as np
from typing import Any
from dataclasses import dataclass, field

from src.Geometry.SDF import SignedDistanceShape, CorrespondingBoundingBox
from src.Geometry.AABB import AABB
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
    