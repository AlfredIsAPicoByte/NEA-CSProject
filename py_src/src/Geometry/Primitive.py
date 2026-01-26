from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, List, Tuple, cast
from dataclasses import dataclass, field

from src.Data.Transform import Transform
from .AABB import AABB
from .Core import SignedDistanceShape
from src.Material.Core import PBRMaterial

@dataclass
class Primitive:
    """
    A node in the scene graph. 
    Combines a logical shape, a material, and a position in 3D space.
    Can act as a parent to other Primitives.
    """
    name: str = "Object"
    transform: Transform = field(default_factory=Transform.identity)
    shape: Optional[SignedDistanceShape] = None
    material: Optional[PBRMaterial] = None

    # Hierarchy
    children: List['Primitive'] = field(default_factory=list)
    parent: Optional['Primitive'] = field(default=None, repr=False)
    
    # Caching / Optimization
    # We store the calculated world matrix here so we don't recalculate it for every ray
    _world_matrix: Optional[np.ndarray] = None
    _inverse_world_matrix: Optional[np.ndarray] = None
    _aabb_bounds: Optional[AABB] = None
    _cache_objects: Optional[List['Primitive']] = None

    def add_child(self, child: 'Primitive'):
        """Attaches a child node to this node."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: 'Primitive'):
        """Detaches a child node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def update_matrices(self, parent_matrix: Optional[np.ndarray] = None):
        """
        Recursive pass to update the world matrix of this node and all children.
        Call this ONCE before rendering starts.
        """
        # 1. Get Local Matrix
        local_mat = self.transform.to_matrix()

        # 2. Multiply by Parent (if exists)
        if parent_matrix is not None:
            self._world_matrix = parent_matrix @ local_mat
        else:
            self._world_matrix = local_mat
            
        # 3. Calculate Inverse (Needed for Ray Intersection: World -> Local)
        try:
            self._inverse_world_matrix = np.linalg.inv(self._world_matrix)
        except np.linalg.LinAlgError:
            self._inverse_world_matrix = np.eye(4)

        # 4. Propagate down the tree
        for child in self.children:
            child.update_matrices(self._world_matrix)

    def get_world_matrix(self) -> np.ndarray:
        if self._world_matrix is None:
            self.update_matrices()
        
        return self._world_matrix
    
    def get_world_inverse_matrix(self) -> np.ndarray:
        if self._inverse_world_matrix is None:
            self.update_matrices()
        
        return self._inverse_world_matrix

    @property
    def world_transform(self) -> Transform:
        """Returns a `Transform` representing the object's world transform (position/rotation/scale).
        Useful for APIs that expect a `Transform` object rather than raw matrices."""
        # Ensure matrices are up-to-date
        mat = self.get_world_matrix()
        return Transform.from_matrix(mat)

    def flatten_children(self, include_self: bool = True):
        """
        Returns a flat list of this object and all descendants.
        Useful for building the global list of objects for the BVH or Renderer.
        """
        result = []
        stack = [self] if include_self else []
        
        while stack:
            current = stack.pop()
            
            if current is not None:
                result.append(current)

                for child in reversed(current.children):
                    stack.append(child)
            else:
                for child in reversed(self.children):
                    stack.append(child)

        self._cache_objects = result

    def get_objects_flat(self, include_self: bool = True):
        if self._cache_objects is None:
            self.flatten_children(include_self)
        
        return self._cache_objects
    
    def generate_bounds(self, padding: float = 1e-2) -> None:
        self_bounds = getattr(self, "_aabb_bounds", AABB(np.zeros(3), np.zeros(3)))
        
        shape = getattr(self, "shape", None)
        if shape is None:
            self._aabb_bounds = self_bounds
        safe_shape = cast(Shape, shape)
        
        self._aabb_bounds = self_bounds.from_transform_shape(self.world_transform, safe_shape, padding)

    def get_bounds(self):
        if self._aabb_bounds is None:
            self.generate_bounds()
        
        return self._aabb_bounds

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other