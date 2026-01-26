from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .AABB import AABB

"""
Mesh module for handling large 3D meshes efficiently.
"""

@dataclass
class Vertex:
    """Represents a single vertex in 3D space."""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        """Convert vertex to numpy array."""
        return np.array([self.x, self.y, self.z])

@dataclass
class Face:
    """Represents a face (triangle) in the mesh."""
    v1: int
    v2: int
    v3: int

    uv: Optional[Tuple[float, float]] = None
    normal: Optional[np.ndarray] = None
    tangent: Optional[np.ndarray] = None

class Mesh:
    """Handles large 3D meshes with efficient storage and operations."""
    
    def __init__(self, name: str = "Mesh"):
        """Initialize an empty mesh."""
        self.name = name
        self.vertices: List[Vertex] = []
        self.faces: List[Face] = []
        self._vertex_array: Optional[np.ndarray] = None
    
    def add_vertex(self, x: float, y: float, z: float) -> int:
        """Add a vertex and return its index."""
        self.vertices.append(Vertex(x, y, z))
        self._vertex_array = None  # Invalidate cache
        return len(self.vertices) - 1
    
    def add_face(self, v1: int, v2: int, v3: int) -> None:
        """Add a triangular face."""
        if not (0 <= v1 < len(self.vertices) and 
                0 <= v2 < len(self.vertices) and 
                0 <= v3 < len(self.vertices)):
            raise IndexError("Invalid vertex indices")
        self.faces.append(Face(v1, v2, v3))
    
    def get_vertex_array(self) -> np.ndarray:
        """Get cached numpy array of all vertices."""
        if self._vertex_array is None:
            self._vertex_array = np.array(
                [v.to_array() for v in self.vertices]
            )
        return self._vertex_array
    
    def vertex_count(self) -> int:
        """Return number of vertices."""
        return len(self.vertices)
    
    def face_count(self) -> int:
        """Return number of faces."""
        return len(self.faces)
    
    def clear(self) -> None:
        """Clear all mesh data."""
        self.vertices.clear()
        self.faces.clear()
        self._vertex_array = None

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        """
        Compute the axis-aligned bounding box (AABB) of the mesh after applying a transformation.
        
        :param transformation_matrix: A 4x4 transformation matrix as a numpy array.
        :param padding: A small padding value to expand the AABB.
        :return: An AABB instance representing the transformed bounding box.
        """
        if not self.vertices:
            return AABB.empty()
        
        vertex_array = self.get_vertex_array()
        
        for i in range(len(vertex_array)):
            vertex_array[i] += np.sign(vertex_array[i]) * padding

        transformed_vertices = AABB.transform_local_bounds(transformation_matrix, vertex_array)
        
        min_bounds = np.min(transformed_vertices, axis=0)
        max_bounds = np.max(transformed_vertices, axis=0)

        return AABB(min_bounds, max_bounds)

    def __repr__(self) -> str:
        return f"Mesh(name={self.name} vertices={self.vertex_count()} faces={self.face_count()})"