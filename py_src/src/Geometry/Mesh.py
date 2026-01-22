from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

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