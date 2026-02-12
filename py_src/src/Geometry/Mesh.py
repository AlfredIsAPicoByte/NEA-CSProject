import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


from .AABB import convert_bounds_to_corners

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
    bitangent: Optional[np.ndarray] = None

class Mesh:
    """Handles large 3D meshes with efficient storage and operations."""
    
    def __init__(self, name: str = "Mesh", vertices: Optional[List[Vertex]] = None, faces: Optional[List[Face]] = None):
        self.name = name
        self.vertices: List[Vertex] = vertices if vertices is not None else []
        self.faces: List[Face] = faces if faces is not None else []
        
        self._vertex_array: Optional[np.ndarray] = None  # Cached numpy array of vertices
        self._version: int = 0  # Versioning for change tracking

    def add_vertex(self, x: float, y: float, z: float) -> int:
        """Add a vertex and return its index."""
        self.vertices.append(Vertex(x, y, z))
        self._vertex_array = None  # Invalidate cache
        self.update_version()
        return len(self.vertices) - 1
    
    def add_face(self, v1: int, v2: int, v3: int) -> None:
        """Add a triangular face."""
        if not (0 <= v1 < len(self.vertices) and 
                0 <= v2 < len(self.vertices) and 
                0 <= v3 < len(self.vertices)):
            raise IndexError("Invalid vertex indices")
        
        self.faces.append(Face(v1, v2, v3))
        self.update_version()
    
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

    def update_version(self) -> None:
        """Increment the version to indicate a change."""
        self._version += 1

    def get_local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute local axis-aligned bounding box."""
        v_array = self.get_vertex_array()
        min_pt = v_array.min(axis=0)
        max_pt = v_array.max(axis=0)
        return (min_pt, max_pt)
    
    def get_local_corners(self) -> np.ndarray:
        """Get local bounding box corners."""
        bounds = self.get_local_bounds()
        return convert_bounds_to_corners(bounds[0], bounds[1])

    def __repr__(self) -> str:
        return f"Mesh(name={self.name} vertices={self.vertex_count()} faces={self.face_count()})"