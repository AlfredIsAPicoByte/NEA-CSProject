from math import cos, sin, acos
import numpy as np
from typing import List
from dataclasses import dataclass, field

from src.Utilities.Common import unit

@dataclass(slots=True)
class Ray:
    origin: np.ndarray
    orientation: np.ndarray
    name: str = "Ray"

    def __post_init__(self):
        """
        Dataclasses run this AFTER the auto-generated __init__.
        We use this to enforce normalization logic.
        """
        # Safety check for zero-length vector
        if np.linalg.norm(self.orientation) == 0:
            # Fallback to a safe default (e.g., Up) to prevent crash
            object.__setattr__(self, 'orientation', np.array([0.0, 1.0, 0.0]))
        else:
            # Normalize and re-assign (bypass frozen check if frozen=True, though unnecessary here)
            object.__setattr__(self, 'orientation', unit(self.orientation))

    @property
    def direction(self) -> np.ndarray:
        return self.orientation
    @direction.setter
    def direction(self, v):
        self.orientation = v

    def point_at(self, t: float):
        """Returns the point along the ray at parameter t."""
        return self.origin + self.orientation  * t

    def check_point_on_ray(self, point: np.ndarray):
        """Checks if a given point lies on the ray."""
        if point.shape != self.origin.shape:
            raise ValueError("Point and ray origin must be of the same dimension")
        
        to_point = point - self.origin
        if np.linalg.norm(to_point) == 0:
            return True
        
        to_point_normalized = unit(to_point)
        return np.allclose(to_point_normalized, self.orientation)

    def check_point_in_front(self, point: np.ndarray):
        """Checks if a given point is in front of the ray's origin along its orientation ."""
        if point.shape != self.origin.shape:
            raise ValueError("Point and ray origin must be of the same dimension")
        
        to_point = point - self.origin
        return np.dot(self.orientation , to_point) > 0

    def check_point_behind(self, point: np.ndarray):
        """Checks if a given point is behind the ray's origin opposite its orientation ."""
        if point.shape != self.origin.shape:
            raise ValueError("Point and ray origin must be of the same dimension")
        
        to_point = point - self.origin
        return np.dot(self.orientation , to_point) < 0

    def rotate(self, angle: float, axis: np.ndarray):
        """
        Rotates the ray's orientation  vector around the given axis by the specified angle (in radians).
        Only works for 3D rays.
        """
        if self.orientation .shape[0] != 3 or axis.shape[0] != 3:
            raise ValueError("Rotate only supports 3D vectors")
        
        axis = unit(axis)
        cos_a = cos(angle)
        sin_a = sin(angle)
        ux, uy, uz = axis
        
        # Rodrigues' rotation formula
        R = np.array([
            [cos_a + ux**2 * (1 - cos_a), ux*uy*(1 - cos_a) - uz*sin_a, ux*uz*(1 - cos_a) + uy*sin_a],
            [uy*ux*(1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a), uy*uz*(1 - cos_a) - ux*sin_a],
            [uz*ux*(1 - cos_a) - uy*sin_a, uz*uy*(1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
        ])
        
        self.orientation = unit(R @ self.orientation)

    def translate(self, vector: np.ndarray):
        """Translates the ray's origin by the given vector."""
        if vector.shape != self.origin.shape:
            raise ValueError("Translation vector must match the ray's origin dimension")
        self.origin += vector

    def get_angle(self, line: np.ndarray):
        """Retruns the angles created from another vector"""
        if line.shape != self.orientation.shape:
            raise ValueError("Input vector must match the ray's orientation  dimension")
        dot_product = np.dot(self.orientation , line) / (np.linalg.norm(self.orientation) * np.linalg.norm(line))
        angle = acos(dot_product)
        return angle

    def __repr__(self):
        return f"Ray(origin={self.origin}, orientation={self.orientation})"

# Define a ray that holds the ray and data
@dataclass
class TracingRay(Ray):
    """
    A Ray that carries extra state for the recursive path tracing engine.
    """
    current_depth: int = 0
    pixel_x: int = -1
    pixel_y: int = -1
    
    # How much light this ray carries (Color multiplier)
    # Storing as object to avoid import cycles with 'Color' class
    throughput: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    
    # Is the ray currently traveling inside a medium (like glass)?
    is_inside: bool = False
    
    # UV Coordinates for sub-pixel sampling
    sample_u: float = 0.5
    sample_v: float = 0.5

    def __repr__(self):
        return f"TracingRay(name={self.name}, origin={self.origin}, orientation={self.orientation})"

class RayPool:
    def __init__(self, block_size=10000):
        self._pool = []
        self._block_size = block_size

    def get_ray(self, origin, orientation, x, y):
        if self._pool:
            ray = self._pool.pop()
            
            ray.origin = origin
            ray.orientation = orientation
            ray.pixel_x = x
            ray.pixel_y = y
            ray.current_depth = 0
            ray.is_inside = False
            ray.throughput = np.array([1.0, 1.0, 1.0, 1.0])
            return ray
        else:
            # Create new if pool is empty
            return TracingRay(origin, orientation, x, y)

    def return_ray(self, ray: TracingRay):
        self._pool.append(ray)