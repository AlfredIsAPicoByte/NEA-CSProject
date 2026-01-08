from math import cos, sin, acos, gcd
import numpy as np
from typing import Optional, Any, List
from dataclasses import dataclass, field

from CommonUtils import unit

def unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

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
            object.__setattr__(self, 'orientation', unit_vector(self.orientation))

    # alias 'direction' to the same data (keeps compatibility)
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
    depth: int = 0
    pixel_x: int = -1
    pixel_y: int = -1
    
    # How much light this ray carries (Color multiplier)
    # Storing as object to avoid import cycles with 'Color' class
    throughput: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    pdf: float = 0
    
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
            ray.depth = 0
            ray.is_inside = False
            ray.throughput = np.array([1.0, 1.0, 1.0])
            return ray
        else:
            # Create new if pool is empty
            return TracingRay(origin, orientation, x, y)

    def return_ray(self, ray: TracingRay):
        self._pool.append(ray)

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

class Transform:
    """
    Represents a 3D transformation with position, rotation, scale, and hierarchical parent support.

    Modes:
      - "global": methods update the public position/rotation/scale (default, keeps simple test usage)
      - "local" : methods update the local_* offsets which are applied on top of the public transform
    """
    def __init__(self, position: np.ndarray, rotation: np.ndarray, scale: np.ndarray, parent=None, name: str='transform'):
        if position.shape != (3,) or rotation.shape != (3,) or scale.shape != (3,):
            raise ValueError("position, rotation, and scale must be 3D vectors")
            
        # public/base transform (treated as the "global/base" transform)
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)  # Euler angles in radians
        self.scale = np.asarray(scale, dtype=float)

        # explicit local offsets applied on top of the base transform
        self.local_position = np.zeros(3, dtype=float)
        self.local_rotation = np.zeros(3, dtype=float)
        self.local_scale = np.ones(3, dtype=float)

        self.parent = parent
        self.name = name

        self.update_orientations()

    @classmethod
    def identity(cls):
        return cls(
            position=np.zeros(3),
            rotation=np.zeros(3),
            scale=np.ones(3),
        )

    def _matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Converts a 3x3 rotation matrix to ZYX Euler angles."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def update_orientations(self):
        """Updates forward/right/up vectors based on current rotation."""
        rx, ry, rz = self.rotation + self.local_rotation
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        
        # Rotation Matrix (Z * Y * X)
        R = np.array([
            [cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
            [cy*sz, cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz],
            [-sy,   cy*sx,            cx*cy]
        ])
        
        self.right   = R @ np.array([1, 0, 0])
        self.up      = R @ np.array([0, 1, 0])
        self.forward = R @ np.array([0, 0, 1])
    
    def _rotation_matrix_from_euler(self, euler: np.ndarray) -> np.ndarray:
        rx, ry, rz = euler
        rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        rot_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        rot_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        return rot_z @ rot_y @ rot_x

    def _make_transform_matrix(self, position: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
        R = self._rotation_matrix_from_euler(rotation)
        S = np.diag(scale)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = R @ S
        mat[:3, 3] = position
        return mat

    def get_global_matrix(self) -> np.ndarray:
        """Returns the global transformation matrix: parent @ base @ local."""
        base = self._make_transform_matrix(self.position, self.rotation, self.scale)
        local = self._make_transform_matrix(self.local_position, self.local_rotation, self.local_scale)
        combined = base @ local 
        
        if self.parent is not None:
            return self.parent.get_global_matrix() @ combined
        return combined
    
    def get_global_position(self) -> np.ndarray:
        return self.get_global_matrix()[:3, 3]
    
    def look_at(self, target_position: np.ndarray, world_up: np.ndarray = np.array([0.0, 1.0, 0.0])):
        if world_up is None: world_up = np.array([0, 1, 0])
        target_position = np.asarray(target_position, dtype=float)
        
        direction = target_position - self.get_global_position()
        dist = np.linalg.norm(direction)
        if dist < 1e-6: return

        forward = direction / dist
        if np.abs(np.dot(forward, world_up)) > 0.999:
            right = np.array([1, 0, 0])
        else:
            right = np.cross(world_up, forward)
            right = unit(right)
        
        up = np.cross(forward, right)
        
        # Matrix to Euler
        rotation_matrix = np.array([
            [right[0], up[0], forward[0]],
            [right[1], up[1], forward[1]],
            [right[2], up[2], forward[2]]
        ])
        
        new_euler = self._matrix_to_euler(rotation_matrix)
        self.rotation = new_euler - self.local_rotation
        self.update_orientations()

    def translate(self, vector: np.ndarray, space: str = "global"):
        if space == "global": self.position += vector
        else: self.local_position += vector
        self.update_orientations()

    def rotate(self, angle: float, axis: np.ndarray, space: str = "global"):
        axis = unit(np.asarray(axis, dtype=float))
        if np.linalg.norm(axis) == 0: return
        delta = axis * angle
        if space == "global": self.rotation += delta
        else: self.local_rotation += delta
        self.update_orientations()

    def enlarge(self, vector: np.ndarray, space: str = "global"):
        if space == "global": self.scale *= vector
        else: self.local_scale *= vector
        self.update_orientations()

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Applies the full transformation (Scale -> Rotation -> Translation) to a 3D point.
        Useful for converting object-space vertices to world-space.
        """
        # P_world = M * P_local
        # Homogeneous coordinate w=1 is implied for translation
        point = np.asarray(point, dtype=float)
        model_matrix = self.get_global_matrix()
        
        # Expand point to 4D homogeneous coordinates [x, y, z, 1]
        p_hom = np.append(point, 1.0)
        transformed = model_matrix @ p_hom
        
        return transformed[:3] # Return XYZ

    def transform_direction(self, direction: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Applies only Rotation and Scale to a vector (Translation is ignored).
        Useful for ray directions or velocity vectors.
        """
        direction = np.asarray(direction, dtype=float)
        model_matrix = self.get_global_matrix()
        
        # Extract upper-left 3x3 (Rotation & Scale)
        rs_matrix = model_matrix[:3, :3]
        
        transformed = rs_matrix @ direction
        
        if normalize:
            return unit(transformed)
        return transformed

    def transform_normal(self, normal: np.ndarray) -> np.ndarray:
        """
        Transforms a surface normal vector.
        Uses the Inverse Transpose of the upper 3x3 matrix to handle non-uniform scaling correctly,
        ensuring the normal remains perpendicular to the surface.
        """
        normal = np.asarray(normal, dtype=float)
        model_matrix = self.get_global_matrix()
        
        # 1. Extract Rotation-Scale matrix (3x3)
        rs_matrix = model_matrix[:3, :3]
        
        # 2. Compute Normal Matrix: (M^-1)^T
        # This is required because normals transform differently than points/lines under non-uniform scaling.
        try:
            norm_matrix = np.linalg.inv(rs_matrix).T
        except np.linalg.LinAlgError:
            # Fallback if scale is 0
            norm_matrix = rs_matrix 

        transformed = norm_matrix @ normal
        
        # Normals should almost always be normalized after transformation
        return unit(transformed)
    
    def transform_ray(self, ray: Ray, normalize: bool = False) -> Ray:
        """
        Transforms a ray into the local space.
        """

        local_origin = self.transform_point(ray.origin)
        local_orientation = unit(self.transform_direction(ray.orientation, normalize))

        return Ray(local_origin, local_orientation)
    
    def get_inverse_matrix(self) -> np.ndarray:
        """Returns the inverse of the global transform matrix (World -> Local)."""
        # Note: For high-performance rendering engines, this is usually cached.
        try:
            return np.linalg.inv(self.get_global_matrix())
        except np.linalg.LinAlgError:
            # Handle singular matrix (e.g., scale is 0) gracefully
            return np.eye(4)

    def inverse_transform_point(self, world_point: np.ndarray) -> np.ndarray:
        """
        Transforms a point from World Space back to Local Object Space.
        """
        world_point = np.asarray(world_point, dtype=float)
        inv_matrix = self.get_inverse_matrix()
        
        # Homogeneous coordinate w=1
        p_hom = np.append(world_point, 1.0)
        transformed = inv_matrix @ p_hom
        
        return transformed[:3]

    def inverse_transform_direction(self, world_direction: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Transforms a direction vector from World Space to Local Space.
        (Rotation and Scale only, ignores Translation).
        """
        world_direction = np.asarray(world_direction, dtype=float)
        inv_matrix = self.get_inverse_matrix()
        
        # Extract upper-left 3x3 (Inverse Rotation & Inverse Scale)
        inv_rs_matrix = inv_matrix[:3, :3]
        
        transformed = inv_rs_matrix @ world_direction
        
        if normalize:
            return unit(transformed)
        return transformed

    def inverse_transform_normal(self, world_normal: np.ndarray) -> np.ndarray:
        """
        Transforms a normal vector from World Space to Local Space.
        
        Math Note:
        Forward Normal = (M^-1).T @ local_normal
        Inverse Normal = (Inverse_Matrix^-1).T @ world_normal
                       = ((M^-1)^-1).T @ world_normal
                       = (M).T @ world_normal
        
        We use the Transpose of the Forward Model Matrix.
        """
        world_normal = np.asarray(world_normal, dtype=float)
        
        # Get the Forward Model Matrix (not inverse)
        model_matrix = self.get_global_matrix()
        
        # Extract 3x3 and Transpose it
        # This handles the non-uniform scaling 'un-skewing' correctly.
        transpose_matrix = model_matrix[:3, :3].T
        
        transformed = transpose_matrix @ world_normal
        
        return unit(transformed)
    
    def inverse_transform_ray(self, ray: Ray, normalize: bool = False) -> Ray:
        """
        Transforms a ray into the local space.
        """

        local_origin = self.inverse_transform_point(ray.origin)
        local_orientation = unit(self.inverse_transform_direction(ray.orientation, normalize))

        return Ray(local_origin, local_orientation)

    def reflect_axis(self, axis: np.ndarray, space: str = "global"):
        """
        Reflect the transform across the given axis (2D only).
        space: "global" reflects the public/base scale, "local" reflects the local_scale.
        """
        ax = np.asarray(axis, dtype=float)
        if ax.shape != (2,):
            raise ValueError("Reflection axis must be 2D")
        norm = np.linalg.norm(ax)
        if norm == 0:
            raise ValueError("Cannot reflect across a zero-length axis")
        
        # Extend axis to 3D for compatibility with scale
        axis3d = np.zeros(3, dtype=float)
        axis3d[:2] = ax / norm
        reflection = np.sign(axis3d)
        if space == "global":
            self.scale = self.scale * reflection
        else:
            self.local_scale = self.local_scale * reflection
        self.update_orientations()

    def copy_transform(self, other: 'Transform') -> 'Transform':
        """Combines this transform with another, returning a new Transform."""
        new_position = self.position + other.position
        new_rotation = self.rotation + other.rotation
        new_scale = self.scale * other.scale
        
        new_local_position = self.local_position + other.local_position
        new_local_rotation = self.local_rotation + other.local_rotation
        new_local_scale = self.local_scale * other.local_scale
        
        combined = Transform(new_position, new_rotation, new_scale)
        combined.local_position = new_local_position
        combined.local_rotation = new_local_rotation
        combined.local_scale = new_local_scale
        combined.update_orientations()
        return combined
    
    def __repr__(self):
        return (f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale}, "
                f"local_position={self.local_position}, local_rotation={self.local_rotation}, local_scale={self.local_scale})")

class Ratio:
    """
    Represents a ratio (fraction) with a width and height.
    Supports basic arithmetic operations and comparisons.
    """
    def __init__(self, width: float, height: float):
        """
        Creates a ratio (fraction) with a width and height.
        """
        if width == 0:
            raise ValueError("Width (denominator) cannot be zero")
        
        if height == 0:
            raise ValueError("Height (numerator) cannot be zero")
            
        self.width = width
        self.height = height

    def simplify(self):
        """Simplify this Ratio in-place and return self (e.g., 1920/1080 -> 16/9)."""
        w_int = int(self.width)
        h_int = int(self.height)

        divisor = gcd(w_int, h_int)
        if divisor == 0:
            return self

        # Use integer division to keep them integral
        self.width = w_int // divisor
        self.height = h_int // divisor
        return self

    def __add__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        
        new_height = self.height * other.width + other.height * self.width
        new_width = self.width * other.width
        return Ratio(new_width, new_height)
    
    def __sub__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
            
        new_height = self.height * other.width - other.height * self.width
        new_width = self.width * other.width
        return Ratio(new_width, new_height)

    def __mul__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
            
        new_height = self.height * other.height
        new_width = self.width * other.width
        return Ratio(new_width, new_height)
    
    def __truediv__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        if other.height == 0:
            raise ValueError("Cannot divide by a Ratio with a height of zero")
            
        new_height = self.height * other.width
        new_width = self.width * other.height
        return Ratio(new_width, new_height)
    
    def __repr__(self):
        return f"Ratio({self.height}/{self.width})"
    
    def __float__(self):
        return self.height / self.width
    
    def __neg__(self):
        return Ratio(-self.width, -self.height)
    
    def __eq__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width == self.width * other.height
    
    def __lt__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width < self.width * other.height
        
    def __le__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width <= self.width * other.height
        
    def __gt__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width > self.width * other.height
        
    def __ge__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width >= self.width * other.height
        
    def __ne__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.height * other.width != self.width * other.height
    
    @property
    def value(self):
        """Returns the float value of the ratio."""
        return self.width / self.height
    
"""
PrimaryStructures module: Provides datastructures essential for most graphical computations.

Classes:
- Ray: Represents a ray in 2D/3D space with origin and orientation.
- HitInfo: Stores information about ray-object intersections.
- Transform: Represents a 3D transformation with position, rotation, scale, and hierarchical parent support.
- Ratio: Represents a ratio (fraction) with width and height, supporting basic arithmetic operations and comparisons.
"""