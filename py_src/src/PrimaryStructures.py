from math import cos, sin, acos
import numpy as np

class Ray:
    def __init__(self, origin: np.ndarray, orientation: np.ndarray, name: str = "Ray"):
        """
        A ray defined by an origin point and an orientation vector.
        """
        if np.linalg.norm(orientation) == 0:
            raise ValueError("Orientation vector cannot be zero-length")

        self.origin = np.asarray(origin, dtype=float)
        self._orientation = None
        self.orientation = orientation  # uses property setter (normalizes)
        self.name = name

    @property
    def orientation(self) -> np.ndarray:
        # return a copy so callers can't mutate internal storage accidentally
        return self._orientation.copy()

    @orientation.setter
    def orientation(self, v):
        v = np.asarray(v, dtype=float)
        if v.ndim != 1:
            raise ValueError("Orientation must be a 1D vector")
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("Orientation vector cannot be zero-length")
        self._orientation = v / norm

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
        
        to_point_normalized = to_point / np.linalg.norm(to_point)
        return np.allclose(to_point_normalized, self.orientation )

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
        
        axis = axis / np.linalg.norm(axis)
        cos_a = cos(angle)
        sin_a = sin(angle)
        ux, uy, uz = axis
        
        # Rodrigues' rotation formula
        R = np.array([
            [cos_a + ux**2 * (1 - cos_a), ux*uy*(1 - cos_a) - uz*sin_a, ux*uz*(1 - cos_a) + uy*sin_a],
            [uy*ux*(1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a), uy*uz*(1 - cos_a) - ux*sin_a],
            [uz*ux*(1 - cos_a) - uy*sin_a, uz*uy*(1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
        ])
        
        self.orientation  = R @ self.orientation 
        self.orientation  /= np.linalg.norm(self.orientation )

    def translate(self, vector: np.ndarray):
        """Translates the ray's origin by the given vector."""
        if vector.shape != self.origin.shape:
            raise ValueError("Translation vector must match the ray's origin dimension")
        self.origin += vector

    def get_angle(self, line: np.ndarray):
        """Retruns the angles created from another vector"""
        if line.shape != self.orientation .shape:
            raise ValueError("Input vector must match the ray's orientation  dimension")
        dot_product = np.dot(self.orientation , line) / (np.linalg.norm(self.orientation ) * np.linalg.norm(line))
        angle = acos(dot_product)
        return angle

    def __repr__(self):
        return f"Ray(origin={self.origin}, orientation ={self.orientation })"

class Transform:
    """
    Represents a 3D transformation with position, rotation, scale, and hierarchical parent support.
    """
    def __init__(self, position: np.ndarray, rotation: np.ndarray, scale: np.ndarray, parent=None, name: str = "Transform"):
        if position.shape != (3,) or rotation.shape != (3,) or scale.shape != (3,):
            raise ValueError("position, rotation, and scale must be 3D vectors")
            
        self.position = position
        self.rotation = rotation  # in radians (Euler angles)
        self.scale = scale
        self.local_position = np.zeros(3)
        self.local_rotation = np.zeros(3)  # in radians
        self.local_scale = np.ones(3)
        self.parent = parent
        self.name = name

        self.update_orientations()

    def update_orientations(self):
        """Updates the forward, right, and up vectors based on the current rotation."""
        rx, ry, rz = self.rotation
        cx, sx = cos(rx), sin(rx)
        cy, sy = cos(ry), sin(ry)
        cz, sz = cos(rz), sin(rz)
        
        # Combined rotation matrix (Z * Y * X)
        R = np.array([
            [cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
            [cy*sz, cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz],
            [-sy, cy*sx, cx*cy]
        ])
        
        self.forward = R @ np.array([0, 0, 1])
        self.right = R @ np.array([1, 0, 0])
        self.up = R @ np.array([0, 1, 0])

    def get_local_matrix(self):
        """Returns the local transformation matrix."""
        tx, ty, tz = self.local_position
        sx, sy, sz = self.local_scale
        rx, ry, rz = self.local_rotation
        
        # Rotation matrices
        rot_x = np.array([
            [1, 0, 0],
            [0, cos(rx), -sin(rx)],
            [0, sin(rx), cos(rx)]
        ])
        rot_y = np.array([
            [cos(ry), 0, sin(ry)],
            [0, 1, 0],
            [-sin(ry), 0, cos(ry)]
        ])
        rot_z = np.array([
            [cos(rz), -sin(rz), 0],
            [sin(rz), cos(rz), 0],
            [0, 0, 1]
        ])

        # Scale and translation
        scale = np.diag([sx, sy, sz])
        translate = np.array([tx, ty, tz])

        # Combine: T * Rz * Ry * Rx * S
        return np.eye(4)
        
    def get_global_matrix(self):
        """Returns the global transformation matrix."""
        local = self.get_local_matrix()
        if self.parent is not None:
            return self.parent.get_global_matrix() @ local
        return local
    
    def get_global_position(self):
        """Returns the global position of the transform."""
        global_matrix = self.get_global_matrix()
        return global_matrix[:3, 3]

    def get_global_rotation(self):
        """Returns the global rotation (Euler angles) of the transform."""
        global_matrix = self.get_global_matrix()
        # This is a complex operation and depends on the specific matrix decomposition.
        # A simplified version is not straightforward.
        return np.array([acos(global_matrix[0,0]), acos(global_matrix[1,1]), acos(global_matrix[2,2])])

    def get_global_scale(self):
        """Returns the global scale of the transform."""
        global_matrix = self.get_global_matrix()
        return np.linalg.norm(global_matrix[:3, :3], axis=0)

    def translate(self, vector: np.ndarray, isWorld: bool = False):
        """Translates the transform in the specified space (world or local)."""
        if isWorld:
            self.position += vector
        else:
            self.local_position += vector
    
    def rotate(self, angle: float, axis: np.ndarray, isWorld: bool = False):
        """Rotates the transform around a specified axis."""
        if np.linalg.norm(axis) == 0:
            raise ValueError("Cannot rotate around a zero-length vector")

        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        cos_angle = cos(angle)
        sin_angle = sin(angle)
        cross_product_matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = np.eye(3) * cos_angle + cross_product_matrix * sin_angle + np.outer(axis, axis) * (1 - cos_angle)

        if isWorld:
            self.position = rotation_matrix @ self.position
            self.rotation = rotation_matrix @ self.rotation
        else:
            self.local_position = rotation_matrix @ self.local_position
            self.local_rotation = rotation_matrix @ self.local_rotation
    
    def enlarge(self, vector: np.ndarray, isWorld: bool = False):
        """Enlarges the transform in the specified space (world or local)."""
        if isWorld:
            self.scale += vector
        else:
            self.local_scale += vector
    
    def reflect(self, axis: Ray, isWorld: bool = False):
        """Reflects the transform across a specified axis."""
        if np.linalg.norm(axis.orientation ) == 0:
            raise ValueError("Cannot reflect across a zero-length vector")
            
        n = axis.orientation  / np.linalg.norm(axis.orientation )
        reflection_matrix = np.eye(3) - 2 * np.outer(n, n)
        
        if isWorld:
            self.position = reflection_matrix @ self.position
            self.rotation = reflection_matrix @ self.rotation
        else:
            self.local_position = reflection_matrix @ self.local_position
            self.local_rotation = reflection_matrix @ self.local_rotation

    def __repr__(self):
        return f"Transform(position={self.position}, rotation={self.rotation}, scale={self.scale})"

class Ratio:
    """
    Represents a ratio (fraction) with a denominator and numerator.
    Supports basic arithmetic operations and comparisons.
    """
    def __init__(self, denominator: float, numerator: float):
        """
        Creates a ratio (fraction) with a denominator and numerator.
        """
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
            
        self.denominator = denominator
        self.numerator = numerator

    def __add__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Ratio(new_denominator, new_numerator)
    
    def __sub__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
            
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Ratio(new_denominator, new_numerator)

    def __mul__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
            
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Ratio(new_denominator, new_numerator)
    
    def __truediv__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        if other.numerator == 0:
            raise ValueError("Cannot divide by a Ratio with a numerator of zero")
            
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return Ratio(new_denominator, new_numerator)
    
    def __repr__(self):
        return f"Ratio({self.numerator}/{self.denominator})"
    
    def __float__(self):
        return self.numerator / self.denominator
    
    def __neg__(self):
        return Ratio(-self.denominator, -self.numerator)
    
    def __eq__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.numerator * other.denominator == self.denominator * other.numerator
    
    def __lt__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.numerator * other.denominator < self.denominator * other.numerator
        
    def __le__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.numerator * other.denominator <= self.denominator * other.numerator
        
    def __gt__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.numerator * other.denominator > self.denominator * other.numerator
        
    def __ge__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.numerator * other.denominator >= self.denominator * other.numerator
        
    def __ne__(self, other):
        if not isinstance(other, Ratio):
            return NotImplemented
        return self.numerator * other.denominator != self.denominator * other.numerator