import numpy as np

from .Ray import Ray
from src.Utilities.Common import unit

class Transform:
    """
    Represents a 3D transformation with position, rotation, scale.
    
    Coordinate System: Right-Handed, Y-Up.
    Rotation Order: Z * Y * X
    """
    def __init__(
        self, 
        position: np.ndarray = np.zeros(3), 
        rotation: np.ndarray = np.zeros(3), 
        scale: np.ndarray = np.ones(3),
        name: str = 'transform'
    ):
        # Default values if None
        self.position = np.array(position, dtype=float)
        self.rotation = np.array(rotation, dtype=float)
        self.scale    = np.array(scale, dtype=float)

        # explicit local offsets applied on top of the base transform
        self.local_position = np.zeros(3, dtype=float)
        self.local_rotation = np.zeros(3, dtype=float)
        self.local_scale = np.ones(3, dtype=float)

        self.name = name
        
        # Cache for orientation vectors
        self.forward = np.array([0.0, 0.0, 1.0])
        self.right   = np.array([1.0, 0.0, 0.0])
        self.up      = np.array([0.0, 1.0, 0.0])

        self.update_orientations()

    @classmethod
    def identity(cls):
        return cls(np.zeros(3), np.zeros(3), np.ones(3))
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Transform':
        """Creates a Transform from a 4x4 matrix."""
        p, r, s = cls.decompose_matrix(matrix)
        return cls(p, r, s)

    # -------------------------------------------------------------------------
    # Math Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _matrix_to_euler(R: np.ndarray) -> np.ndarray:
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
    
    @classmethod
    def decompose_matrix(cls, matrix: np.ndarray, epsilon: float = 1e-8):
        """
        Extracts Position, Rotation (Euler), and Scale from a 4x4 transformation matrix.
        """
        position = matrix[:3, 3]

        col_0 = matrix[:3, 0]
        col_1 = matrix[:3, 1]
        col_2 = matrix[:3, 2]

        sx = np.linalg.norm(col_0)
        sy = np.linalg.norm(col_1)
        sz = np.linalg.norm(col_2)
        scale = np.array([sx, sy, sz])

        if sx > epsilon: col_0 /= sx
        if sy > epsilon: col_1 /= sy
        if sz > epsilon: col_2 /= sz

        rotation_mat = np.column_stack((col_0, col_1, col_2))
        
        if np.linalg.det(rotation_mat) < 0:
            scale = -scale
            rotation_mat = -rotation_mat

        rotation = cls._matrix_to_euler(rotation_mat)
        return position, rotation, scale

    # -------------------------------------------------------------------------
    # State Updates
    # -------------------------------------------------------------------------

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
        
        # Basis Vectors
        # Assuming standard identity is Right=(1,0,0), Up=(0,1,0), Forward=(0,0,1)
        self.right   = R @ np.array([1, 0, 0])
        self.up      = R @ np.array([0, 1, 0])
        self.forward = R @ np.array([0, 0, 1])

    def get_global_matrix(self) -> np.ndarray:
        """Returns the global transformation matrix: base @ local."""
        base = self._make_transform_matrix(self.position, self.rotation, self.scale)
        local = self._make_transform_matrix(self.local_position, self.local_rotation, self.local_scale)
        combined = base @ local 
        
        return combined

    # Backwards-compatible alias used by older code
    def to_matrix(self) -> np.ndarray:
        """Alias for `get_global_matrix`. Kept for compatibility with existing code paths."""
        return self.get_global_matrix()

    def get_inverse_matrix(self) -> np.ndarray:
        """Returns the inverse of the global transform matrix (World -> Local)."""
        try:
            return np.linalg.inv(self.get_global_matrix())
        except np.linalg.LinAlgError:
            return np.eye(4)
            
    # -------------------------------------------------------------------------
    # Manipulation Methods
    # -------------------------------------------------------------------------
    
    def look_at(self, target_position: np.ndarray, world_up: np.ndarray = np.array([0, 1, 0])):
        target_position = np.asarray(target_position, dtype=float)
        
        # Calculate direction from eye to target
        direction = target_position - self.position
        dist = np.linalg.norm(direction)
        if dist < 1e-6: return

        forward = direction / dist
        
        # Check for gimbal lock (looking straight up/down)
        if np.abs(np.dot(forward, world_up)) > 0.999:
            right = np.array([1, 0, 0])
        else:
            right = np.cross(world_up, forward)
            right = unit(right)
        
        up = np.cross(forward, right)
        
        # Construct rotation matrix from basis vectors
        # [ R U F ]
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
        # Note: This simple addition logic works for small updates but isn't 
        # mathematically perfect for accumulating arbitrary axis rotations over time.
        # For a full engine, you'd use Quaternions here.
        axis = unit(np.asarray(axis, dtype=float))
        delta = axis * angle
        if space == "global": self.rotation += delta
        else: self.local_rotation += delta
        self.update_orientations()
    
    def enlarge(self, vector: np.ndarray, space: str = "global"):
        if space == "global": self.scale *= vector
        else: self.local_scale *= vector
        self.update_orientations()

    # -------------------------------------------------------------------------
    # Application Methods (Transforming data)
    # -------------------------------------------------------------------------

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Local -> World (Points)"""
        point = np.asarray(point, dtype=float)
        model_matrix = self.get_global_matrix()
        p_hom = np.append(point, 1.0)
        return (model_matrix @ p_hom)[:3]

    def transform_direction(self, direction: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Local -> World (Directions)"""
        direction = np.asarray(direction, dtype=float)
        # Extract 3x3 for rotation/scale
        rs_matrix = self.get_global_matrix()[:3, :3]
        transformed = rs_matrix @ direction
        return unit(transformed) if normalize else transformed

    def transform_normal(self, normal: np.ndarray) -> np.ndarray:
        """Local -> World (Normals, handling non-uniform scale)"""
        normal = np.asarray(normal, dtype=float)
        # Inverse Transpose of the upper 3x3
        rs_matrix = self.get_global_matrix()[:3, :3]
        try:
            norm_matrix = np.linalg.inv(rs_matrix).T
        except np.linalg.LinAlgError:
            norm_matrix = rs_matrix 
        return unit(norm_matrix @ normal)
    
    def transform_ray(self, ray: Ray, normalize: bool = False) -> Ray:
        """Local -> World (Ray)"""
        # Note: If your Ray class expects World -> Local, use inverse_transform_ray
        # Usually: Object stores geometry in Local. Ray is World.
        # To intersect, we transform Ray World -> Local.
        return Ray(
            self.transform_point(ray.origin),
            self.transform_direction(ray.direction, normalize)
        )

    def inverse_transform_point(self, world_point: np.ndarray) -> np.ndarray:
        """World -> Local (Points)"""
        world_point = np.asarray(world_point, dtype=float)
        inv_matrix = self.get_inverse_matrix()
        p_hom = np.append(world_point, 1.0)
        return (inv_matrix @ p_hom)[:3]

    def inverse_transform_direction(self, world_direction: np.ndarray, normalize: bool = False) -> np.ndarray:
        """World -> Local (Directions)"""
        world_direction = np.asarray(world_direction, dtype=float)
        inv_rs_matrix = self.get_inverse_matrix()[:3, :3]
        transformed = inv_rs_matrix @ world_direction
        return unit(transformed) if normalize else transformed

    def inverse_transform_ray(self, ray: Ray) -> Ray:
        """
        World -> Local (Ray).
        Essential for Ray Tracing: We transform the incoming ray into the object's local space,
        intersect with the perfect sphere/cube, and then transform the result back.
        """
        return Ray(
            self.inverse_transform_point(ray.origin),
            self.inverse_transform_direction(ray.direction, normalize=True)
        )

    # -------------------------------------------------------------------------
    # Dunder Methods
    # -------------------------------------------------------------------------

    def __mul__(self, other):
        """Allows `t3 = t1 * t2` or `point = t1 * point`"""
        if isinstance(other, Transform):
            # Combine matrices
            mat_a = self.get_global_matrix()
            mat_b = other.get_global_matrix()
            new_mat = mat_a @ mat_b
            return Transform.from_matrix(new_mat)

        elif isinstance(other, np.ndarray):
            # Transform Vector/Point
            if other.shape == (3,):
                return self.transform_point(other)
            return self.get_global_matrix() @ other
        
        raise TypeError(f"Cannot multiply Transform by type {type(other)}")

    def __repr__(self):
        return f"Transform(pos={self.position}, rot={self.rotation}, scale={self.scale})"