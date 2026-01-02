import numpy as np
from enum import Enum
from typing import Tuple

from CommonUtils import unit
from PrimaryStructures import Transform, Ratio

class CameraType (Enum):
    """
    Enum for different camera projection types.
    PERSPECTIVE: Perspective projection.
    ORTHOGRAPHIC: Orthographic projection.
    """
    PERSPECTIVE = 1
    ORTHOGRAPHIC = 2

class CameraMode (Enum):
    """
    Enum for different camera movement modes.
    FIRST_PERSON: First-person movement.
    PLANE: Plane movement (like an airplane).
    ORBIT: Orbit around a target.
    """
    FIRST_PERSON = 1
    PLANE = 2
    ORBIT = 3

class VCamera:
    """
    A virtual camera class used to store properties and process calculations.

    Backwards-compatible constructor supports the following keyword names:
     - width / height (preferred in tests)
     - resolution_width / resolution_height (explicit)
     - camType (alternate name for camera_type)
    """
    def __init__ (
            self,
            transform: Transform,
            fov: float,
            near: float,
            far: float,
            resolution_width: int = None,
            resolution_height: int = None,
            camera_type: CameraType = CameraType.PERSPECTIVE,
            camera_mode: CameraMode = CameraMode.FIRST_PERSON,
            name: str = "Camera"
        ):
        self.transform = transform

        if fov <= 0:
            raise ValueError("FOV must be greater than 0")
        self.fov = fov
        
        if near <= 0 or far <= near:
            raise ValueError("Invalid near and far plane values")
        self.near = near
        self.far = far

        # Allow either camera_type or legacy camType
        self.type = camera_type
        self.mode = camera_mode

        # Resolve width/height arguments (support both new and old names)
        w = resolution_width if resolution_width is not None else width
        h = resolution_height if resolution_height is not None else height
        if w is None or h is None:
            raise ValueError("Width and Height must be provided (either as 'width,height' or 'resolution_width,resolution_height')")

        if int(w) <= 0 or int(h) <= 0:
            raise ValueError("Width and Height must be greater than 0")

        # Store canonical integer attributes used across the codebase
        self.resolution_width = int(w)
        self.resolution_height = int(h)
        # Backwards-compatible properties `width` and `height` are provided below
        self.aspect_ratio = Ratio(self.resolution_width, self.resolution_height)

        self.name = name

    # Backwards compatible aliases
    @property
    def width(self) -> int:
        return int(self.resolution_width)

    @property
    def height(self) -> int:
        return int(self.resolution_height)

    @width.setter
    def width(self, val: int):
        self.resolution_width = int(val)
        self.aspect_ratio = Ratio(self.resolution_width, self.resolution_height)

    @height.setter
    def height(self, val: int):
        self.resolution_height = int(val)
        self.aspect_ratio = Ratio(self.resolution_width, self.resolution_height)

    def get_view_matrix(self) -> np.ndarray:
        # The view matrix is typically the inverse of the camera's global transform
        return np.linalg.inv(np.array(self.transform.get_global_matrix()))

    def get_projection_matrix(self) -> np.ndarray:
        import math
        if self.type == CameraType.PERSPECTIVE:
            # Perspective projection matrix calculation (column-major as standard)
            f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
            aspect = float(self.aspect_ratio.value)
            near, far = self.near, self.far
            m = np.array([
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
                [0.0, 0.0, -1.0, 0.0]
            ], dtype=float)
            return m
        elif self.type == CameraType.ORTHOGRAPHIC:
            # Orthographic projection matrix calculation
            # 'fov' here acts as half-height for the orthographic plane
            half_height = float(self.fov)
            half_width = half_height * float(self.aspect_ratio.value)

            left = -half_width
            right = half_width
            bottom = -half_height
            top = half_height

            near, far = self.near, self.far
            m = np.array([
                [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
                [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
                [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=float)
            return m
        else:
            raise ValueError("Unknown camera type")

    def get_camera_matrix (self) -> np.ndarray:
        # Projection * View (standard order)
        return self.get_projection_matrix() @ self.get_view_matrix()
    
    def resize(self, width: float, height: float):
        self.resolution_width = width
        self.resolution_height = height
        self.aspect_ratio = Ratio(width, height)

    def resize_aspect(self, aspect: Ratio, scale: float = 1.0):
        self.aspect_ratio = aspect
        self.resolution_width = int(aspect.width * scale)
        self.resolution_height = int(aspect.height * scale)

    def get_frustum_planes(self) -> dict:
        """
        Compute the 6 frustum planes (left, right, top, bottom, near, far).
        Useful for C++ culling and intersection tests.
        Returns: {"left": (normal, d), "right": ..., etc.}
        """
        # Extract planes from combined matrix for frustum culling
        M = self.get_camera_matrix()
        planes = {}
        
        # Normalize and extract each plane
        planes["left"] = (M[3] + M[0], M[3, 3] + M[0, 3])
        planes["right"] = (M[3] - M[0], M[3, 3] - M[0, 3])
        planes["top"] = (M[3] - M[1], M[3, 3] - M[1, 3])
        planes["bottom"] = (M[3] + M[1], M[3, 3] + M[1, 3])
        planes["near"] = (M[3] + M[2], M[3, 3] + M[2, 3])
        planes["far"] = (M[3] - M[2], M[3, 3] - M[2, 3])
        
        return planes

    def to_perspective(self) -> dict:
        """
        Export camera parameters in GLM/OpenGL-friendly format.
        C++ can read this and call glm::perspective(fov, aspect, near, far).
        """
        return {
            "fov_radians": np.radians(self.fov),
            "fov_degrees": self.fov,
            "aspect": float(self.aspect_ratio),
            "near": float(self.near),
            "far": float(self.far),
        }

    def to_matrices(self) -> dict:
        """
        Export matrices in OpenGL-friendly format (column-major, transposed if needed).
        """
        return {
            "view": self.get_view_matrix().T.tolist(),  # Transpose for column-major
            "projection": self.get_projection_matrix().T.tolist(),
            "view_projection": (self.get_projection_matrix() @ self.get_view_matrix()).T.tolist(),
        }

    def get_point_normal_from_screen_cords(self, screen_x: float, screen_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert screen coordinates (0-width, 0-height) to a world-space ray.
        Useful for picking/raycasting.
        Returns: (origin, direction) both as np.ndarray
        """
        if self.type == CameraType.ORTHOGRAPHIC:
            plane_height = self.fov
            plane_width = plane_height * self.aspect_ratio.value

            # Map u,v (0..1) to Plane Coordinates (-Width/2 .. +Width/2)
            px = (screen_x - 0.5) * plane_width
            py = (0.5 - screen_y) * plane_height # Flip Y if needed for standard coordinate systems
            # Origin = CameraPos + (Right * px) + (Up * py) + camera near plane

            direction = self.transform.forward
            origin = (
                self.transform.position + 
                (self.transform.right * px) + 
                (self.transform.up * py)
            ) + direction * self.near
            return (origin, direction)
        
        # Prespective
        # Normalize screen coordinates to NDC (-1 to 1)
        ndc_x = (2.0 * screen_x)
        ndc_y = 1.0 - (2.0 * screen_y)

        # Create homogeneous clip-space point
        clip_space = np.array([ndc_x, ndc_y, -1.0, 1.0])

        # Transform back through inverse projection and view matrices
        inv_projection = np.linalg.inv(self.get_projection_matrix())
        inv_view = np.linalg.inv(self.get_view_matrix())

        eye_space = inv_projection @ clip_space
        eye_space[2] = -1.0
        eye_space[3] = 0.0

        direction = unit((inv_view @ eye_space)[:3])

        origin = self.transform.position + direction * self.near 
        return (origin, direction)

    def export_uniforms(self) -> str:
        """
        Generate GLSL uniform declarations for copy-paste into vertex/fragment shaders.
        """
        matrices = self.to_matrices()
        return f"""
// Camera uniforms (auto-generated from VCamera)
uniform mat4 uView;           // View matrix
uniform mat4 uProjection;     // Projection matrix
uniform mat4 uViewProjection; // Combined VP matrix
uniform vec3 uCameraPos;      // Camera position (world-space)
uniform float uNear;          // Near plane
uniform float uFar;           // Far plane
uniform float uFOV;           // Field of view (radians)
"""

    def __repr__(self):
        return (f"VCamera(name={self.name}, type={self.type.name}, mode={self.mode.name}, "
                f"fov={self.fov}, aspect={self.aspect_ratio}, near={self.near}, far={self.far}, "
                f"res={self.resolution_width}x{self.resolution_height})")

"""
Camera module: Providing the Camera class with properties, projection types and movement modes.
"""