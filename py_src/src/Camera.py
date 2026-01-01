import numpy as np
from enum import Enum

from PrimaryStructures import Transform, Ratio

"""

"""

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
    """
    def __init__ (
            self,
            transform: Transform,
            fov: float,
            near: float,
            far: float,
            resolution_width: int,
            resolution_height: int,
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

        self.type = camera_type
        self.mode = camera_mode

        if resolution_width <= 0 or resolution_height <= 0:
            raise ValueError("Width and Height must be greater than 0")
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.aspect_ratio = Ratio(resolution_width, resolution_height)

        self.name = name

    def get_view_matrix(self) -> np.ndarray:
        # The view matrix is typically the inverse of the camera's global transform
        return np.linalg.inv(np.array(self.transform.get_global_matrix()))

    def get_projection_matrix(self) -> np.ndarray:
        if self.type == CameraType.PERSPECTIVE:
            # Perspective projection matrix calculation
            f = 1.0 / (self.fov / 2).tan()
            aspect = float(self.aspect_ratio)
            near, far = self.near, self.far
            m = np.array([
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0]
            ])
            return m
        elif self.type == CameraType.ORTHOGRAPHIC:
            # Orthographic projection matrix calculation
            right = self.aspect_ratio * self.fov
            left = -right
            top = self.fov
            bottom = -top
            near, far = self.near, self.far
            m = np.array([
                [2 / (right - left), 0, 0, -(right + left) / (right - left)],
                [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                [0, 0, 0, 1]
            ])
            return m
        else:
            raise ValueError("Unknown camera type")

    def get_camera_matrix (self) -> np.ndarray:
        return self.get_projection_matrix() * self.get_view_matrix()
    
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

    def get_ray_from_screen(self, screen_x: float, screen_y: float) -> tuple:
        """
        Convert screen coordinates (0-width, 0-height) to a world-space ray.
        Useful for picking/raycasting.
        Returns: (origin, direction) both as np.ndarray
        """
        # Normalized device coordinates
        ndc_x = (2.0 * screen_x) / self.resolution_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / self.resolution_height  # Flip Y for OpenGL
        
        # Inverse projection to get view-space coordinates
        proj_inv = np.linalg.inv(self.get_projection_matrix())
        view_x = ndc_x * np.tan(np.radians(self.fov / 2)) * float(self.aspect_ratio)
        view_y = ndc_y * np.tan(np.radians(self.fov / 2))
        
        # View-space ray
        ray_view = np.array([view_x, view_y, -1.0, 0.0])
        
        # Transform to world space
        view_inv = np.linalg.inv(self.get_view_matrix())
        ray_world = view_inv @ ray_view
        
        origin = self.transform.get_global_position()
        direction = ray_world[:3] / np.linalg.norm(ray_world[:3])
        
        return origin, direction

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