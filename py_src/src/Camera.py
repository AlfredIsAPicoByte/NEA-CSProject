import numpy as np
import math
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
    """
    def __init__ (
            self,
            transform: Transform,
            fov: float = 60,
            near: float = 0.01,
            far: float = 1000,
            resolution_width: int = 16,
            resolution_height: int = 9,
            camera_type: CameraType = CameraType.PERSPECTIVE,
            camera_mode: CameraMode = CameraMode.FIRST_PERSON,
            name: str = "Camera"
        ):
        self.transform = transform

        self.fov = fov
        self.near = near
        self.far = far

        # Allow either camera_type or legacy camType
        self.type = camera_type
        self.mode = camera_mode

        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.aspect_ratio = Ratio(self.resolution_width, self.resolution_height)

        self.name = name

    def __post_init__(self):
        if self.fov <= 0:
            raise ValueError("FOV must be greater than 0")
        
        if self.near <= 0 or self.far <= self.near:
            raise ValueError("Invalid near and far plane values")
    
        if self.resolution_width is None or self.resolution_height is None:
            raise ValueError("Width and Height must be provided (as 'resolution_width, resolution_height')")

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
            "aspect": float(self.aspect_ratio.value()),
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

    def get_camera_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a ray (origin, direction) for UV coordinates (0-1).
        """
        # --- ORTHOGRAPHIC ---
        if self.type == CameraType.ORTHOGRAPHIC:
            # 1. Direction is always forward
            direction = self.transform.forward
            
            # 2. Origin shifts along the camera plane
            height_size = self.ortho_scale # or self.fov if reusing that field
            width_size = height_size * self.aspect_ratio.value

            # Map 0..1 to -Width/2 .. +Width/2
            offset_x = (u - 0.5) * width_size
            offset_y = (0.5 - v) * height_size 

            origin = (
                self.transform.position + 
                (self.transform.right * offset_x) + 
                (self.transform.up * offset_y)
            )
            return origin + direction * self.near, direction

        # --- PERSPECTIVE ---
        # Geometric approach (Snippet 1 style) is usually faster for CPU raytracing 
        # than matrix inversion if you aren't caching the inverse matrix.
        
        # 1. NDC (-1 to 1)
        ndc_x = (2.0 * u) - 1.0
        ndc_y = 1.0 - (2.0 * v)

        # 2. Scale by FOV
        # vertical_fov is in degrees
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        
        # Camera Space Direction
        # Assuming standard camera: Right=+X, Up=+Y, Forward=+Z (or -Z depending on convention)
        # We use the Basis vectors directly to go straight to World Space
        
        x_scale = ndc_x * self.aspect_ratio.value * tan_half_fov
        y_scale = ndc_y * tan_half_fov
        
        # Construct direction in world space by summing basis vectors
        # This assumes 'forward' is the direction the camera looks.
        direction = (
            (self.transform.forward) +         # 1 unit forward
            (self.transform.right * x_scale) + # Horizontal spread
            (self.transform.up * y_scale)      # Vertical spread
        )
        
        # Normalize
        direction = direction / np.linalg.norm(direction)
        
        return self.transform.position + direction * self.near, direction

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