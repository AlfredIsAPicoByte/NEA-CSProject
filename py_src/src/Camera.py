
from enum import Enum

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

class CameraObject:
    """
    A Camera class that supports different projection types and movement modes.
    Attributes:
        transform (Transform): The camera's transform.
        fov (float): Field of view for perspective projection or size for orthographic projection.
        near (float): Near clipping plane.
        far (float): Far clipping plane.
        type (CameraType): The camera's projection type.
        mode (CameraMode): The camera's movement mode.
        width (float): Width of the viewport.
        height (float): Height of the viewport.
        aspect (Ratio): Aspect ratio of the viewport.
        speed (float): Movement speed.
        speed_multiplier (float): Speed multiplier when a modifier key is pressed.
        sensitivity (float): Mouse sensitivity for looking around.
        
    Methods:
        get_view_matrix(): Returns the view matrix.
        get_projection_matrix(): Returns the projection matrix.
        get_camera_matrix(): Returns the combined camera matrix (projection * view).
        move(input: InputManager): Updates the camera's position and orientation based on input.
        FirstPersonMovement(input: InputManager, delta_time: float): Handles first-person movement.
        PlaneMovement(input: InputManager, delta_time: float): Handles plane movement.
        OrbitMovement(input: InputManager, delta_time: float): Handles orbit movement.
        look_at(target: Transform): Orients the camera to look at a target transform.
        look_at_screen_coords(screen_x: float, screen_y: float, screen_width: float, screen_height: float): Orients the camera to look at a point in screen coordinates.
    """
    def __init__ (self, transform: Transform, fov: float, near: float, far: float, width: int, height: int, camType: CameraType = CameraType.PERSPECTIVE, camMode: CameraMode = CameraMode.FIRST_PERSON, name: str = "Camera", id: int = 0):
        """
        Initializes the Camera with the given parameters.
        """
        self.transform = transform
        self.fov = fov
        self.near = near
        self.far = far

        self.type = camType
        self.mode = camMode

        self.width = width
        self.height = height
        self.aspect = Ratio(width, height)

        self.name = name
        self.id = id

    def get_view_matrix(self) -> np.ndarray:
        # The view matrix is typically the inverse of the camera's global transform
        return np.linalg.inv(np.array(self.transform.get_global_matrix()))

    def get_projection_matrix(self) -> np.ndarray:
        if self.type == CameraType.PERSPECTIVE:
            # Perspective projection matrix calculation
            f = 1.0 / (self.fov / 2).tan()
            aspect = float(self.aspect)
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
            right = self.aspect * self.fov
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
        self.width = width
        self.height = height
        self.aspect = Ratio(width, height)