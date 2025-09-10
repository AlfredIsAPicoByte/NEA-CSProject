from src.Basic import *
from src.InputManager import InputManager
from enum import Enum
from math import sin, cos, tan, radians

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

class Camera:
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
    def __init__ (self, transform: Transform, fov: float, near: float, far: float, width: int, height: int, type: CameraType = CameraType.PERSPECTIVE, mode: CameraMode = CameraMode.FIRST_PERSON, speed: float = 5.0, speed_multiplier: float = 2.0, sensitivity: float = 10.0, name: str = "Camera", id: int = 0):
        """
        Initializes the Camera with the given parameters.
        """
        self.transform = transform
        self.fov = fov
        self.near = near
        self.far = far

        self.type = type
        self.mode = mode

        self.width = width
        self.height = height
        self.aspect = Ratio(width, height)

        self.speed = speed
        self.speed_multiplier = speed_multiplier
        self.sensitivity = sensitivity

        self.name = name
        self.id = id

    def get_view_matrix(self) -> Matrix:
        # The view matrix is typically the inverse of the camera's global transform
        return self.transform.get_global_matrix().Inverse()

    def get_projection_matrix(self) -> Matrix:
        if self.type == CameraType.PERSPECTIVE:
            # Perspective projection matrix calculation
            f = 1.0 / (self.fov / 2).tan()
            aspect = float(self.aspect)
            near, far = self.near, self.far
            m = Matrix([
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
            m = Matrix([
                [2 / (right - left), 0, 0, -(right + left) / (right - left)],
                [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                [0, 0, 0, 1]
            ])
            return m
        else:
            raise ValueError("Unknown camera type")

    def get_camera_matrix (self) -> Matrix:
        return self.get_projection_matrix() * self.get_view_matrix()
    
    def resize(self, width: float, height: float):
        self.width = width
        self.height = height
        self.aspect = Ratio(width, height)

    def move(self, input: InputManager, delta_time: float):
        if self.mode == CameraMode.FIRST_PERSON:
            self.FirstPersonMovement(input, delta_time)
        elif self.mode == CameraMode.PLANE:
            self.PlaneMovement(input, delta_time)
        elif self.mode == CameraMode.ORBIT:
            self.OrbitMovement(input, delta_time)
        else:
            raise ValueError("Unknown camera mode")

    firstClick: bool = False

    def FirstPersonMovement(self, input: InputManager, delta_time: float):
        velocity = self.speed * delta_time
        if input.is_key_pressed('SHIFT'):
            velocity *= self.speed_multiplier

        # Move forward/backward
        if input.is_key_pressed('W'):
            self.transform.position += self.transform.forward * velocity
        if input.is_key_pressed('S'):
            self.transform.position -= self.transform.forward * velocity
        # Move left/right
        if input.is_key_pressed('A'):
            self.transform.position -= self.transform.right * velocity
        if input.is_key_pressed('D'):
            self.transform.position += self.transform.right * velocity
        # Move up/down
        if input.is_key_pressed('SPACE'):
            self.transform.position += self.transform.up * velocity
        if input.is_key_pressed('CTRL'):
            self.transform.position -= self.transform.up * velocity

        # Mouse look (FPS style)
        if input.is_mouse_button_pressed('RIGHT'):
            input.hide_cursor()
            if self.firstClick:
                input.set_mouse_position(self.width / 2, self.height / 2)
            self.firstClick = False

            mouse_dx, mouse_dy = input.get_mouse_delta((self.width / 2, self.height / 2))
            yaw = (self.sensitivity * mouse_dx / self.width) * delta_time
            pitch = (self.sensitivity * mouse_dy / self.height) * delta_time

            # Clamp pitch to avoid flipping
            max_pitch = radians(89.0)
            current_pitch = self.transform.rotation.get_pitch() if hasattr(self.transform.rotation, "get_pitch") else 0
            new_pitch = current_pitch + pitch
            if abs(new_pitch) > max_pitch:
                pitch = max_pitch * (1 if new_pitch > 0 else -1) - current_pitch

            # Yaw rotation around global up
            yaw_matrix = Matrix([
            [cos(yaw), 0, sin(yaw), 0],
            [0, 1, 0, 0],
            [-sin(yaw), 0, cos(yaw), 0],
            [0, 0, 0, 1]
            ])
            # Pitch rotation around local right
            pitch_matrix = Matrix([
            [1, 0, 0, 0],
            [0, cos(pitch), -sin(pitch), 0],
            [0, sin(pitch), cos(pitch), 0],
            [0, 0, 0, 1]
            ])
            # Apply yaw then pitch
            self.transform.rotation = yaw_matrix * self.transform.rotation
            self.transform.rotation = self.transform.rotation * pitch_matrix

            input.set_mouse_position(self.width / 2, self.height / 2)
        else:
            self.firstClick = True
            input.show_cursor()

    def PlaneMovement(self, input: InputManager, delta_time: float):
        velocity = self.speed * delta_time
        if input.is_key_pressed('SHIFT'):
            velocity *= self.speed_multiplier

        if input.is_key_pressed('W'):
            self.transform.position += self.transform.forward * velocity
        if input.is_key_pressed('S'):
            self.transform.position -= self.transform.forward * velocity
        if input.is_key_pressed('A'):
            self.transform.position -= self.transform.right * velocity
        if input.is_key_pressed('D'):
            self.transform.position += self.transform.right * velocity
        if input.is_key_pressed('SPACE'):
            self.transform.position += self.transform.up * velocity
        if input.is_key_pressed('CTRL'):
            self.transform.position -= self.transform.up * velocity

        if input.is_mouse_button_pressed('RIGHT'):
            input.hide_cursor()

            if self.firstClick:
                input.set_mouse_position(self.width / 2, self.height / 2)
                self.firstClick = False

            mouse_dx, mouse_dy = input.get_mouse_delta((self.width / 2, self.height / 2))
            rot_x = (self.sensitivity * mouse_dx / self.width) * delta_time
            rot_y = (self.sensitivity * mouse_dy / self.height) * delta_time

            if abs(rot_y) > 89.0:
                rot_y = 89.0 * (1 if rot_y > 0 else -1)

            # Create a rotation matrix for yaw (rot_x) and pitch (rot_y)
            yaw_matrix = Matrix([
                [cos(radians(rot_x)), 0, sin(radians(rot_x)), 0],
                [0, 1, 0, 0],
                [-sin(radians(rot_x)), 0, cos(radians(rot_x)), 0],
                [0, 0, 0, 1]])
            pitch_matrix = Matrix(
                [1, 0, 0, 0],
                [0, cos(radians(rot_y)), -sin(radians(rot_y)), 0],
                [0, sin(radians(rot_y)), cos(radians(rot_y)), 0],
                [0, 0, 0, 1]
            )
            rotation_matrix = yaw_matrix * pitch_matrix
            
            # Apply the rotation to the camera's transform
            self.transform.rotation = rotation_matrix * self.transform.rotation

            input.set_mouse_position(self.width / 2, self.height / 2)
        else:
            self.firstClick = True
            input.show_cursor()

    target: Transform = Transform(Vector(0, 0, 0), Vector(0, 0, 0), Vector(1, 1, 1))
    def SelectOrbitTarget(self, target: Transform):
        self.target = target
    
    def OrbitMovement(self, input: InputManager, delta_time: float):
        # Orbit movement: rotate camera around self.target based on mouse input, zoom with scroll
        if input.is_mouse_button_pressed('RIGHT'):
            input.hide_cursor()
            if self.firstClick:
                input.set_mouse_position(self.width / 2, self.height / 2)
                self.firstClick = False

            mouse_dx, mouse_dy = input.get_mouse_delta((self.width / 2, self.height / 2))
            # Sensitivity controls orbit speed
            orbit_speed = self.sensitivity * delta_time

            # Calculate angles
            yaw = orbit_speed * mouse_dx / self.width
            pitch = orbit_speed * mouse_dy / self.height

            # Clamp pitch to avoid flipping
            max_pitch = radians(89.0)
            current_pitch = getattr(self, "orbit_pitch", 0)
            new_pitch = current_pitch + pitch
            if abs(new_pitch) > max_pitch:
                pitch = max_pitch * (1 if new_pitch > 0 else -1) - current_pitch
            self.orbit_pitch = current_pitch + pitch

            # Calculate orbit radius (distance to target)
            radius = (self.transform.position - self.target.position).length()
            # Zoom in/out with mouse wheel
            scroll = input.get_mouse_scroll()
            radius -= scroll * self.speed * delta_time
            radius = max(radius, 0.1)  # Prevent negative/zero radius

            # Calculate new camera position in spherical coordinates
            orbit_yaw = getattr(self, "orbit_yaw", 0) + yaw
            self.orbit_yaw = orbit_yaw

            x = self.target.position.x + radius * cos(self.orbit_pitch) * sin(orbit_yaw)
            y = self.target.position.y + radius * sin(self.orbit_pitch)
            z = self.target.position.z + radius * cos(self.orbit_pitch) * cos(orbit_yaw)
            self.transform.position = Vector(x, y, z)

            # Look at the target
            self.look_at(self.target)

            input.set_mouse_position(self.width / 2, self.height / 2)
        else:
            self.firstClick = True
            input.show_cursor()

    def look_at(self, target: Transform):
        direction = (target.position - self.transform.position).normalized()
        # Assuming the up vector is always (0, 1, 0)
        up = Vector(0, 1, 0)
        right = direction.cross(up).normalized()
        up = right.cross(direction).normalized()

        # Create a rotation matrix
        rotation_matrix = Matrix([
            [right.x, right.y, right.z, 0],
            [up.x, up.y, up.z, 0],
            [-direction.x, -direction.y, -direction.z, 0],
            [0, 0, 0, 1]
        ])

        self.transform.rotation = rotation_matrix.to_euler_angles()
    
    def look_at_screen_coords(self, screen_x: float, screen_y: float, screen_width: float, screen_height: float):
        # Convert screen coordinates to normalized device coordinates (NDC)
        ndc_x = (2.0 * screen_x) / screen_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / screen_height

        # Convert NDC to world coordinates
        projection_matrix = self.get_projection_matrix()
        view_matrix = self.get_view_matrix()
        inv_projection_view = (projection_matrix * view_matrix).Inverse()
        clip_coords = Vector(ndc_x, ndc_y, -1.0, 1.0)
        world_coords = inv_projection_view * clip_coords
        world_coords = world_coords / world_coords.w
        direction = (world_coords - self.transform.position).normalized()
        self.transform.look_at(self.transform.position + direction)
