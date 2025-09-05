from src.Basic import *
from src.InputManager import InputManager
from enum import Enum
from math import sin, cos, tan, radians

class CameraType (Enum):
    PERSPECTIVE = 1
    ORTHOGRAPHIC = 2

class CameraMode (Enum):
    FIRST_PERSON = 1
    PLANE = 2
    ORBIT = 3

class Camera:
    speed: float = 5.0
    speed_multiplier: float = 2.0
    sensitivity: float = 0.1

    def __init__ (self, transform: Transform, fov: float, near: float, far: float, width: float, height: float, type: CameraType = CameraType.PERSPECTIVE, mode: CameraMode = CameraMode.FIRST_PERSON):
        self.transform = transform
        self.fov = fov
        self.near = near
        self.far = far

        self.type = type
        self.mode = mode

        self.width = width
        self.height = height

        self.aspect = Ratio(width, height)

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
    
    def move(self, input: InputManager):
        if self.mode == CameraMode.FIRST_PERSON:
            self.FirstPersonMovement(input, 0.016)
        elif self.mode == CameraMode.PLANE:
            self.PlaneMovement(input, 0.016)
        elif self.mode == CameraMode.ORBIT:
            self.OrbitMovement(input, 0.016)
        else:
            raise ValueError("Unknown camera mode")

    firstClick: bool = False

    def FirstPersonMovement(self, input: InputManager, delta_time: float):
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

            mouse_dx, mouse_dy = input.get_mouse_delta()
            rot_x = self.sensitivity * mouse_dx / self.width
            rot_y = self.sensitivity * mouse_dy / self.height

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

    def PlaneMovement(self, input: InputManager, delta_time: float):
        raise NotImplementedError("PlaneMovement not implemented yet")

    target: Transform = Transform()
    def OrbitMovement(self, input: InputManager, delta_time: float):
        raise NotImplementedError("OrbitMovement not implemented yet")

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
