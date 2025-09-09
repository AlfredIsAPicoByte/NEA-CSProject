from pynput import keyboard, mouse
from src.Basic import Ray, Vector
from src.Camera import Camera

class InputManager:
    def __init__(self):
        self.keys_pressed = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            self.keys_pressed.add(str(key))

    def _on_release(self, key):
        try:
            self.keys_pressed.discard(key.char)
        except AttributeError:
            self.keys_pressed.discard(str(key))

    def is_key_pressed(self, key):
        return key in self.keys_pressed
    
    def clear(self):
        self.keys_pressed.clear()
    
    def get_pressed_keys(self):
        return list(self.keys_pressed)
    
    def set_mouse_position(self, x, y):
        if self.can_mouse_move:
            mouse.Controller().position = (x, y)
    
    def get_mouse_position(self):
        return mouse.Controller().position

    def get_mouse_delta(self, prev: tuple = None):
        current_pos = mouse.Controller().position

        if prev is None:
            return (0, 0)
        
        return (current_pos[0] - prev[0], current_pos[1] - prev[1])
    
    def is_mouse_button_pressed(self, button):
        return mouse.Controller().pressed(button)
    
    def hide_cursor(self):
        mouse.Controller().visible = False

    def show_cursor(self):
        mouse.Controller().visible = True

    can_mouse_move = True
    def lock_cursor(self):
        self.can_mouse_move = False

    def unlock_cursor(self):
        self.can_mouse_move = True

    def ray_from_mouse(self, camera: Camera):
        mouse_x, mouse_y = self.get_mouse_position()
        ndc_x = (2.0 * mouse_x) / camera.width - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y) / camera.height
        ndc_z = 1.0

        ray_clip = Vector(ndc_x, ndc_y, -1.0, 1.0)
        ray_eye = camera.get_projection_matrix().Inverse() * ray_clip
        ray_eye = Vector(ray_eye.x, ray_eye.y, -1.0, 0.0)
        ray_world = camera.get_view_matrix().Inverse() * ray_eye
        ray_world = Vector(ray_world.x, ray_world.y, ray_world.z).Normalize()

        return Ray(camera.position, ray_world)
