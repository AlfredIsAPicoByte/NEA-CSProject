from src.Basic import Ray, Vector
from src.Camera import Camera

class EventListener:
    def __init__(self, **kwargs):
        self.on_press = kwargs.get('on_press', lambda key: None)
        self.on_release = kwargs.get('on_release', lambda key: None)
        self.on_move = kwargs.get('on_move', lambda x, y: None)
        self.on_click = kwargs.get('on_click', lambda x, y, button, pressed: None)
        self.on_scroll = kwargs.get('on_scroll', lambda x, y, dx, dy: None)
        self.running = False
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False

class KeyboardManager:
    def __init__(self):
        self.pressed_keys = set()
        self.listener = EventListener(
            on_press=self.press_key,
            on_release=self.release_key
        )

    def press_key(self, key):
        self.pressed_keys.add(key)

    def release_key(self, key):
        self.pressed_keys.discard(key)

    def is_key_pressed(self, key):
        return key in self.pressed_keys

class MouseManager:
    def __init__(self):
        self.position = [0, 0]
        self.pressed_buttons = set()
        self.listener = EventListener(
            on_move=self.move,
            on_click=self.is_button_pressed
        )
        self.visible = True

    def move(self, x, y):
        if self.can_move:
            self.position = [x, y]
    
    def delta_move(self, x, y):
        if self.can_move:
            self.position = [self.position[0] + x, self.position[1] + y]
    
    def get_delta(self, prev):
        return (self.position[0] - prev[0], self.position[1] - prev[1])

    def press_button(self, button):
        self.pressed_buttons.add(button)

    def release_button(self, button):
        self.pressed_buttons.discard(button)

    def is_button_pressed(self, button):
        return button in self.pressed_buttons
    
    def hide_cursor(self):
        self.visible = False

    def show_cursor(self):
        self.visible = True

    can_move = True
    def lock_cursor(self):
        self.can_move = False

    def unlock_cursor(self):
        self.can_move = True

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


class InputManager:
    def __init__(self):
        self.mouse = MouseManager()
        self.keyboard = KeyboardManager()
    
    def start_listening(self):
        self.mouse.listener.start()
        self.keyboard.listener.start()
    
    def stop_listening(self):
        self.mouse.listener.stop()
        self.keyboard.listener.stop()
    
