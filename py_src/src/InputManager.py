from pynput import keyboard, mouse

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

    def get_mouse_delta(self):
        return mouse.Controller().position
    
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
