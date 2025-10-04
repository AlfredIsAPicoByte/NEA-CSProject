from src.PrimaryStructures import Ray
from src.Reflections import reflect_ray
from src.Refractions import refract_ray
import numpy as np

"""

"""

def clamp(value, min_value: float|int = 0.0, max_value: float|int = 1.0):
    return max(min_value, min(value, max_value))

class ColorData:
    def __init__(self, r: float = 0, g: float = 0, b: float = 0, alpha: float = 1.0):
        """Clamp and store RGBA values between 0.0 and 1.0."""
        self.rgba = np.array([clamp(r), clamp(g), clamp(b), clamp(alpha)], dtype=float)

    @classmethod
    def use_hex(cls, hex_str: str):
        """Create a ColorData object from a hex string."""
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) not in (6, 8):
            raise ValueError("Hex color must be 6 (RGB) or 8 (RGBA) characters long.")

        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = int(hex_str[6:8], 16) / 255.0 if len(hex_str) == 8 else 1.0

        rgba = np.array([clamp(r), clamp(g), clamp(b), clamp(a)], dtype=float)
        return cls(*rgba)

    @property
    def red(self): return self.rgba[0]
    @property
    def green(self): return self.rgba[1]
    @property
    def blue(self): return self.rgba[2]
    @property
    def alpha(self): return self.rgba[3]

    def __mul__(self, other):
        if isinstance(other, ColorData):
            return ColorData(*np.clip(self.rgba * other.rgba, 0, 1))
        elif isinstance(other, (int, float)):
            r, g, b, a = self.rgba
            return ColorData(clamp(r * other), clamp(g * other), clamp(b * other), a)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, ColorData):
            return ColorData(*np.clip(self.rgba + other.rgba, 0, 1))
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, ColorData):
            return ColorData(*np.clip(self.rgba - other.rgba, 0, 1))
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, ColorData):
            return ColorData(*np.clip(self.rgba / np.maximum(other.rgba, 1e-8), 0, 1))
        elif isinstance(other, (int, float)):
            return ColorData(*np.clip(self.rgba / max(other, 1e-8), 0, 1))
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, ColorData):
            return np.allclose(self.rgba, other.rgba)
        return NotImplemented

    def __iter__(self):
        yield from self.rgba

    def to_hex(self, include_alpha=True):
        r, g, b, a = (self.rgba * 255).astype(int)
        return f"#{r:02X}{g:02X}{b:02X}{a:02X}" if include_alpha else f"#{r:02X}{g:02X}{b:02X}"

    def __repr__(self):
        r, g, b, a = self.rgba
        return f"ColorData(r={r:.3f}, g={g:.3f}, b={b:.3f}, a={a:.3f})"

class LightRay(Ray, ColorData):
    def __init__(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        color: ColorData,
        intensity: float = 1.0,
        name: str = "Light Ray"
    ):
        # Normalize direction safely
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero-length.")
        direction = direction / norm

        # Initialize both parent classes
        Ray.__init__(self, origin, direction, name)
        ColorData.__init__(self, color.red, color.green, color.blue, color.alpha)

        self.intensity = float(intensity)
        self.name = name  # keep name stored directly
    
    def Attenuate(self, factor: float):
        """Return a dimmed copy of the LightRay."""
        return LightRay(
            self.origin,
            self.direction,
            self.color_data * factor,
            intensity=self.intensity * factor,
            name=self.name + " (dimmed)"
        )
    
    @property
    def ray(self) -> Ray:
        """Return an independent Ray copy of this LightRay's geometric component."""
        return Ray(self.origin.copy(), self.direction.copy(), self.name)
    
    @property
    def color_data(self) -> ColorData:
        """Return the color data of this LightRay as a ColorData object."""
        return ColorData(self.red, self.green, self.blue, self.alpha)

    def __repr__(self):
        return (
            f"LightRay("
            f"origin={np.round(self.origin, 3)}, "
            f"direction={np.round(self.direction, 3)}, "
            f"color={self.color_data}, "
            f"intensity={self.intensity:.2f})"
        )

class Material:
    def __init__(self, color: ColorData, roughness: float, glossiness: float):
        """
        A simple material that reflects LightRay based on its color, roughness, and glossiness.
        
        Attribute:
            color: The base color of the material.
            roughness: A value between 0.0 and 1.0 that determines
                how rough the surface is. 0.0 = perfectly smooth, 1.0 = very rough.
            glossiness: A value between 0.0 and 1.0 that determines
                how glossy the surface is. 0.0 = matte, 1.0 = perfectly glossy.
        """
        self.color = color
        self.roughness = clamp(roughness)
        self.glossiness = clamp(glossiness)

    def Affect(self, data):
        """Compute reflected color based on surface color and light color."""
        return self.color * (light_color * self.glossiness)

    def __repr__(self):
        return (
            f"Material(color={self.color}, "
            f"roughness={self.roughness:.2f}, glossiness={self.glossiness:.2f})"
        )