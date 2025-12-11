import numpy as np
from PrimaryStructures import Ray
from Reflections import reflect_ray
from Refractions import refract_ray

"""

"""

def clamp(value, min_value: float|int = 0.0, max_value: float|int = 1.0):
    return max(min_value, min(value, max_value))

class Color:
    def __init__(self, r: float = 0, g: float = 0, b: float = 0, alpha: float = 1.0):
        """Clamp and store RGBA values between 0.0 and 1.0."""
        self.rgba = np.array([clamp(r), clamp(g), clamp(b), clamp(alpha)], dtype=float)
    
    def clamp(self):
        """Clamp the RGBA values to be within [0.0, 1.0]."""
        self.rgba = np.clip(self.rgba, 0.0, 1.0)
        return self
    
    @classmethod
    def from_hex(cls, hex_str: str):
        """Create a Color object from a hex string."""
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) not in (6, 8):
            raise ValueError("Hex color must be 6 (RGB) or 8 (RGBA) characters long.")

        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = int(hex_str[6:8], 16) / 255.0 if len(hex_str) == 8 else 1.0

        rgba = np.array([clamp(r), clamp(g), clamp(b), clamp(a)], dtype=float)
        return cls(*rgba)

    @classmethod
    def from_array(cls, arr):
        """Create a Color object from an array-like input."""
        if len(arr) not in (3, 4):
            raise ValueError("Array must have 3 (RGB) or 4 (RGBA) elements.")
        r, g, b = arr[0], arr[1], arr[2]
        a = arr[3] if len(arr) == 4 else 1.0
        return cls(r, g, b, a)
    
    @classmethod
    def from_rgb255(cls, r: int, g: int, b: int, alpha: int = 255):
        """Create a Color object from 0-255 RGB(A) values."""
        return cls(r / 255.0, g / 255.0, b / 255.0, alpha / 255.0)

    @property
    def red(self): return self.rgba[0]
    @red.setter
    def red(self, value): self.rgba[0] = clamp(value)
    @property
    def green(self): return self.rgba[1]
    @green.setter
    def green(self, value): self.rgba[1] = clamp(value)
    @property
    def blue(self): return self.rgba[2]
    @blue.setter
    def blue(self, value): self.rgba[2] = clamp(value)
    @property
    def alpha(self): return self.rgba[3]
    @alpha.setter
    def alpha(self, value): self.rgba[3] = clamp(value)

    @classmethod
    def attenuate(cls, factor: float):
        """Return a new Color attenuated by the given factor."""
        return Color(
            clamp(cls.red * factor),
            clamp(cls.green * factor),
            clamp(cls.blue * factor),
            cls.alpha
        )
    
    @classmethod
    def attenuate_distance_cof(cls, distance: float, a: float = 0.0, b: float = 0.0, c: float = 1.0):
        """
        Return a new Color attenuated by distance using quadratic attenuation.
        factor = 1 / (a*d^2 + b*d + c)
        """
        if c == 0 and a == 0 and b == 0:
            raise ValueError("Attenuation coefficients cannot all be zero")
        factor = 1.0 / (a * (distance ** 2) + b * distance + c)
        return cls.attenuate(factor)
    
    @classmethod
    def attenuate_diatance_max(cls, distance: float, max_distance: float):
        """
        Return a new Color attenuated linearly based on distance and max distance.
        factor = max(0, 1 - (distance / max_distance))
        """
        if max_distance <= 0:
            raise ValueError("max_distance must be greater than zero")
        factor = max(0.0, 1.0 - (distance / max_distance))
        return cls.attenuate(factor)

    def __add__(self, other):
        if type(other).__name__ == 'Color' or isinstance(other, Color):
             return Color(self.red + other.red, self.green + other.green, self.blue + other.blue, self.alpha + other.alpha)
        elif isinstance(other, (int, float, np.floating)):
             return Color(self.red + other, self.green + other, self.blue + other, self.alpha)
        raise TypeError("Unsupported operand type(s) for +: 'Color' and '{}'".format(type(other)))
    def __sub__(self, other):
        if type(other).__name__ == 'Color' or isinstance(other, Color):
             return Color(self.red - other.red, self.green - other.green, self.blue + other.blue, self.alpha + other.alpha)
        elif isinstance(other, (int, float, np.floating)):
             return Color(self.red - other, self.green - other, self.blue - other, self.alpha)
        raise TypeError("Unsupported operand type(s) for +: 'Color' and '{}'".format(type(other)))
    def __mul__(self, other):
        if type(other).__name__ == 'Color' or isinstance(other, Color):
            return Color(self.red * other.red, self.green * other.green, self.blue * other.blue, self.alpha * other.alpha)
        # Color * scalar -> scale all channels
        elif isinstance(other, (int, float, np.floating)):
            return Color(self.red * other, self.green * other, self.blue * other, self.alpha * other)
        raise TypeError("Unsupported operand type(s) for *: 'Color' and '{}'".format(type(other)))
    def __rmul__(self, other):
        if type(other).__name__ == 'Color' or isinstance(other, Color):
            return Color(self.red * other, self.green * other, self.blue * other, self.alpha * other)
        # allow Color * Color via commutativity (should be handled by __mul__ already)
        if isinstance(other, Color):
            return other.__mul__(self)
        raise TypeError("Unsupported operand type(s) for *: '{}' and 'Color'".format(type(other)))
    def __truediv__(self, other):
        # Color / scalar
        if isinstance(other, (int, float, np.floating)):
            return Color(self.red / other, self.green / other, self.blue / other, self.alpha / other)
        # Color / Color -> component-wise
        if isinstance(other, Color):
            return Color(self.red / other.red, self.green / other.green, self.blue / other.blue, self.alpha / other.alpha)
        raise TypeError("Unsupported operand type(s) for /: 'Color' and '{}'".format(type(other)))
    def __neg__(self):
        """Negate color (invert)."""
        return Color(1.0 - self.red, 1.0 - self.green, 1.0 - self.blue, self.alpha)
    def __eq__(self, other):
        return np.array_equal(self.rgba, other.rgba)
    def __iter__(self):
        yield from self.rgba
        
    def to_hex(self, include_alpha=True):
        r, g, b, a = (self.rgba * 255).astype(int)
        return f"#{r:02X}{g:02X}{b:02X}{a:02X}" if include_alpha else f"#{r:02X}{g:02X}{b:02X}"
    def __repr__(self):
        r, g, b, a = self.rgba
        return f"Color(r={r:.3f}, g={g:.3f}, b={b:.3f}, a={a:.3f})"

class Material:
    def __init__(self, color: Color, emissive: Color, roughness: float, glossiness: float, metallic: float, **kwargs):
        """
        A simple material that reflects LightRay based on its color, roughness, and glossiness.
        
        Attribute:
            color: The base color of the material.
            roughness: A value between 0.0 and 1.0 that determines
                how rough the surface is. 0.0 = perfectly smooth, 1.0 = very rough.
            glossiness: A value between 0.0 and 1.0 that determines
                how glossy the surface is. 0.0 = matte, 1.0 = perfectly glossy.
        """
        self.base_color = color
        self.emissive = emissive
        self.roughness = clamp(roughness)
        self.glossiness = clamp(glossiness)
        self.metallic = clamp(metallic)

        self.can_refract = False
        self.is_transparent = False
        self._rng = np.random.default_rng()
        self.name = "Material"

        for key, value in kwargs.items():
            setattr(self, key, value)

    def manipulate_color(self, color: Color) -> Color:
        """Manipulate the input color based on material properties."""
        # Combine diffuse, specular, emissive, and metallic components
        diffuse = self.get_diffuse_component(color)
        specular = self.get_specular_component(color)
        emissive = self.get_emissive_component()
        metallic = self.get_metallic_component(color)
        col = diffuse + specular + emissive + metallic
        return col.clamp()

    def calculate_optical_redirection(self, incoming_ray: Ray, surface_normal: np.ndarray, incoming_color: Color, new_origin: np.ndarray) -> tuple[Ray, Color]:
        """
        Compute the redirected (reflected or refracted) ray and its color after interaction with the material.
        Returns a LightRay with updated direction and color.
        """
        # Decide between reflection and refraction based on material properties
        if self.can_refract and self.is_transparent:
            # Refract the incoming ray orientation about the normal
            refracted_orientation = refract_ray(incoming_ray.orientation, surface_normal)
            new_orientation = refracted_orientation
        else:
            # Reflect the incoming ray orientation about the normal
            reflected_ray = Ray(*reflect_ray(surface_normal, incoming_ray.origin, incoming_ray.orientation))
            new_orientation = reflected_ray.orientation if hasattr(reflected_ray, 'orientation') else reflected_ray

        # Calculate the new color after material effect
        new_color = self.manipulate_color(incoming_color)

        # Create the new LightRay
        redirected_ray = Ray(
            origin=new_origin,
            orientation=new_orientation,
            name=f"{incoming_ray.name} (reflected)"
        )
        return redirected_ray, new_color

    def get_diffuse_component(self, incoming_color: Color) -> Color:
        """Get the diffuse component of the material response."""
        return self.base_color * incoming_color * (1.0 - self.glossiness)
    
    def get_specular_component(self, incoming_color: Color) -> Color:
        """Get the specular component of the material response."""
        return incoming_color * self.glossiness * (1.0 - self.roughness)
    
    def get_emissive_component(self) -> Color:
        """Get the emissive color of the material."""
        return self.emissive
    
    def get_metallic_component(self, incoming_color: Color) -> Color:
        """Get the metallic component of the material response."""
        return (1 - self.metallic) + self.base_color * incoming_color * self.metallic

    def __repr__(self):
        return (
            f"Material(color={self.base_color}, "
            f"roughness={self.roughness:.2f}, glossiness={self.glossiness:.2f}, emissive={self.emissive}, metallic={self.metallic:.2f})"
        )
    
class LightSource:
    def __init__(self, position: np.ndarray, color: Color, intensity: float = 1.0, name: str = "Light Source"):
        self.position = position
        self.color = color
        self.intensity = intensity
        self.name = name

    def __repr__(self):
        return (
            f"LightSource(position={np.round(self.position, 3)}, "
            f"color={self.color}, intensity={self.intensity:.2f})"
        )
"""
Luminance module: Provides classes for color representation, light rays, materials, and light sources.

Provides:
  - Color: RGBA color with conversions (hex, RGB255, array)
  - Material: Surface properties (roughness, glossiness) affecting light interaction
  - LightSource: Point light emitting rays with color and intensity
"""