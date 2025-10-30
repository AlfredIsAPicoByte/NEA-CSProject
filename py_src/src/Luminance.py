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
    
    def clamp(self):
        """Clamp the RGBA values to be within [0.0, 1.0]."""
        self.rgba = np.clip(self.rgba, 0.0, 1.0)
        return self
    
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

    @classmethod
    def use_array(cls, arr):
        """Create a ColorData object from an array-like input."""
        if len(arr) not in (3, 4):
            raise ValueError("Array must have 3 (RGB) or 4 (RGBA) elements.")
        r, g, b = arr[0], arr[1], arr[2]
        a = arr[3] if len(arr) == 4 else 1.0
        return cls(r, g, b, a)
    
    @classmethod
    def from_rgb255(cls, r: int, g: int, b: int, alpha: int = 255):
        """Create a ColorData object from 0-255 RGB(A) values."""
        return cls(r / 255.0, g / 255.0, b / 255.0, alpha / 255.0)

    @property
    def red(self): return self.rgba[0]
    @red.setter
    def red(self, value: float):
        self.rgba[0] = clamp(value)
    @property
    def green(self): return self.rgba[1]
    @green.setter
    def green(self, value: float):
        self.rgba[1] = clamp(value)
    @property
    def blue(self): return self.rgba[2]
    @blue.setter
    def blue(self, value: float):
        self.rgba[2] = clamp(value)
    @property
    def alpha(self): return self.rgba[3]
    @alpha.setter
    def alpha(self, value: float):
        self.rgba[3] = clamp(value)

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
        orientation : np.ndarray,
        color: ColorData,
        intensity: float = 1.0,
        name: str = "Light Ray"
    ):
        # Normalize orientation  safely
        norm = np.linalg.norm(orientation)
        if norm == 0:
            raise ValueError("Orientation vector cannot be zero-length.")
        orient  = orientation  / norm

        # Initialize both parent classes
        Ray.__init__(self, origin, orient , name)
        ColorData.__init__(self, color.red, color.green, color.blue, color.alpha)

        self.intensity = float(intensity)
        self.name = name  # keep name stored directly
    
    @classmethod
    def from_ray(cls, ray: Ray, color: ColorData, intensity: float = 1.0, name: str = "Light Ray"):
        """Create a LightRay from an existing Ray and color."""
        return cls(ray.origin.copy(), ray.orientation .copy(), color, intensity, name)
    
    @classmethod
    def from_components(cls, origin, orientation , r, g, b, alpha=1.0, intensity=1.0, name="Light Ray"):
        """Create a LightRay from individual components."""
        color = ColorData(r, g, b, alpha)
        return cls(origin, orientation , color, intensity, name)
    
    @classmethod
    def from_hex(cls, origin, orientation , hex_str, intensity=1.0, name="Light Ray"):
        """Create a LightRay from a hex color string."""
        color = ColorData.use_hex(hex_str)
        return cls(origin, orientation , color, intensity, name)
    
    @classmethod
    def from_rgb255(cls, origin, orientation , r, g, b, alpha=255, intensity=1.0, name="Light Ray"):
        """Create a LightRay from 0-255 RGB(A) values."""
        color = ColorData.from_rgb255(r, g, b, alpha)
        return cls(origin, orientation , color, intensity, name)

    def Attenuate(self, distance: float, a: float = 0.0, b: float = 0.0, c: float = 1.0):
        """Return a dimmed copy of the LightRay."""
        factor = 1 / (a * (distance ** 2) + b * distance + c)
        return LightRay(
            self.origin,
            self.orientation ,
            self.rgba,
            intensity=self.intensity * factor,
            name=self.name + f" (X{factor})"
        )
    
    @property
    def ray(self) -> Ray:
        """Return an independent Ray copy of this LightRay's geometric component."""
        return Ray(self.origin.copy(), self.orientation .copy(), self.name)
    
    @property
    def base_color(self) -> ColorData:
        """Return the color data of this LightRay without intensity scaling."""
        return ColorData(self.red, self.green, self.blue, self.alpha)
    
    @property
    def final_color(self) -> ColorData:
        """Return the effective color of this LightRay after intensity scaling."""
        return ColorData(
            clamp(self.red * self.intensity),
            clamp(self.green * self.intensity),
            clamp(self.blue * self.intensity),
            self.alpha
        )

    def __repr__(self):
        return (
            f"LightRay("
            f"origin={np.round(self.origin, 3)}, "
            f"orientation ={np.round(self.orientation , 3)}, "
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
        self.base_color = color
        self.roughness = clamp(roughness)
        self.glossiness = clamp(glossiness)
        self.can_refract = False
        self.is_transparent = False
            
    def AffectColor(self, color: ColorData) -> ColorData:
        """Apply material attributes to a color."""
        return self.base_color * color * self.glossiness + color * (1 - self.glossiness) * (1 - self.roughness)

    def RedirectLightRay(self, incoming_ray: LightRay, normal: np.ndarray):
        """
        Compute the redirected (reflected) ray and its color after interaction with the material.
        Returns a tuple: (LightRay, ColorData)
        """
        # Decide between reflection and refraction based on material properties
        if self.can_refract and self.is_transparent:
            # Assume refract_ray returns the refracted orientation 
            refracted_ray = refract_ray(incoming_ray.orientation , normal)
            reflected = refracted_ray.orientation 
        else:
            # Reflect the incoming ray orientation  about the normal
            reflected_ray = reflect_ray(incoming_ray.orientation , normal)
            reflected = reflected_ray.orientation 

        # Calculate the new color after material effect
        new_color = self.AffectColor(incoming_ray.final_color)

        # Create the new LightRay
        redirected_ray = LightRay(
            origin=reflected.origin,
            orientation =reflected.orientation ,
            color=new_color,
            intensity=incoming_ray.intensity,
            name=f"{incoming_ray.name} (reflected)"
        )
        return redirected_ray

    def __repr__(self):
        return (
            f"Material(color={self.base_color}, "
            f"roughness={self.roughness:.2f}, glossiness={self.glossiness:.2f})"
        )