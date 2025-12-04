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
    def use_hex(cls, hex_str: str):
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
    def use_array(cls, arr):
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

    def __add__(self, other):
        return Color(self.red + other.red, self.green + other.green, self.blue + other.blue)
    def __sub__(self, other):
        """Subtract two colors (component-wise)."""
        return Color(self.red - other.red, self.green - other.green, self.blue - other.blue, self.alpha)
    def __mul__(self, scalar):
        return Color(self.red * scalar, self.green * scalar, self.blue * scalar)
    def __rmul__(self, scalar):
        """Allow scalar * Color (reverse multiplication)."""
        return Color(self.red * scalar, self.green * scalar, self.blue * scalar, self.alpha)
    def __truediv__(self, other):
        return Color(self.red / other.red, self.green / other.green, self.blue / other.blue)
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

class LightRay(Ray, Color):
    def __init__(
        self,
        origin: np.ndarray,
        orientation : np.ndarray,
        color: Color,
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
        Color.__init__(self, color.red, color.green, color.blue, color.alpha)

        self.intensity = float(intensity)
        self.name = name  # keep name stored directly
    

    @classmethod
    def from_ray(cls, ray: Ray, color: "Color", intensity: float):
        """
        Helper constructor that preserves the Ray and ColorData instances passed in.
        Accepts signature used by tests: from_ray(ray, color, intensity)
        """
        # Ensure intensity is numeric and color is passed through
        try:
            intensity_f = float(intensity)
        except Exception as exc:
            raise TypeError(f"Invalid intensity value: {intensity}") from exc
        # Construct from an existing Ray and Color
        return cls(ray.origin.copy(), ray.orientation.copy(), color, intensity_f, ray.name)
    
    @classmethod
    def from_components(cls, origin, orientation , r, g, b, alpha=1.0, intensity=1.0, name="Light Ray"):
        """Create a LightRay from individual components."""
        color = Color(r, g, b, alpha)
        return cls(origin, orientation , color, intensity, name)
    
    @classmethod
    def from_hex(cls, origin, orientation , hex_str, intensity=1.0, name="Light Ray"):
        """Create a LightRay from a hex color string."""
        color = Color.use_hex(hex_str)
        return cls(origin, orientation , color, intensity, name)
    
    @classmethod
    def from_rgb255(cls, origin, orientation , r, g, b, alpha=255, intensity=1.0, name="Light Ray"):
        """Create a LightRay from 0-255 RGB(A) values."""
        color = Color.from_rgb255(r, g, b, alpha)
        return cls(origin, orientation , color, intensity, name)
    
    def AttenuateDistance(self, distance: float, a: float = 0.0, b: float = 0.0, c: float = 1.0) -> "LightRay":
        """
        Return a dimmed copy of the LightRay using quadratic distance attenuation.
        factor = 1 / (a*d^2 + b*d + c)
        """
        if c == 0 and a == 0 and b == 0:
            raise ValueError("Attenuation coefficients cannot all be zero")
        factor = 1.0 / (a * (distance ** 2) + b * distance + c)
        return LightRay(
            self.origin.copy(),
            self.orientation.copy(),
            self.base_color,
            intensity=self.intensity * factor,
            name=self.name + f" (X{factor:.2f})"
        )
    
    def AttenuateFactor(self, factor: float) -> "LightRay":
        """Return a dimmed copy of the LightRay by a multiplicative factor."""
        return LightRay(
            self.origin.copy(),
            self.orientation.copy(),
            self.base_color,
            intensity=self.intensity * factor,
            name=self.name + f" (X{factor:.2f})"
        )
    
    def Reflect(self, normal: np.ndarray) -> "LightRay":
        """Return a reflected copy of this LightRay about the given normal."""
        reflected_ray = reflect_ray(self.orientation, normal)
        reflected_orientation = reflected_ray.orientation if hasattr(reflected_ray, 'orientation') else reflected_ray
        return LightRay(
            self.origin.copy(),
            reflected_orientation,
            self.base_color,
            intensity=self.intensity,
            name=f"{self.name} (reflected)"
        )
    
    def Refract(self, normal: np.ndarray, eta: float = 1.0) -> "LightRay":
        """Return a refracted copy of this LightRay about the given normal."""
        refracted_orientation = refract_ray(self.orientation, normal)
        return LightRay(
            self.origin.copy(),
            refracted_orientation,
            self.base_color,
            intensity=self.intensity,
            name=f"{self.name} (refracted)"
        )
    
    @property
    def ray(self) -> Ray:
        """Return an independent Ray copy of this LightRay's geometric component."""
        return Ray(self.origin.copy(), self.orientation .copy(), self.name)
    
    @property
    def base_color(self) -> Color:
        """Return the color data of this LightRay without intensity scaling."""
        return Color(self.red, self.green, self.blue, self.alpha)
    
    @property
    def final_color(self) -> Color:
        """Return the effective color of this LightRay after intensity scaling."""
        return Color(
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
            f"color={self.base_color}, "
            f"intensity={self.intensity:.2f})"
        )

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
            
    def AffectColor(self, color: Color) -> Color:
        """Apply material attributes to a color."""
        col = self.base_color * color * self.glossiness # Specular component
        col += color * (1 - self.glossiness) * (1 - self.roughness) # Diffuse component

        col += self.emissive # Emissive component

        if self.metallic > 0:
            col *= (1 - self.metallic) + self.base_color * color * self.metallic # Metallic tint
        
        return col

    def RedirectLightRay(self, incoming_ray: LightRay, normal: np.ndarray) -> LightRay:
        """
        Compute the redirected (reflected or refracted) ray and its color after interaction with the material.
        Returns a LightRay with updated direction and color.
        """
        # Decide between reflection and refraction based on material properties
        if self.can_refract and self.is_transparent:
            # Refract the incoming ray orientation about the normal
            refracted_orientation = refract_ray(incoming_ray.orientation, normal)
            new_orientation = refracted_orientation
        else:
            # Reflect the incoming ray orientation about the normal
            reflected_ray = reflect_ray(incoming_ray.orientation, normal)
            new_orientation = reflected_ray.orientation if hasattr(reflected_ray, 'orientation') else reflected_ray

        # Calculate the new color after material effect
        new_color = self.AffectColor(incoming_ray.final_color)

        # Create the new LightRay
        redirected_ray = LightRay(
            origin=incoming_ray.origin.copy(),
            orientation=new_orientation,
            color=new_color,
            intensity=incoming_ray.intensity,
            name=f"{incoming_ray.name} (reflected)"
        )
        return redirected_ray

    def GetDiffuse(self, incoming_color: Color) -> Color:
        """Get the diffuse component of the material response."""
        return self.base_color * incoming_color * (1.0 - self.glossiness)
    
    def GetSpecular(self, incoming_color: Color) -> Color:
        """Get the specular component of the material response."""
        return incoming_color * self.glossiness * (1.0 - self.roughness)
    
    def GetEmissive(self) -> Color:
        """Get the emissive color of the material."""
        return self.emissive
    
    def GetMetallic(self, incoming_color: Color) -> Color:
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
    
    def LightRayReturn(self, incoming_ray: Ray) -> LightRay:
        """Generate a LightRay from this light source towards the incoming ray's origin."""
        direction = incoming_ray.origin - self.position
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Incoming ray origin cannot be the same as light source position.")
        orientation  = direction / norm

        return LightRay(
            origin=self.position.copy(),
            orientation=orientation ,
            color=self.color,
            intensity=self.intensity,
            name=f"{self.name} to {incoming_ray.name}"
        )

    def __repr__(self):
        return (
            f"LightSource(position={np.round(self.position, 3)}, "
            f"color={self.color}, intensity={self.intensity:.2f})"
        )
"""
Luminance module: Provides classes for color representation, light rays, materials, and light sources.

Provides:
  - Color: RGBA color with conversions (hex, RGB255, array)
  - LightRay: Ray + Color + intensity for light transport
  - Material: Surface properties (roughness, glossiness) affecting light interaction
  - LightSource: Point light emitting rays with color and intensity
"""