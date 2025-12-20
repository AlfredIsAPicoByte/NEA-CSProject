import numpy as np
from typing import Callable, List, Optional
from PrimaryStructures import Ray
from Reflections import reflect_ray
from Refractions import refract_ray, calculate_critical_angle

"""

"""

def clamp(value, min_value: float|int = 0.0, max_value: float|int = 1.0):
    return max(min_value, min(value, max_value))

class Color:
    def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0, clamp=False):
        # Robust initialization: Handle case where 'r' is actually a list/array/Color
        if hasattr(r, "__len__") and len(r) == 3:
            # If input is a tuple/list/array (e.g. from numpy)
            self.rgba = np.array([float(r[0]), float(r[1]), float(r[2]), float(r[3])], dtype=np.float32)
        elif hasattr(r, "rgba"):
            # If input is another Color object (Copy Constructor)
            self.rgba = np.array(r.rgba, dtype=np.float32)
        else:
            # Standard initialization
            self.rgba = np.array([float(r), float(g), float(b), float(a)], dtype=np.float32)

        if clamp:
            self.clamp()
    
    def clamp(self):
        """Clamp the RGBA values to be within [0.0, 1.0]."""
        self.rgba = np.clip(self.rgba, 0.0, 1.0)
        return self
    
    @classmethod
    def from_hex(cls, hex_str: str, clamp: bool = False):
        """Create a Color object from a hex string."""
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) not in (6, 8):
            raise ValueError("Hex color must be 6 (RGB) or 8 (RGBA) characters long.")

        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = int(hex_str[6:8], 16) / 255.0 if len(hex_str) == 8 else 1.0

        rgba = np.array([r, g, b, a], dtype=float)
        return cls(*rgba, clamp)

    @classmethod
    def from_array(cls, arr, clamp: bool = False):
        """Create a Color object from an array-like input."""
        if len(arr) not in (3, 4):
            raise ValueError("Array must have 3 (RGB) or 4 (RGBA) elements.")
        r, g, b = arr[0], arr[1], arr[2]
        a = arr[3] if len(arr) == 4 else 1.0
        return cls(r, g, b, a, clamp)
    
    @classmethod
    def from_rgb255(cls, red: int = 0, green: int = 0, blue: int = 0, alpha: int = 255, clamp: bool = True):
        """Create a Color object from 0-255 RGB(A) values."""
        return cls(red / 255.0, green / 255.0, blue / 255.0, alpha / 255.0, clamp)
    
    @staticmethod
    def average_colors(colors: List['Color']) -> 'Color':
        if not colors:
            return Color() # Return black if list is empty
            
        sum_r, sum_g, sum_b, sum_a = 0.0, 0.0, 0.0, 0.0
        
        for c in colors:
            sum_r += c.red
            sum_g += c.green
            sum_b += c.blue
            sum_a += c.alpha
            
        N = len(colors)
        return Color(
            sum_r / N,
            sum_g / N,
            sum_b / N,
            sum_a / N
        )

    @property
    def red(self): return self.rgba[0]
    @red.setter
    def red(self, value): self.rgba[0] = value
    @property
    def green(self): return self.rgba[1]
    @green.setter
    def green(self, value): self.rgba[1] = value
    @property
    def blue(self): return self.rgba[2]
    @blue.setter
    def blue(self, value): self.rgba[2] = value
    @property
    def alpha(self): return self.rgba[3]
    @alpha.setter
    def alpha(self, value): self.rgba[3] = value

    # Legacy alias support (fixes "object has no attribute 'r'")
    @property
    def r(self): return self.rgba[0]
    @property
    def g(self): return self.rgba[1]
    @property
    def b(self): return self.rgba[2]
    @property
    def a(self): return self.rgba[3]
    
    @property
    def components(self): return self.rgba

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
        # 1. Color * Color (Hadamard product - this is where your error lies)
        if isinstance(other, Color):
            # Ensure the attributes (red, green, blue, alpha) are correctly accessed
            return Color(self.red * other.red, self.green * other.green, 
                         self.blue * other.blue, self.alpha * other.alpha)
        
        # 2. Color * scalar 
        elif isinstance(other, (int, float, np.floating)):
            return Color(self.red * other, self.green * other, 
                         self.blue * other, self.alpha * other)
        
        # 3. Color * list
        elif isinstance(other, (list, tuple, set)):
            return Color(
                self.red * other[0],
                self.green * other[1],
                self.blue * other[2],
                self.alpha * other[3] 
            )
        
        # 4. Check for "Color-like" object (Duck Typing)
        # We try to access .red, .green, .blue. If it fails, it's not a color.
        try:
            return Color(self.red * other.red, self.green * other.green, 
                         self.blue * other.blue, self.alpha * other.alpha)
        except AttributeError:
            pass # Not a color-like object, continue to error
                         
        # If the operand is neither a Color nor a scalar, raise an error
        raise TypeError("Unsupported operand type(s) for *: 'Color' and '{}'".format(type(other).__name__))
    def __rmul__(self, other):
        """
        Handles reverse multiplication: scalar * Color.
        Delegates the work to __mul__ to reuse the correct logic.
        """
        # Since multiplication is commutative (scalar * vector = vector * scalar),
        # we can simply call the forward multiplication method.
        # This will correctly handle the case where 'other' is a scalar.
        return self.__mul__(other)
        
        # Note: We do NOT need the checks for 'Color' types in __rmul__ 
        # because the method is only called if the left operand (other) 
        # doesn't support the operation, which is typically true for scalars.
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
    def to_np_ndarray(self):
        return self.rgba.copy()
    def to_rgb255(self):
        r, g, b, a = (self.rgba * 255).astype(int)
        return (r, g, b)
    
    
    def __repr__(self):
        return f"Color(red={self.red:.2f}, green={self.green:.2f}, blue={self.blue:.2f}, alpha={self.alpha:.2f})"

class ColorGradient:
    def __init__(self, colors: List[Color], positions: List[float]):
        if len(colors) != len(positions):
            raise ValueError("Colors and positions must have the same length.")
        if any(p < 0.0 or p > 1.0 for p in positions):
            raise ValueError("Positions must be in the range [0.0, 1.0].")
        if sorted(positions) != positions:
            raise ValueError("Positions must be in ascending order.")
        
        self.colors = colors
        self.positions = positions
    
    def get_color(self, t: float) -> Color:
        """Get interpolated color at position t in [0.0, 1.0]."""
        t = clamp(t)
        for i in range(1, len(self.positions)):
            if t <= self.positions[i]:
                t0, t1 = self.positions[i-1], self.positions[i]
                c0, c1 = self.colors[i-1], self.colors[i]
                factor = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                r = c0.red + factor * (c1.red - c0.red)
                g = c0.green + factor * (c1.green - c0.green)
                b = c0.blue + factor * (c1.blue - c0.blue)
                a = c0.alpha + factor * (c1.alpha - c0.alpha)
                return Color(r, g, b, a)
        return self.colors[-1]
        
class LightSource:
    def __init__(self, position: np.ndarray, color: Color, intensity: float = 1.0, radius: float = 1.0, name: str = "Light Source"):
        self.position = position
        self.color = color
        self.intensity = intensity
        self.radius = radius
        self.name = name

    def get_light_direction(self, hit_point: np.ndarray) -> np.ndarray:
        """Return the normalized direction vector from the light source to the hit point."""
        direction = hit_point - self.position
        norm = np.linalg.norm(direction)
        if norm == 0:
            return direction  # Zero vector if at the same point
        return direction / norm
    
    @classmethod
    def attenuate_color(cls, color, factor: float) -> Color:
        """Return a new Color attenuated by the given factor."""
        # Note: 'color' is used to access the specific instance properties
        return Color(
            clamp(color.red * factor),
            clamp(color.green * factor),
            clamp(color.blue * factor),
            color.alpha
        )
    
    @classmethod
    def attenuate_distance_cof(cls, distance: float, a: float = 0.0, b: float = 0.0, c: float = 1.0) -> float:
        """
        Return a factor attenuated by distance using quadratic attenuation.
        factor = 1 / (a*d^2 + b*d + c)
        """
        if c == 0 and a == 0 and b == 0:
            raise ValueError("Attenuation coefficients cannot all be zero")
        factor = 1.0 / (a * (distance ** 2) + b * distance + c)
        return factor
    
    @classmethod
    def attenuate_distance_max(cls, distance: float, max_distance: float) -> float:
        """
        Return a factor attenuated linearly based on distance and max distance.
        factor = max(0, 1 - (distance / max_distance))
        """
        if max_distance <= 0:
            raise ValueError("max_distance must be greater than zero")
        factor = max(0.0, 1.0 - (distance / max_distance))
        return factor
    
    @classmethod
    def attenuate_sqr_distance(cls, distance: float) -> float:
        """
        Return a factor attenuated linearly based on distance squared.
        factor = 1 / ((distance ** 2) + 1e-6)
        """
        factor = 1 / ((distance ** 2) + 1e-6)
        return factor

    def get_final_color(self, distance: float, attenuation_type: str = "distance_cof", **kwargs) -> Color:
        """Return the final color of the light source after applying attenuation based on distance."""
        if attenuation_type == "distance_cof":
            return self.attenuate_distance_cof(self.color, distance, **kwargs)
        elif attenuation_type == "distance_max":
            return self.attenuate_distance_max(self.color, distance, **kwargs)
        else:
            raise ValueError(f"Unknown attenuation type: {attenuation_type}")

    def __repr__(self):
        return (
            f"LightSource(position={np.round(self.position, 3)}, "
            f"color={self.color}, intensity={self.intensity:.2f})"
        )

class Material:
    def __init__(self, color: Color, roughness: float = 0.5, glossiness: float = 0.5, metallic: float = 0.0, ior: float = 1.0, emissive: Optional[Color] = None, emissive_intensity: float = 1.0):
        # Ensure base_color is stored as a Color object
        self.base_color = color if isinstance(color, Color) else Color(color)
        
        # Handle emissive defaulting
        if emissive is None:
            self.emissive_color = Color(0, 0, 0)
        else:
            self.emissive_color = emissive if isinstance(emissive, Color) else Color(emissive)
            
        self.roughness = roughness
        self.glossiness = glossiness
        self.metallic = metallic
        self.ior = ior
        self.emissive_intensity = emissive_intensity
        
        # Flags
        self.is_transparent = False
        self.can_refract = False

    def apply_material_color(self, light_sources: List[LightSource], hit_point: np.ndarray, normal: np.ndarray, view_dir: np.ndarray, ambient_color: Color, visibility_function: Callable) -> Color:
        """
        Calculates the full color of the material by iterating over all lights in the scene.
        """
        final_color = Color(0.0, 0.0, 0.0, 1.0)
        
        for light in light_sources:
            # 1. Geometry & Attenuation
            light_vec = light.position - hit_point
            light_dist = np.linalg.norm(light_vec)
            
            if light_dist <= 1e-6: continue # too close

            light_dir = light_vec / light_dist
            
            NdotL = max(0.0, np.dot(normal, light_dir))
            if NdotL <= 0.0: continue # behind the surface

            attenuation = LightSource.attenuate_sqr_distance(light_dist)
            light_intensity = light.intensity * attenuation

            # 2. Visibility / Shadows (Callback to the renderer/strategy)
            # We pass the light info back to the strategy's visibility function
            visibility = visibility_function(light, light_dir, light_dist)
            
            if visibility <= 0.0: continue

            # 3. PBR Components
            diffuse = self.get_diffuse_component(normal, light_dir, light_intensity)
            specular = self.get_specular_component(normal, light_dir, light_intensity, view_dir)
            
            # 4. Accumulate
            final_color += (diffuse + specular) * light.color * visibility

        # Add in ambient color 
        final_color += ambient_color * self.base_color

        # Add Emissive (Self-illumination)
        final_color += self.get_emissive_component()
        
        return final_color.clamp()

    def calculate_optical_redirection(self, incoming_ray: Ray, surface_normal: np.ndarray, 
                                  incoming_color: Color, hit_point: np.ndarray, bias: float = 1e-4) -> tuple[Ray, Color]:
        unit_dir = incoming_ray.orientation / np.linalg.norm(incoming_ray.orientation)
        unit_normal = surface_normal / np.linalg.norm(surface_normal)

        final_dir = reflect_ray(unit_normal, incoming_ray.orientation)
        final_origin = hit_point + (unit_normal * bias)

        if self.can_refract and self.is_transparent:
            ior_air = 1.0
            ior_material = getattr(self, 'ior', 1.5)

            # Check if entering or exiting
            cos_i = -np.dot(unit_normal, unit_dir)
            if cos_i > 0:
                # Entering the material (Air -> Glass)
                n1, n2 = ior_air, ior_material
                normal = unit_normal
            else:
                # Exiting the material (Glass -> Air)
                n1, n2 = ior_material, ior_air
                normal = -unit_normal # Flip normal to point into the new medium
            
            try:
                final_dir = refract_ray(normal, hit_point, incoming_ray.direction, n1, n2)
                final_origin = hit_point - (normal * bias)
            except ValueError:
                pass

        attenuation = LightSource.attenuate_color(self.base_color + incoming_color, LightSource.attenuate_sqr_distance(np.linalg.norm(incoming_ray.origin - hit_point)))
        attenuation += self.get_emissive_component()

        # If it's a metal, the attenuation matches the reflection color strongly
        # If it's glass, it might be clear (white attenuation) or tinted
        
        # Create the new ray
        redirected_ray = Ray(
            origin=final_origin,
            orientation=final_dir,
            name=f"{incoming_ray.name}_bounce"
        )
        
        return redirected_ray, attenuation

    def get_diffuse_component(self, surface_normal: np.ndarray, light_dir: np.ndarray, light_intensity: float) -> Color:
        """Get the diffuse component of the material response."""
        diffuse = self.base_color * light_intensity * max(0, np.dot(surface_normal, light_dir))
        return diffuse
    
    def get_specular_component(self, surface_normal: np.ndarray, light_dir: np.ndarray, light_intensity: float, view_dir: np.ndarray) -> Color:
        """Get the specular component of the material response using the Micro-Facet BRDF."""
        
        # --- 0. Pre-Calculations and Constants ---
        
        alpha = self.roughness
        alpha_sq = alpha * alpha
        
        # Halfway Vector (H)
        H = (light_dir + view_dir)
        H = H / np.linalg.norm(H) 
        
        # Dot Products (must be clamped to avoid negative light/view angles)
        NdotH = max(0.0, np.dot(surface_normal, H))
        NdotL = max(0.0, np.dot(surface_normal, light_dir))
        NdotV = max(0.0, np.dot(surface_normal, view_dir))
        VdotH = max(0.0, np.dot(view_dir, H))

        # --- 1. Normal Distribution Function (NDF - GGX) ---
        denom_ndf = (NdotH * NdotH * (alpha_sq - 1.0) + 1.0)
        NDF = alpha_sq / (np.pi * denom_ndf * denom_ndf)
        
        # --- 2. Geometric Shadowing Function (GSF - Schlick-GGX Approximation) ---
        k = (alpha + 1.0) / 2.0
        GS_Schlick = lambda n_dot_k: n_dot_k / (n_dot_k * (1.0 - k) + k)
        GSF = GS_Schlick(NdotL) * GS_Schlick(NdotV)
        
        # --- 3. Fresnel Function (FF - Schlick Approximation) ---
        
        # REUSED COMPONENT: Calculate F0 using the dedicated function
        F0 = self.get_metallic_component()
        
        # F_schlick calculation (Color operations are handled correctly)
        term_pow5 = (1.0 - VdotH) ** 5 
        FF = F0 + (Color(1.0, 1.0, 1.0) - F0) * term_pow5

        # --- 4. Final BRDF Term (Fs) and Specular Contribution ---
        
        # Denominator of the BRDF term
        denom_fs = 4.0 * NdotL * NdotV 
        
        if denom_fs > 0:
            # Fs is the specular BRDF (Fs = D * G * F / denominator)
            Fs = (NDF * GSF * FF) * (1.0 / denom_fs) 
        else:
            Fs = Color(0.0, 0.0, 0.0) 

        # Final Specular Color: Light Intensity * BRDF * Cosine Term
        specular = light_intensity * Fs * NdotL 
        
        return specular
    
    def get_emissive_component(self) -> Color:
        """Get the emissive color of the material."""
        return self.emissive_color * getattr(self, "emissive_intensity", 1)
    
    def get_metallic_component(self) -> Color:
        """
        Calculates the F0 (Base Reflectivity) vector, blending between 
        dielectric (non-metal) and metallic properties.
        """
        # F_dielectric is the standard 4% reflection at normal incidence
        # for non-metals (represented as an RGB color vector).
        F_dielectric = Color(0.04, 0.04, 0.04)
        
        # F0 is a linear interpolation (Lerp) controlled by self.metallic:
        # If metallic = 0, F0 = F_dielectric
        # If metallic = 1, F0 = self.base_color
        F0 = F_dielectric * (1.0 - self.metallic) + self.base_color * self.metallic
        
        return F0

    def __repr__(self):
        return (
            f"Material(color={self.base_color}, "
            f"roughness={self.roughness:.2f}, glossiness={self.glossiness:.2f}, emissive={self.emissive_color}, metallic={self.metallic:.2f})"
        )
"""
Luminance module: Provides classes for color representation, light rays, materials, and light sources.

Provides:
  - Color: RGBA color with conversions (hex, RGB255, array)
  - Material: Surface properties (roughness, glossiness) affecting light interaction
  - LightSource: Point light emitting rays with color and intensity
"""