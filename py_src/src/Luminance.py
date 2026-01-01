from enum import Enum
import numpy as np
from typing import Callable, List, Tuple, Optional

from PrimaryStructures import Ray, HitInfo
from Reflections import calculate_surface_reflection_ray
from Refractions import calculate_refraction_vector, calculate_reflectance, REFRACTIVE_INDICES
from Sampling import Sampler
import logging

def clamp(value, min_value: float|int = 0.0, max_value: float|int = 1.0):
    return max(min_value, min(value, max_value))

def lerp(a, b, t):
    return a + (b - a) * t

def attenuate_distance_cof(distance: float, a: float = 0.0, b: float = 0.0, c: float = 1.0) -> float:
    """
    Return a factor attenuated by distance using quadratic attenuation.
    factor = 1 / (a*d^2 + b*d + c)
    """
    if c == 0 and a == 0 and b == 0:
        raise ValueError("Attenuation coefficients cannot all be zero")
    factor = 1.0 / (a * (distance ** 2) + b * distance + c)
    return factor

def attenuate_distance_max(distance: float, max_distance: float) -> float:
    """
    Return a factor attenuated linearly based on distance and max distance.
    factor = max(0, 1 - (distance / max_distance))
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be greater than zero")
    factor = max(0.0, 1.0 - (distance / max_distance))
    return factor

def attenuate_sqr_distance(distance: float) -> float:
    """
    Return a factor attenuated linearly based on distance squared.
    factor = 1 / ((distance ** 2) + 1e-6)
    """
    factor = 1 / ((distance ** 2) + 1e-6)
    return factor

def attenuate_distance_exponential(distance: float, decay_rate: float = 1.0) -> float:
    """
    Return a factor attenuated exponentially based on distance.
    factor = exp(-decay_rate * distance)
    """
    factor = np.exp(-decay_rate * distance)
    return factor

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


def attenuate_color(color, factor: float) -> Color:
    """Return a new Color attenuated by the given factor."""
    # Note: 'color' is used to access the specific instance properties
    return Color(
        clamp(color.red * factor),
        clamp(color.green * factor),
        clamp(color.blue * factor),
        color.alpha
    )

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

    def get_color(
            self,
            t: float,
            interpolation_function: Callable[[float], float] = lambda x: x # Linear by default
        ) -> Color:
        """Get interpolated color at position t in [0.0, 1.0]."""
        t = clamp(t)
        for i in range(1, len(self.positions)):
            if t <= self.positions[i]:
                t0, t1 = self.positions[i-1], self.positions[i]
                c0, c1 = self.colors[i-1], self.colors[i]

                factor = interpolation_function((t - t0) / (t1 - t0) if t1 > t0 else 0.0)

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

    def get_light_direction(self, hit_point: np.ndarray, bias: float = 1e-4) -> np.ndarray:
        """Return the normalized direction vector from the hit point towards the light source."""
        direction = self.position - hit_point
        norm = np.linalg.norm(direction)
        
        return direction / (norm + bias)

    def __repr__(self):
        return (
            f"LightSource(position={np.round(self.position, 3)}, "
            f"color={self.color}, intensity={self.intensity:.2f})"
        )

class MaterialType(Enum):
    DIFFUSE = 1 # Only diffuse reflections
    SPECULAR = 2 # Includes specular reflections and metallic properties
    GLASS = 3 # Uses refraction and reflection based on IOR
    TRANSPARENT = 4 # No surface, only transparency and albedo tint
    EMISSIVE = 5 # Self-illuminating material, no light calculations

class Material:
    def __init__(
            self,
            albedo_color: Color = Color(1, 1, 1),
            roughness: float = 0.5,
            metallicness: float = 0.0,
            specular_intensity: float = 0.0,
            specular_tint_amount: float = 0.0,
            emissive_intensity: float = 1.0,
            ior: float = REFRACTIVE_INDICES["glass"],
            absorption_color: Color = Color(1.0, 0.1, 1.0),
            absorption_strength = 0.5,
            is_transparent: bool = False,
            opacity: float = 1.0,
            type: MaterialType = MaterialType.DIFFUSE,
        ):
        # Ensure colors are stored as a Color objects
        self.albedo = albedo_color if isinstance(albedo_color, Color) else Color(albedo_color)
        self.absorption_color = absorption_color if isinstance(absorption_color, Color) else Color(absorption_color)
        
        # Surface properties
        self.roughness = clamp(roughness, 0.0, 1.0)
        self.metallic = clamp(metallicness, 0.0, 1.0)
        self.specular_intensity = clamp(specular_intensity, 0.0, 1.0)
        self.specular_tint_amount = clamp(specular_tint_amount, 0.0, 1.0)
        self.absorption_strength = clamp(absorption_strength, 0.0, 1.0)
        self.opacity = clamp(opacity, 0.0, 1.0)
        self.emissive_intensity = emissive_intensity
        self.ior = ior

        # Flags
        self.is_transparent = is_transparent
        self.type = type

    @classmethod
    def create_diffuse(cls, albedo: Color, roughness: float = 0.5):
        """
        Creates a standard non-metallic (dielectric) material like plastic, wood, or chalk.
        """
        return cls(
            albedo_color=albedo,
            roughness=roughness,
            metallicness=0
        )

    @classmethod
    def create_specular(cls, albedo: Color, roughness: float = 0.2, metallicness: float = 1.0, specular_intensity: float = 1.0, specular_tint_amount: float = 0.5):
        """
        Creates a reflective material like gold, aluminum, or copper.
        """
        return cls(
            albedo_color=albedo,
            roughness=roughness,
            metallicness=metallicness,
            specular_intensity=specular_intensity,
            specular_tint_amount=specular_tint_amount,
            type=MaterialType.SPECULAR
        )
    
    @classmethod
    def create_glass(cls, albedo: Color, absorption_color: Color, roughness: float = 0.0, metallicness: float = 0.0, ior: float = 1.5, absorption_strength: float = 1):
        """
        Creates a dielectric transparent material.
        """
        return cls(
            albedo_color=albedo,
            roughness=roughness,
            metallicness=metallicness,
            ior=ior,
            absorption_color=absorption_color,
            absorption_strength=absorption_strength,
            is_transparent=True,
            type=MaterialType.GLASS
        )
    
    @classmethod
    def create_transparent(cls, albedo: Color):
        """
        Creates a 'thin' transparent material (alpha blending), like a ghost or a hologram.
        This does not use IOR/Refraction logic, just simple opacity.
        """

        return cls(
            albedo_color=albedo,
            is_transparent=True,
            type=MaterialType.TRANSPARENT
        )
    
    @classmethod
    def create_emissive(cls, albedo: Color, intensity: float):
        """
        Creates a self-illuminated material that ignores.
        """
        return cls(
            albedo_color=albedo,
            emissive_intensity=intensity,
            type=MaterialType.EMISSIVE,
        )

    def apply_material(
            self,
            scene_lights: List[LightSource],
            hit_info: HitInfo,
            view_dir: np.ndarray,
            visibility_function: Callable[[np.ndarray, np.ndarray], float],
            bias: float = 1e-4
        ) -> Color:
        """
        Calculates the full color of the material by iterating over all lights in the scene.
        """
        if self.type == MaterialType.EMISSIVE:
            return self.get_emissive_component()
        
        surface_normal = hit_info.normal
        hit_point = hit_info.point

        direct_light_color = Color(0, 0, 0)
        for light in scene_lights:
            light_dir = light.get_light_direction(hit_point)
            light_dist = np.linalg.norm(light.position - hit_point)

            if light_dist <= bias:
                continue # Light is too close or at the hit point
            
            # Determine if the ray is escaping or entering the surface
            NdotL = max(0.0, np.dot(surface_normal, light_dir))
            if NdotL <= 0.0 and self.type != MaterialType.TRANSPARENT and self.type != MaterialType.GLASS:
                continue # behind the surface, ignore for glass and transparent materials
            
            light_attenuation = attenuate_sqr_distance(light_dist) # Using inverse square law for point lights
            light_intensity = light.intensity * light_attenuation

            if self.type == MaterialType.TRANSPARENT:
                visibility = 1.0
            else:
                visibility = visibility_function(hit_point, light.position)

            if visibility <= 0.0:
                continue # Light is fully blocked

            # Material response based on type
            if self.type == MaterialType.DIFFUSE:
                diffuse = self.get_diffuse_component(light.color, light_intensity * visibility, light_dir, surface_normal) 
                direct_light_color += diffuse

            if self.type == MaterialType.SPECULAR:
                diffuse = self.get_diffuse_component(light.color, light_intensity * visibility, light_dir, surface_normal)
                specular = self.get_specular_component(light.color, light_intensity * visibility, light_dir, surface_normal, view_dir)
                direct_light_color += diffuse + specular

            if self.type == MaterialType.GLASS:
                specular = self.get_specular_component(light.color, light_intensity * visibility, light_dir, surface_normal, view_dir)
                direct_light_color += specular

        final_color = direct_light_color
        
        return final_color

    def get_diffuse_component(self, light_color: Color, light_intensity: float, light_dir: np.ndarray, surface_normal: np.ndarray) -> Color:
        """
        Get the diffuse component (Lambertian). 
        Corrected to respect the Metallic workflow energy conservation.
        """
        NdotL = max(0.0, np.dot(surface_normal, light_dir))
        
        # 1. Standard Lambert Diffuse (Simple but physically consistent)
        diffuse = light_intensity * self.albedo * NdotL
        
        # 2. ENERGY CONSERVATION: Metals have NO diffuse.
        # As metallic approaches 1.0, diffuse must approach 0.0.
        diffuse = light_color * diffuse * (1.0 - self.metallic)
        
        return diffuse

    def get_specular_component(self, light_color: Color, light_intensity: float, light_dir: np.ndarray, surface_normal: np.ndarray, view_dir: np.ndarray, bias: float = 1e-4) -> Color:
        """Get the specular component of the material response using the Micro-Facet BRDF."""
        
        # --- 0. Pre-Calculations and Constants ---
        safe_roughness = max(self.roughness, bias)
        alpha = safe_roughness ** 2
        alpha_sq = alpha ** 2
        
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
        k = ((alpha + 1.0) ** 2) / 8.0 
        
        GS_Schlick = lambda n_dot_k: n_dot_k / (n_dot_k * (1.0 - k) + k)
        GSF = GS_Schlick(NdotL) * GS_Schlick(NdotV)
        
        # --- 3. Fresnel Function (FF - Schlick Approximation) ---
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
        specular = light_color * Fs * NdotL * light_intensity * self.specular_intensity
        
        return specular

    def get_metallic_component(self, bias: float = 1e-4) -> Color:
        """
        Calculates the F0 (Base Reflectivity) for Fresnel calculations.
        """
        # --- 1. Dielectric (Non-Metal) F0 ---
        # Base reflectivity for plastic/glass/water is approx 0.04 (Linear Grey)
        F0_dielectric = Color(0.04, 0.04, 0.04)
        
        # If the material is very dark, we don't want the F0 to drop to 0.
        # So we ensure the tinted version maintains some brightness if needed, 
        # but usually just using Albedo is fine for the tint *color*.
        # For correct magnitude, we just lerp the *color* of the 0.04 base.
        
        # Lerp between Grey (0.04, 0.04, 0.04) and Tinted (0.04 * AlbedoColor)
        # Note: We normalize albedo to preserve the 0.04 intensity magnitude.
        max_val = max(self.albedo.r, self.albedo.g, self.albedo.b, bias)
        albedo_normalized = self.albedo / max_val
        
        F0_dielectric_tinted = F0_dielectric * albedo_normalized
        
        # Apply the Specular Tint Slider
        F0_final_dielectric = lerp(F0_dielectric, F0_dielectric_tinted, self.specular_tint_amount)

        # --- 2. Metallic F0 ---
        # Metals use their Albedo directly as F0
        F0_metal = self.albedo

        # --- 3. Final Blend ---
        # Lerp between Dielectric (0.04) and Metal (Albedo)
        final_F0 = lerp(F0_final_dielectric, F0_metal, self.metallic)
        
        return final_F0

    def get_emissive_component(self) -> Color:
        """Get the emissive color of the material."""
        return self.albedo * self.emissive_intensity

    def get_volumetric_component(self, light_color: Color, distance: float) -> Color:
        """
        Calculates the volumetric attenuation of light passing through the material
        """
        sigma = (np.array([1.0, 1.0, 1.0]) - self.absorption_color) * self.absorption_strength
        
        # Calculate attenuation per channel
        # e.g. If Red has high sigma, exp(-high * dist) -> 0.0 (Red is blocked)
        attenuation = attenuate_distance_exponential(distance, sigma)
        
        return light_color * attenuation

    def get_transparency_component(self, light_color: Color) -> Color:
        """
        Calculates the transparency color contribution of the material.
        """
        return lerp(light_color, self.albedo, 1 - self.albedo.a)

    def apply_ambient_color(self, ambient_color: Color, ambient_intensity: float) -> Color:
        """
        Calculates the ambient color contribution of the material.
        """
        ambient = ambient_color * ambient_intensity * self.albedo
        return ambient

    def evaluate_brdf(self, normal: np.ndarray, view_dir: np.ndarray, light_dir: np.ndarray, bias: float = 1e-4) -> Color:
        """
        Returns the BRDF value (ratio of radiance). 
        Does NOT include NdotL or Light Intensity.
        """
        V = view_dir
        L = light_dir
        
        H = (V + L)
        # Safety check for degenerate vectors
        h_len = np.linalg.norm(H)
        if h_len < bias:
            return Color(0, 0, 0)
        H = H / h_len

        # Clamp dot products to avoid negative values (lighting from behind surface)
        VdotH = max(np.dot(V, H), bias)

        # F (Fresnel - Schlick)
        F0 = self.get_metallic_component()
        
        # Compute Fresnel Term (F is a Color because F0 is a Color)
        F_color = Color.from_array(schlick_fresnel(VdotH, F0.to_np_ndarray()))

        # Compute PBR Terms
        specular_term = self.get_specular_component(F_color, 1, light_dir, normal, view_dir)
        diffuse_term = self.get_diffuse_component(F_color, 1, light_dir, normal) * (1.0/np.pi)

        return diffuse_term + specular_term

    def __repr__(self):
        return (
            f"Material()"
        )
    
def schlick_fresnel(cos_theta: float, f0: np.ndarray) -> np.ndarray:
    """
    Calculates the portion of light that is reflected (Specular) vs. absorbed/refracted (Diffuse).
    
    Args:
        cos_theta: The dot product of View Vector and Surface Normal (N dot V).
                Must be clamped between 0.0 and 1.0.
        f0: The base reflectivity of the material at 0 degrees incidence.
            For non-metals (dielectrics), this is usually constant (e.g., 0.04).
            For metals, this is the surface color itself.
    
    Returns:
        The reflection coefficient (F), a value between 0.0 and 1.0.
    """
    return f0 + (1.0 - f0) * ((1.0 - cos_theta) ** 5)

def calculate_redirection_ray(
        incoming_ray: Ray,
        surface_normal: np.ndarray,
        new_origin: np.ndarray,
        roughness: float,
        sampler: Sampler,
        refractive_index_incident: float = REFRACTIVE_INDICES['air'],
        refactive_index: float = REFRACTIVE_INDICES['glass'],
        seed: Optional[int] = None,
        bias: float = 1e-4
    ) -> Tuple[Ray, float, bool, bool]:
    rng = np.random.default_rng(seed)

    unit_dir = incoming_ray.orientation / np.linalg.norm(incoming_ray.orientation)
    unit_normal = surface_normal / np.linalg.norm(surface_normal)

    NdotL = np.dot(surface_normal, incoming_ray.orientation)

    reflectance = calculate_reflectance(
        np.degrees(np.arccos(-np.dot(unit_normal, unit_dir))),
        refractive_index_incident,
        refactive_index,
    )

    if reflectance is None:
        # Total internal reflection occurred
        logging.debug("Total internal reflection (no refracted vector)")
        return *calculate_surface_reflection_ray(unit_normal, unit_dir, new_origin, roughness, sampler), True, NdotL >= 0

    # Decide between reflection and refraction based on reflectance
    if rng.random() < reflectance:
        # Reflect
        logging.debug("Redirecting: chosen reflection by Fresnel")
        return *calculate_surface_reflection_ray(unit_normal, unit_dir, new_origin, roughness, sampler), True, NdotL >= 0
    
    # Refract
    logging.debug("Redirecting: chosen refraction path")
    refracted_vector = calculate_refraction_vector(unit_normal, unit_dir, refractive_index_incident, refactive_index)
    if not refracted_vector is None:
        redirected_ray = Ray(
            origin=new_origin - unit_normal * bias,
            orientation=refracted_vector,
            name=f"{incoming_ray.name}_bounce"
        )

        logging.debug("Successfully refracted")
        return redirected_ray, False, NdotL >= 0.0
    
    # Total internal reflection occurred
    logging.debug("Total internal reflection (fallback reflection)")
    return *calculate_surface_reflection_ray(unit_normal, unit_dir, new_origin, roughness, sampler), True, NdotL >= 0

"""
Luminance module: Provides classes for color representation, light rays, materials, and light sources.

Classes:
    - Color: RGBA color with conversions (hex, RGB255, array)
    - ColorGradient: Interpolated color gradients
    - LightSource: Point light emitting rays with color and intensity
    - Material: Surface properties (roughness, glossiness) affecting light interaction
Functions:
    - Various distance attenuation functions
    - Color attenuation function
    - Optical redirection calculation for reflection/refraction
"""
