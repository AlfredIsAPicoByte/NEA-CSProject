import numpy as np
from math import cos, sin, sqrt
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional

from CommonUtils import clamp, lerp, unit, orthonormal_basis, attenuate_sqr_distance, attenuate_distance_exponential
from PrimaryStructures import Ray, HitInfo
from Reflections import calculate_reflection_vector
from Refractions import calculate_refraction_vector, calculate_reflectance, REFRACTIVE_INDICES
from Sampling import Sampler
import logging

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

@dataclass
class PBRMaterialData:
    name: str = "DefaultMat"
    
    # --- PBR Parameters ---
    albedo: Color = field(default_factory=lambda: Color(1, 1, 1))
    roughness: float = 0.7
    metallic: float = 0.0
    specular_intensity: float = 0.5
    specular_tint: float = 0.0
    ior: float = 1.45
    
    # --- Transparency/Volume Parameters ---
    transmission: float = 0.0      # 0=Opaque, 1=Glass
    absorption_color: Color = field(default_factory=lambda: Color(1, 1, 1))
    absorption_density: float = 0.0
    
    # --- Emissive ---
    emission: Color = field(default_factory=lambda: Color(0, 0, 0))
    emission_intensity: float = 0.0

    # --- Flags ----
    type: MaterialType = MaterialType.DIFFUSE

class PBRMaterial:
    def __init__(
            self,
            data: PBRMaterialData
        ):
        self.data = data

    @classmethod
    def create_diffuse(cls, albedo: Color, roughness: float = 0.5):
        """
        Creates a standard non-metallic (dielectric) material like plastic, wood, or chalk.
        """
        data = PBRMaterialData(
            name="DiffuseMat",
            type=MaterialType.DIFFUSE,
            albedo=albedo,
            roughness=roughness,
            metallic=0.0,             # Non-metal
            specular_intensity=0.5,   # Most dielectrics have approx 4% reflectance (0.5 scale factor depending on your BRDF)
            transmission=0.0
        )
        return cls(data)

    @classmethod
    def create_specular(cls, albedo: Color, roughness: float = 0.2, metallicness: float = 1.0, specular_intensity: float = 1.0, specular_tint_amount: float = 0.5):
        """
        Creates a reflective material like gold, aluminum, or copper.
        """
        data = PBRMaterialData(
            name="MetalMat",
            type=MaterialType.SPECULAR,
            albedo=albedo,
            roughness=roughness,
            metallic=metallicness,    # 1.0 = Pure Metal
            specular_intensity=specular_intensity,
            specular_tint=specular_tint_amount,
            transmission=0.0
        )
        return cls(data)

    @classmethod
    def create_glass(cls, albedo: Color, absorption_color: Color, roughness: float = 0.0, metallicness: float = 0.0, ior: float = 1.5, absorption_density: float = 1.0):
        """
        Creates a dielectric transparent material (Refractive).
        """
        data = PBRMaterialData(
            name="GlassMat",
            type=MaterialType.GLASS,
            
            # Surface Properties
            albedo=albedo,            # Surface tint (usually White for clear glass)
            roughness=roughness,      # 0.0 = Clear, 0.5 = Frosted
            metallic=metallicness,    # Usually 0.0
            ior=ior,
            
            # Volumetric/Transmission Properties
            transmission=1.0,         # Enables Refraction logic
            absorption_color=absorption_color, # The color inside the glass (Beer's Law)
            absorption_density=absorption_density
        )
        return cls(data)

    @classmethod
    def create_transparent(cls, albedo: Color):
        """
        Creates a "See-Through" material using Alpha Blending (Ghosts, Holograms, Decals).
        Different from Glass because it does not refract light.
        """
        data = PBRMaterialData(
            name="TransparentMat",
            type=MaterialType.TRANSPARENT,
            albedo=albedo,            # albedo.a (Alpha) controls opacity
            roughness=0.8,            # Usually fairly rough to avoid sharp specular highlights on a ghost
            metallic=0.0,
            transmission=0.0          # 0 because we use Alpha Blending, not Refraction
        )
        return cls(data)

    @classmethod
    def create_emissive(cls, color: Color, intensity: float = 1.0):
        """
        Creates a glowing material (Light Bulb, Neon Sign).
        """
        data = PBRMaterialData(
            name="EmissiveMat",
            type=MaterialType.EMISSIVE,
            albedo=color,
            emission=color,
            emission_intensity=intensity,
            roughness=1.0 # Roughness doesn't matter much for emitters
        )
        return cls(data)
    
    def evaluate_direct_light(
            self,
            scene_lights: List[LightSource],
            hit_info: HitInfo,
            view_dir: np.ndarray,
            visibility_function: Callable[[np.ndarray, np.ndarray], float],
            bias: float = 1e-4
        ) -> Color:
        """
        Calculates the Direct Lighting (Shadows + Light Sources) for this material.
        Enforces Energy Conservation (Reflected + Diffuse <= Incoming).
        """
        # 1. Handle Emission (Self-Illumination)
        # Note: This is usually added separately, but returning it here is valid if 
        # you treat the object as its own light source.
        if self.type == MaterialType.EMISSIVE:
            return self.get_emissive_component()

        surface_normal = hit_info.normal
        hit_point = hit_info.point
        accumulated_light = Color(0, 0, 0)

        for light in scene_lights:
            # --- Light Calculation ---
            light_dir = light.get_light_direction(hit_point)
            dist = np.linalg.norm(light.position - hit_point)
            
            # Optimization: Skip lights that are too close or behind the surface
            if dist <= bias: continue
            
            NdotL = max(0.0, np.dot(surface_normal, light_dir))
            
            # Translucent materials (Glass/Thin) can be lit from behind
            if NdotL <= 0.0 and not self.is_transparent and self.type != MaterialType.GLASS:
                continue

            # --- Visibility (Shadows) ---
            if self.type == MaterialType.TRANSPARENT:
                visibility = 1.0 # Simple transparency ignores shadows
            else:
                visibility = visibility_function(hit_point, light.position)

            if visibility <= 0.0: continue

            # --- Attenuation ---
            attenuation = attenuate_sqr_distance(dist)
            incoming_radiance = light.color * light.intensity * attenuation * visibility

            # --- PBR Response (The Core Math) ---
            # 1. Calculate Fresnel (kS - Specular Fraction)
            H = unit(view_dir + light_dir)
            VdotH = max(0.0, np.dot(view_dir, H))
            
            F0 = self.get_metallic_component()
            kS = self.get_fresnel_component(VdotH, f0=F0)

            # 2. Calculate Diffuse Fraction (kD)
            # Conservation of Energy: kD = 1.0 - kS
            # If 90% reflects (kS), only 10% is left for diffuse (kD).
            kD = Color(1.0, 1.0, 1.0) - kS
            
            # Metals absorb all refracted light (no diffuse)
            kD = kD * (1.0 - self.metallic)

            # 3. Evaluate BRDF Terms
            diffuse_term = Color(0,0,0)
            specular_term = Color(0,0,0)

            if self.type != MaterialType.GLASS:
                 # Standard Lambertian Diffuse
                 diffuse_term = self.albedo * (1.0 / np.pi)

            if self.type == MaterialType.SPECULAR or self.type == MaterialType.GLASS or self.metallic > 0:
                 # Cook-Torrance Specular
                 specular_term = self.get_specular_brdf(kS, light_dir, view_dir, surface_normal)

            # 4. Final Combination
            # Out = (kD * Diffuse + Specular) * Light * NdotL
            light_contribution = (kD * diffuse_term + specular_term) * incoming_radiance * NdotL
            accumulated_light += light_contribution

        return accumulated_light

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

    def get_fresnel_component(self, cos_theta: float) -> Color:
        """
        Calculates the portion of light that is reflected (Specular) vs. absorbed/refracted (Diffuse).
        """
        f0 = self.get_metallic_component().to_np_ndarray()
        return Color.from_array(schlick_fresnel(cos_theta, f0))

    def get_emissive_component(self) -> Color:
        """Get the emissive color of the material."""
        return self.albedo * self.emissive_intensity

    def get_volumetric_component(self, light_color: Color, distance: float) -> Color:
        """
        Calculates the volumetric attenuation of light passing through the material
        """
        sigma = (np.array([1.0, 1.0, 1.0]) - self.absorption_color) * self.absorption_density
        
        # Calculate attenuation per channel
        # e.g. If Red has high sigma, exp(-high * dist) -> 0.0 (Red is blocked)
        attenuation = attenuate_distance_exponential(distance, sigma)
        
        return light_color * attenuation

    def get_transparency_component(self, light_color: Color) -> Color:
        """
        Calculates the transparency color contribution of the material.
        """
        return lerp(light_color, self.albedo, 1 - self.albedo.a)

    def get_ambient_color(self, ambient_color: Color, ambient_intensity: float) -> Color:
        """
        Calculates the ambient color contribution of the material.
        """
        ambient = ambient_color * ambient_intensity * self.albedo
        return ambient

    def calculate_microfacet_reflection_ray(
            self,
            surface_normal: np.ndarray,
            direction: np.ndarray,
            new_origin: np.ndarray,
            sampler: Sampler,
            bias: float = 1e-8
        ) -> Tuple[Ray, float]:
        """
            Generates a reflection ray using GGX Importance Sampling.
            Returns: (Ray, PDF)
        """
        # 1. Get Random Samples (u, v)
        # These determine "where" on the roughness hemisphere we pick a direction
        u, v = sampler.next_2d()

        # 2. Importance Sampling (GGX)
        # We map the random (u, v) to a 3D direction based on Roughness (alpha)
        # The rougher the surface, the wider the spread of possible directions.
        alpha = self.data.roughness ** 2

        phi = 2.0 * np.pi * u
        cos_theta = sqrt((1.0 - v) / (1.0 + ((alpha ** 2) - 1.0) * v))
        sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        H_tangent = np.array([
            sin_theta * cos(phi),
            sin_theta * sin(phi),
            cos_theta
        ])

        tangent, bitangent = orthonormal_basis(surface_normal)

        H_world = (tangent * H_tangent[0]) + (bitangent * H_tangent[1]) + (surface_normal * H_tangent[2])
        H_world = unit(H_world)

        view_dir = -unit(direction)

        dot_v_h = np.dot(view_dir, H_world)
        reflection_dir = (2.0 * dot_v_h * H_world) - view_dir
        final_dir = unit(reflection_dir)

        # 5. Create the new Ray
        # Offset origin to prevent acne
        final_origin = new_origin + (surface_normal * bias) # or hit_point + bias
        
        # Calculate Probability Density Function (PDF)
        # This is needed for the color math (throughput) to balance correctly.
        # (Simplified for demonstration)
        pdf = (2.0 * dot_v_h) / ((cos_theta * alpha * alpha) + bias) # Approximation

        new_ray = Ray(origin=final_origin, orientation=final_dir, name="mifro_facet_reflection")
        return new_ray, pdf
    
    def calculate_microfacet_refraction_ray(
            self,
            surface_normal: np.ndarray,
            direction: np.ndarray,
            new_origin: np.ndarray,
            sampler: Sampler,
            ior_incident: float,
            ior_transmitted: float,
            bias: float = 1e-4
        ) -> Tuple[Optional[Ray], float]:
        """
        Generates a refraction ray using GGX Importance Sampling (Frosted Glass).
        Returns: (Ray, PDF) or (None, 0.0) if Total Internal Reflection occurs.
        """
        # 1. Calculate Relative IOR (Eta)
        # -----------------------------
        eta = ior_incident / ior_transmitted

        # 2. Get Random Microfacet Normal (H)
        # -----------------------------------
        # Same GGX sampling as reflection. We sample a "tilt" for the surface
        # at this microscopic point.
        u, v = sampler.next_2d()
        alpha = self.data.roughness ** 2
        
        # Avoid singularities with perfectly smooth surfaces (alpha=0)
        alpha = max(alpha, 1e-4)

        phi = 2.0 * np.pi * u
        cos_theta = np.sqrt((1.0 - v) / (1.0 + ((alpha ** 2) - 1.0) * v))
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        H_tangent = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])

        tangent, bitangent = orthonormal_basis(surface_normal)
        H_world = (tangent * H_tangent[0]) + (bitangent * H_tangent[1]) + (surface_normal * H_tangent[2])
        H_world = unit(H_world)

        # 3. Refract the View Vector through H
        # ------------------------------------
        # We treat 'H_world' as the surface normal for this specific light ray.
        # Vector Math: Snell's Law in 3D
        
        # View direction (pointing TO the light, away from surface)
        view_dir = -unit(direction)
        
        # Dot product of View and Microfacet Normal
        dot_v_h = np.dot(view_dir, H_world)
        
        # Calculate the discriminant for Snell's law
        # sqrt_term = 1 - eta^2 * (1 - (v . h)^2)
        term_k = 1.0 - (eta * eta) * (1.0 - dot_v_h * dot_v_h)

        # CHECK: Total Internal Reflection (TIR)
        if term_k < 0.0:
            # The ray cannot refract through this specific microfacet angle.
            # In a full integrator, this energy would reflect, but for this 
            # specific function request, we return None.
            return None, 0.0

        # Calculate Refraction Vector
        # T = (eta * (v . h) - sqrt(k)) * h - eta * v
        # Note: We use -view_dir to represent the incident vector pointing IN.
        refraction_dir = (eta * dot_v_h - np.sqrt(term_k)) * H_world - (eta * view_dir)
        final_dir = unit(refraction_dir)

        # 4. Construct Ray
        # ----------------
        # Push origin THROUGH the surface (negative normal bias) to avoid self-intersection
        final_origin = new_origin - (surface_normal * bias)

        # 5. Calculate PDF (Jacobian approximation)
        # ---------------------------------------
        # The PDF for refraction is complex because the solid angle changes 
        # as the ray crosses the interface (compression/expansion).
        dot_l_h = np.abs(np.dot(final_dir, H_world))
        dot_v_h = np.abs(dot_v_h)
        
        # GGX Distribution (D) calculation
        denom = (dot_v_h * alpha * alpha) + bias # Simplified D term
        
        # Jacobian for Refraction
        sqrt_denom = dot_v_h + eta * dot_l_h
        jacobian = (eta * eta * dot_l_h) / (sqrt_denom * sqrt_denom + bias)
        
        pdf = denom * jacobian

        return Ray(origin=final_origin, orientation=final_dir, name="microfacet_refraction"), pdf

    def evaluate_brdf(self, light_intensity: float, light_dir: np.ndarray, surface_normal: np.ndarray, view_dir: np.ndarray, bias: float = 1e-4) -> Color:
        """
        Returns the BRDF value (ratio of radiance).
        """
        H = (view_dir + light_dir)
        # Safety check for degenerate vectors
        h_len = np.linalg.norm(H)
        if h_len < bias:
            return Color(0, 0, 0)
        H = H / h_len

        # Clamp dot products to avoid negative values (lighting from behind surface)
        VdotH = max(np.dot(view_dir, H), bias)

        # Compute Fresnel Term (F is a Color because F0 is a Color)
        F_color = self.get_fresnel_component(VdotH)

        # Compute PBR Terms
        specular_term = self.get_specular_component(F_color, light_intensity, light_dir, surface_normal, view_dir)
        diffuse_term = self.get_diffuse_component(F_color, light_intensity, light_dir, surface_normal) * (1.0/np.pi)

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

def calculate_fresnel_ratio(
    incident_dir: np.ndarray, 
    normal: np.ndarray, 
    ior_incident: float, 
    ior_transmitted: float
) -> float:
    """
    Calculates the ratio of light that reflects vs refracts using the Schlick Approximation.
    
    Returns:
        float: The probability of Reflection (0.0 to 1.0).
               The Refraction probability is (1.0 - result).
    """
    # 1. Calculate Cosine of the Incident Angle
    # Ensure vectors are normalized
    unit_incident = incident_dir / np.linalg.norm(incident_dir)
    unit_normal = normal / np.linalg.norm(normal)
    
    # cos_theta is usually -dot(view, normal).
    # Since incident_dir points INTO the surface, we negate it.
    cos_theta = np.dot(-unit_incident, unit_normal)
    
    # 2. Handle Total Internal Reflection (TIR) Check
    # This is required when moving from dense -> rare medium (e.g. Glass -> Air)
    if ior_incident > ior_transmitted:
        eta = ior_incident / ior_transmitted
        sin2_t = (eta ** 2) * (1.0 - cos_theta ** 2)
        
        # If sin2_t > 1.0, the angle is too shallow to escape.
        if sin2_t > 1.0:
            return 1.0 # 100% Reflection (Total Internal Reflection)

        # If not TIR, we must update cos_theta to use the cosine of the 
        # TRANSMITTED angle for the Schlick curve to be accurate.
        cos_theta = np.sqrt(max(0.0, 1.0 - sin2_t))

    # 3. Calculate R0 (Base Reflectivity at 0 degrees)
    r0 = ((ior_incident - ior_transmitted) / (ior_incident + ior_transmitted)) ** 2
    
    # 4. Schlick Approximation
    fresnel_reflectance = schlick_fresnel(cos_theta, np.array([r0]))[0]
    
    return fresnel_reflectance

"""
Luminance module: Provides classes for color representation, light rays, materials, and light sources.

Classes:
    - Color: RGBA color with conversions (hex, RGB255, array)
    - ColorGradient: Interpolated color gradients
    - LightSource: Point light emitting rays with color and intensity
    - Material: Surface properties (roughness, glossiness) affecting light interaction
Functions:
    - Color attenuation function
"""
