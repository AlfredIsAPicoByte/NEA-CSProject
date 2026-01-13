import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field, replace
from typing import Callable, List, Tuple, Optional, Union
import bisect

from CommonUtils import clamp, lerp, unit, orthonormal_basis, attenuate_sqr_distance, attenuate_distance_exponential
from PrimaryStructures import TracingRay, HitInfo
from Sampling import Sampler
from Reflections import calculate_reflection_vector
from Refractions import calculate_refraction_vector, calculate_reflectance, schlick_fresnel

@dataclass
class Color:
    """
    A data class representing a color with red, green, and blue components.
    Internal representation uses floats from 0.0 to 1.0.
    """
    r: float
    g: float
    b: float
    a: float = 1.0

    def clamp(self):
        """Clamps internal RGBA values to be between 0.0 and 1.0."""
        self.r = clamp(self.r)
        self.g = clamp(self.g)
        self.b = clamp(self.b)
        self.a = clamp(self.a)

    # --- Static Methods (The missing pieces) ---

    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Converts HSV (0.0-1.0) to RGB (0.0-1.0)."""
        if s == 0.0:
            return v, v, v
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        if i == 5: return v, p, q
        return 0.0, 0.0, 0.0

    @staticmethod
    def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Converts RGB (0.0-1.0) to HSV (0.0-1.0)."""
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        
        h = 0.0
        if mx == mn:
            h = 0.0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
            
        return (h / 360.0, (0 if mx == 0 else df / mx), mx)

    # --- Constructors ---

    @classmethod
    def from_hex(cls, hex_str: str):
        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6:
            raise ValueError(f"Invalid hex string: {hex_str}")
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return cls(r / 255.0, g / 255.0, b / 255.0)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0):
        r, g, b = cls.hsv_to_rgb(h, s, v)
        return cls(r, g, b, a)

    @classmethod
    def from_int_rgb(cls, r: int, g: int, b: int, a: int = 255):
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    # --- Output / Conversions ---

    def to_hex(self) -> str:
        return '#{:02x}{:02x}{:02x}'.format(
            int(clamp(self.r) * 255), 
            int(clamp(self.g) * 255), 
            int(clamp(self.b) * 255)
        )

    def to_hsv(self) -> Tuple[float, float, float]:
        return self.rgb_to_hsv(self.r, self.g, self.b)
    
    def to_int_rgb(self) -> Tuple[int, int, int]:
        return (int(self.r * 255), int(self.g * 255), int(self.b * 255))

    def to_np_array(self, include_alpha=False) -> np.ndarray:
        if include_alpha:
            return np.array([self.r, self.g, self.b, self.a], dtype=np.float32)
        return np.array([self.r, self.g, self.b], dtype=np.float32)
    
    def to_np_ndarray(self, include_alpha=False) -> np.ndarray:
        return self.to_np_array(include_alpha)

    # --- Arithmetic Operations (Inheritance Safe) ---

    def __add__(self, other: Union['Color', float]):
        if isinstance(other, Color):
            return replace(self, r=self.r + other.r, g=self.g + other.g, b=self.b + other.b)
        elif isinstance(other, (int, float)):
            return replace(self, r=self.r + other, g=self.g + other, b=self.b + other)
        if hasattr(other, 'r') and hasattr(other, 'g') and hasattr(other, 'b'):
            return replace(self, r=self.r + other.r, g=self.g + other.g, b=self.b + other.b)
        return NotImplemented

    def __sub__(self, other: Union['Color', float]):
        if isinstance(other, Color):
            return replace(self, r=self.r - other.r, g=self.g - other.g, b=self.b - other.b)
        elif isinstance(other, (int, float)):
            return replace(self, r=self.r - other, g=self.g - other, b=self.b - other)
        if hasattr(other, 'r') and hasattr(other, 'g') and hasattr(other, 'b'):
            return replace(self, r=self.r - other.r, g=self.g - other.g, b=self.b - other.b)
        return NotImplemented

    def __mul__(self, scale: Union['Color', float, int]):
        if isinstance(scale, (int, float)):
            return replace(self, r=self.r * scale, g=self.g * scale, b=self.b * scale)
        if isinstance(scale, Color):
            return replace(self, r=self.r * scale.r, g=self.g * scale.g, b=self.b * scale.b, a=self.a * scale.a)
        if hasattr(scale, 'r') and hasattr(scale, 'g') and hasattr(scale, 'b'):
            return replace(self, r=self.r * scale.r, g=self.g * scale.g, b=self.b * scale.b)
        return NotImplemented

    def __rmul__(self, scale: Union['Color', float]):
        return self.__mul__(scale)

    def __truediv__(self, scale: float):
        if isinstance(scale, (int, float)) and scale != 0:
            return replace(self, r=self.r / scale, g=self.g / scale, b=self.b / scale)
        return NotImplemented
    
    def __getitem__(self, index):
        if index == 0 or index == "r" or index == "red":
            return self.r
        elif index == 1 or index == "g" or index == "green":
            return self.r
        elif index == 2 or index == "b" or index == "blue":
            return self.r
        elif index == 3 or index == "a" or index == "alpha":
            return self.r

    def __repr__(self):
        return f"Color(r={self.r:.2f}, g={self.g:.2f}, b={self.b:.2f}, a={self.a:.2f})"

@dataclass(slots=True)
class ColorGradient:
    colors: List[Color]
    positions: np.ndarray

    def __post_init__(self):
        """
        Args:
            positions: List of floats between 0.0 and 1.0 (must be sorted).
            colors: List of numpy arrays (e.g. [R, G, B, A]).
        """
        if len(self.positions) != len(self.colors):
            raise ValueError("Positions and colors must have the same length.")
        
        # Ensure sorted data for binary search logic
        sorted_pairs = sorted(zip(self.positions, self.colors), key=lambda x: x[0])
        self.positions = np.array([p for p, c in sorted_pairs], dtype=float)
        self.colors = [c for p, c in sorted_pairs]

    def get_color(
        self, 
        t: float, 
        interpolation_function: Callable[[float], float] = lambda x: x
    ) -> Color:
        """
        Get interpolated color at position t in [0.0, 1.0].
        Optimized using NumPy vectorization and bisect for speed.
        """
        # 1. Clamp t
        t = max(0.0, min(1.0, t))

        # 2. Fast Path: Boundaries
        if t <= self.positions[0]:
            return self.colors[0]
        if t >= self.positions[-1]:
            return self.colors[-1]

        # 3. Find the segment using binary search (faster than loop for many stops)
        # bisect_right returns the insertion point to maintain order. 
        # For t, it gives us the index of the first position > t.
        idx = bisect.bisect_right(self.positions, t)
        
        # The segment is between idx-1 and idx
        t0, t1 = self.positions[idx-1], self.positions[idx]
        c0, c1 = self.colors[idx-1], self.colors[idx]

        # 4. Calculate factor
        denom = t1 - t0
        if denom < 1e-8: # Avoid division by zero
            return c1
            
        local_t = (t - t0) / denom
        factor = interpolation_function(local_t)

        # 5. Vectorized Linear Interpolation (Lerp)
        # Calculates R, G, B, A simultaneously
        return lerp(c0, c1, factor)
        
class LightType(Enum):
    POINT = 1
    DIRECTIONAL = 2

class LightSource:
    def __init__(self, position: np.ndarray, color: Color, intensity: float = 1.0, radius: float = 1.0, name: str = "Light Source"):
        self.position = position
        self.color = color
        self.intensity = intensity
        self.radius = radius
        self.name = name
        self.type = LightType.POINT

    def get_light_direction(self, hit_point: np.ndarray, bias: float = 1e-8) -> np.ndarray:
        """Return the normalized direction vector from the hit point towards the light source."""
        return unit(self.position - hit_point, bias)

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
    albedo: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))
    roughness: float = 0.5
    metallic: float = 0.0
    specular_intensity: float = 0.5
    specular_tint: float = 0.0
    ior: float = 1.45
    
    # --- Transparency/Volume Parameters ---
    transmission: float = 0.0      # 0=Opaque, 1=Glass
    absorption_color: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))
    absorption_density: float = 0.0
    
    # --- Emissive ---
    emission_color: Color = field(default_factory=lambda: Color(0.0, 0.0, 0.0))
    emission_intensity: float = 0.0

    # --- Flags ----
    type: MaterialType = MaterialType.DIFFUSE

class PBRMaterial:
    def __init__(
            self,
            data: PBRMaterialData
        ):
        self.data = data

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying `data` container for
        convenience (so callers can use `material.type` or `material.albedo`).
        """
        return getattr(self.data, name)

    def evaluate_direct_light(
        self,
        scene_lights: List[LightSource],
        hit_info: HitInfo,
        view_dir: np.ndarray,
        visibility_function: Callable[[np.ndarray, LightSource], float],
        bias: float = 1e-4
    ) -> Color:
        """
        Calculates Direct Lighting contribution.
        Assumes hit_info.point and scene_lights are ALL in World Space.
        """
        # Use world-space hit info directly for lighting calculations.
        # Mixing object-space and world-space (e.g., transforming hit point to local
        # space while using world-space light positions) caused incorrect shading.
        surface_normal = unit(hit_info.normal)
        hit_point = hit_info.point
        accumulated_light = Color(0.0, 0.0, 0.0)
        
        if hit_point is None or surface_normal is None:
            return accumulated_light

        for light in scene_lights:
            if self.data.type == MaterialType.EMISSIVE:
                break

            # --- Light Setup (World Space) ---
            light_dir = light.get_light_direction(hit_point)
            light_dist = np.linalg.norm(light.position - hit_point)
            
            if light_dist <= bias: continue
            
            # --- Visibility Check (Shadows) ---
            # Note: If checking visibility through glass, this simple function 
            # usually returns 0 (blocked) unless you implement "Transparent Shadows"
            visibility = visibility_function(hit_point, light)
            if visibility <= 0.0:
                continue
            
            # Pre-calculate attenuation (distance-based).
            attenuation = attenuate_sqr_distance(light_dist)
            intensity = light.intensity * attenuation * visibility

            # --- Material Handling ---
            if self.type == MaterialType.DIFFUSE:
                diffuse = self.get_diffuse_component(
                    light.color, intensity, light_dir, surface_normal
                )
                accumulated_light += diffuse 

                diffuse = self.get_diffuse_component(light.color, intensity, light_dir, surface_normal)
                accumulated_light += diffuse 
            elif self.type == MaterialType.SPECULAR:
                diff = self.get_diffuse_component(light.color, intensity, light_dir, surface_normal)
                spec = self.get_specular_component(light.color, intensity, light_dir, surface_normal, view_dir)
                accumulated_light += diff + spec

        return accumulated_light
    
    def evaluate_bsdf(
        self,
        incident_dir: np.ndarray,   # Incoming Light Direction (L) - Pointing OUT from surface
        view_dir: np.ndarray,       # View Direction (V) - Pointing OUT from surface
        surface_normal: np.ndarray, # Geometric Normal (N)
        roughness: float,
        other_ior: float
    ) -> Color:
        """
        Evaluates the Microfacet BSDF for a specific pair of input/output directions.
        Returns: RGB Color attenuation (BSDF value).
        """
        # 1. Normalize and Setup
        L = unit(incident_dir)
        V = unit(view_dir)
        N = unit(surface_normal)
        
        # Cosines
        dot_n_l = np.dot(N, L)
        dot_n_v = np.dot(N, V)

        # 2. Determine Mode: Reflection or Refraction?
        # If L and V are on the same side of the normal, it's Reflection.
        # If they are on opposite sides, it's Refraction.
        is_reflection = (dot_n_l * dot_n_v) > 0

        if dot_n_v > 0:
            eta_v = other_ior       # Air
            eta_l = self.ior        # Glass (Light is inside)
        else:
            eta_v = self.ior        # Glass
            eta_l = other_ior       # Air (Light is outside)

        if is_reflection:
            # --- REFLECTION LOBE ---
            # 1. Calculate Half Vector (H)
            H = unit(L + V)
            
            # 2. Check Validity
            # If V or L is below horizon relative to H, contribution is 0
            if np.dot(V, H) <= 0 or np.dot(L, H) <= 0:
                return Color(0.0, 0.0, 0.0)

            # 3. Calculate Fresnel (F)
            # Using your helper or Schlick directly
            # Note: We use H as the normal for Fresnel
            dot_v_h = np.abs(np.dot(V, H))
            F = calculate_reflectance(np.degrees(math.acos(dot_v_h)), eta_v, eta_l)
            if F is None: F = 1.0 # Handle TIR case

            # 4. Calculate D and G
            D = ggx_distribution(N, H, roughness)
            G = smith_geometry(N, V, L, roughness)

            # 5. Cook-Torrance Specular BRDF Formula
            # f = (D * G * F) / (4 * (N.L) * (N.V))
            denominator = 4.0 * np.abs(dot_n_l) * np.abs(dot_n_v) + 1e-8
            val = (D * G * F) / denominator
            
            return Color(val, val, val)

        else:
            # --- REFRACTION LOBE ---
            # 1. Calculate Refraction Half Vector (H)
            H_raw = -(eta_l * L + eta_v * V)
            H = unit(H_raw)
            
            # 3. Calculate Dot Products
            # Critical: Keep signs for the denominator!
            dot_v_h = np.dot(V, H)
            dot_l_h = np.dot(L, H)
            dot_h_n = np.dot(H, N) # Used for D

            # 4. Fresnel (F)
            # Use abs for Fresnel calc as it depends on angle magnitude
            F = calculate_reflectance(np.degrees(math.acos(abs(dot_v_h))), eta_l, eta_v)
            if F is None: F = 1.0

            # 5. Geometry & Distribution
            D = ggx_distribution(N, H, roughness)
            G = smith_geometry(N, V, L, roughness)

            # 6. The Walter '07 BTDF Formula
            # Term: (eta_L * (L.H) + eta_V * (V.H))^2
            # Since L and V are opposite, one dot is (+) and one is (-).
            # This results in a small number squared.
            sqrt_denom = (eta_l * dot_l_h + eta_v * dot_v_h)
            
            # Numerator
            # Note: For Radiance transport, we scale by the OUTGOING IOR squared (eta_v^2).
            # We use abs() for the geometric factors in the numerator/denominator logic.
            numerator = abs(dot_v_h) * abs(dot_l_h) * (eta_v ** 2) * D * G * (1.0 - F)
            
            denominator = abs(dot_n_l) * abs(dot_n_v) * (sqrt_denom ** 2) + 1e-8
            
            val = numerator / denominator
            
            return Color(val, val, val)

    def get_diffuse_component(self, light_color: Color, light_intensity: float, light_dir: np.ndarray, surface_normal: np.ndarray) -> Color:
        """
        Get the diffuse component (Lambertian). 
        Corrected to respect the Metallic workflow energy conservation.
        """
        NdotL = max(0.0, np.dot(surface_normal, light_dir))
        
        # 1. Standard Lambert Diffuse (Simple but physically consistent)
        diffuse = light_intensity * self.data.albedo * NdotL
        
        # 2. ENERGY CONSERVATION: Metals have NO diffuse.
        # As metallic approaches 1.0, diffuse must approach 0.0.
        diffuse = light_color * diffuse * (1.0 - self.data.metallic)
        
        return diffuse

    def get_specular_component(self, light_color: Color, light_intensity: float, light_dir: np.ndarray, surface_normal: np.ndarray, view_dir: np.ndarray, bias: float = 1e-4) -> Color:
        """Get the specular component of the material response using the Micro-Facet BRDF."""
        
        # --- 0. Pre-Calculations and Constants ---
        safe_roughness = max(self.data.roughness, 1e-2)
        alpha = safe_roughness ** 2
        alpha_sq = alpha ** 2
        
        # Halfway Vector (H)
        H = (light_dir + view_dir)
        H = unit(H) 
        
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
        specular = light_color * Fs * NdotL * light_intensity * self.data.specular_intensity
        
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
        max_val = max(self.data.albedo.r, self.data.albedo.g, self.data.albedo.b, bias)
        albedo_normalized = self.data.albedo / max_val
        
        F0_dielectric_tinted = F0_dielectric * albedo_normalized
        
        # Apply the Specular Tint Slider
        F0_final_dielectric = lerp(F0_dielectric, F0_dielectric_tinted, self.data.specular_tint)

        # --- 2. Metallic F0 ---
        # Metals use their Albedo directly as F0
        F0_metal = self.data.albedo

        # --- 3. Final Blend ---
        # Lerp between Dielectric (0.04) and Metal (Albedo)
        final_F0: Color = lerp(F0_final_dielectric, F0_metal, self.data.metallic)
        
        return final_F0

    def get_fresnel_component(self, cos_theta: float) -> np.ndarray:
        """
        Calculates the portion of light that is reflected (Specular) vs. absorbed/refracted (Diffuse).
        """
        f0 = self.get_metallic_component().to_np_array()
        return schlick_fresnel(cos_theta, f0)

    def get_emissive_component(self) -> Color:
        """Get the emissive color of the material."""
        return self.data.emission_color * self.data.emission_intensity

    def get_volumetric_component(self, light_color: Color, distance: float) -> Color:
        """
        Calculates the volumetric attenuation of light passing through the material
        """
        sigma = (Color(1.0, 1.0, 1.0) - self.data.absorption_color)
        
        # Calculate attenuation per channel
        # e.g. If Red has high sigma, exp(-high * dist) -> 0.0 (Red is blocked)
        attenuation = sigma * attenuate_distance_exponential(distance, self.data.absorption_density)
        
        return light_color * attenuation

    def get_transparency_component(self, light_color: Color) -> Color:
        """
        Calculates the transparency color contribution of the material.
        """
        return lerp(light_color, self.data.albedo, 1 - self.data.albedo.a)

    def get_ambient_color(self, ambient_color: Color, ambient_intensity: float) -> Color:
        """
        Calculates the ambient color contribution of the material.
        """
        ambient = ambient_color * ambient_intensity * self.data.albedo
        return ambient

    def __repr__(self):
        return (
            f"Material(albedo={self.data.albedo}, other={self.data})"
        )
    
class MaterialFactory:
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
            specular_intensity=0.0,   # Most dielectrics have approx 4% reflectance
            transmission=0.0
        )
        return PBRMaterial(*data)

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
            metallic=metallicness,
            specular_intensity=specular_intensity,
            specular_tint=specular_tint_amount,
            transmission=0.0
        )
        return PBRMaterial(*data)

    @classmethod
    def create_glass(cls, albedo: Color, absorption_color: Color, roughness: float = 0.0, metallicness: float = 0.0, ior: float = 1.5, transmission: float = 1.0, absorption_density: float = 1.0):
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
            
            # Volumetric/Transmission Properties
            ior=ior,
            transmission=transmission,         # Enables Refraction logic
            absorption_color=absorption_color, # The color inside the glass (Beer's Law)
            absorption_density=absorption_density
        )
        return PBRMaterial(*data)

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
        return PBRMaterial(*data)

    @classmethod
    def create_emissive(cls, color: Color, intensity: float = 1.0):
        """
        Creates a glowing material (Light Bulb, Neon Sign).
        """
        data = PBRMaterialData(
            name="EmissiveMat",
            type=MaterialType.EMISSIVE,
            emission_color=color,
            emission_intensity=intensity,
        )
        return PBRMaterial(*data)

def ggx_distribution(normal: np.ndarray, half_vector: np.ndarray, roughness: float) -> float:
    """Calculates the GGX/Trowbridge-Reitz Normal Distribution Function (D)."""
    alpha = max(roughness ** 2, 1e-4)
    dot_n_h = np.dot(normal, half_vector)
    
    # D is 0 if the half vector is below the geometric surface
    if dot_n_h <= 0:
        return 0.0

    denominator = (dot_n_h ** 2) * (alpha ** 2 - 1.0) + 1.0
    return (alpha ** 2) / (np.pi * denominator * denominator)

def smith_geometry(n: np.ndarray, v: np.ndarray, l: np.ndarray, roughness: float) -> float:
    """
    Smith Geometry Shadowing-Masking function.
    Determines what percentage of microfacets are blocked by other microfacets.
    """
    # Using the Schlick-GGX approximation for Smith G
    # k = (alpha + 1)^2 / 8  (for direct lighting / analytic)
    # k = alpha^2 / 2        (for IBL / path tracing) -> We use this usually for consistency
    alpha = roughness ** 2
    k = (alpha) / 2.0 

    dot_n_v = np.abs(np.dot(n, v))
    dot_n_l = np.abs(np.dot(n, l))

    g1_v = dot_n_v / (dot_n_v * (1.0 - k) + k)
    g1_l = dot_n_l / (dot_n_l * (1.0 - k) + k)

    return g1_v * g1_l

def calculate_microfacet_pdf(
    incident_dir: np.ndarray,   # Direction light is coming FROM (World space)
    outgoing_dir: np.ndarray,   # The sampled direction (Reflection or Refraction)
    surface_normal: np.ndarray,
    roughness: float,
    ior_incident: float,
    ior_transmitted: float,
    fresnel_probability: float  # The 'F' value calculated during sampling
) -> float:
    """
    Calculates the PDF for a specific reflection or refraction event.
    """
    # 1. Normalize Vectors
    V = -unit(incident_dir) # View Vector (pointing to viewer)
    L = unit(outgoing_dir)  # Light/Sample Vector
    N = unit(surface_normal)

    # 2. Determine if this is Reflection or Refraction
    # We check if L and N are in the same hemisphere
    is_reflection = np.dot(L, N) > 0

    if is_reflection:
        # --- REFLECTION PDF ---
        
        # Calculate Half Vector (H) for Reflection
        # H = Normalize(V + L)
        H = unit(V + L)
        
        # Calculate Dot Products
        dot_n_h = np.abs(np.dot(N, H))
        dot_v_h = np.abs(np.dot(V, H))
        
        # Calculate D term
        D = ggx_distribution(N, H, roughness)
        
        # Jacobian for Reflection: 1 / (4 * (V.H))
        pdf_geometry = (D * dot_n_h) / (4.0 * dot_v_h + 1e-8)
        
        # Combine with Selection Probability (Fresnel)
        return pdf_geometry * fresnel_probability

    else:
        # --- REFRACTION PDF ---
        
        # Calculate Half Vector (H) for Refraction
        # Standard microfacet H for refraction: -(eta_i * V + eta_t * L)
        # Note: We must be careful with signs. 
        # Usually H is constructed to point into the simpler medium or averaged.
        # Robust method:
        eta_i = ior_incident
        eta_t = ior_transmitted
        
        H_unstand = -(eta_i * V + eta_t * L)
        H = unit(H_unstand)
        
        # D term
        dot_n_h = np.abs(np.dot(N, H))
        D = ggx_distribution(N, H, roughness)
        
        # Calculate Terms for Jacobian
        dot_v_h = np.dot(V, H)
        dot_l_h = np.dot(L, H)
        
        # Denominator part: (eta_i * (V.H) + eta_t * (L.H))^2
        sqrt_denom = (eta_i * dot_v_h + eta_t * dot_l_h)
        denom = sqrt_denom * sqrt_denom
        
        # Jacobian for Refraction
        # J = (eta_t^2 * |L.H|) / (eta_i * (V.H) + eta_t * (L.H))^2
        # Note: The 'D(h) * dot_n_h' part comes from the sampling of H itself.
        jacobian = (eta_t ** 2 * np.abs(dot_l_h)) / (denom + 1e-8)
        
        # PDF in solid angle measure
        pdf_geometry = D * dot_n_h * jacobian 
        
        # Combine with Selection Probability (1 - F)
        # Note: We multiply by derivative of H wrt solid angle
        # Most implementations simplify the weight calculation directly, 
        # but this is the raw PDF value.
        return pdf_geometry * (1.0 - fresnel_probability)
    
def sample_microfacet_glass(
        incident_dir: np.ndarray,
        surface_normal: np.ndarray,
        new_origin: np.ndarray,
        sampler: Sampler,
        roughness: float,
        ior_incident: float,
        ior_transmitted: float,
        bias: float = 1e-4
    ) -> TracingRay:
    """
    Unified Microfacet BSDF (Glass/Dielectric).
    Probabilistically samples Reflection or Refraction based on Fresnel term.
    """
    # 1. Setup & IOR
    # -------------------------
    # Direction vectors should be normalized
    view_dir = -unit(incident_dir) # Pointing towards viewer/light source
    
    # Calculate Eta (Relative IOR)
    eta = ior_incident / ior_transmitted

    # 2. Sample Microfacet Normal (H) (GGX)
    # -------------------------
    # Both reflection and refraction rely on the same microfacet distribution.
    # We sample H once.
    u, v = sampler.sample_bsdf()
    alpha = max(roughness ** 2, bias) # Prevent div by zero

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

    # Ensure H points into the same hemisphere as the view direction
    if np.dot(view_dir, H_world) < 0:
        H_world = -H_world

    # 3. Calculate Fresnel Term (The Selector)
    # -------------------------
    dot_v_h = np.dot(view_dir, H_world)
    dot_v_h = np.clip(dot_v_h, 0.0, 1.0)
    incident_angle_deg = np.degrees(math.acos(dot_v_h))

    # Calculate F (Probability of Reflection)
    # This handles the complex Fresnel equations for you.
    F = calculate_reflectance(incident_angle_deg, ior_incident, ior_transmitted)
    
    # Handle Total Internal Reflection (TIR)
    # Your module returns None if TIR occurs
    if F is None:
        F = 1.0

    # 4. Russian Roulette: Reflect or Refract?
    # -------------------------
    # We use a new random sample to decide the path.
    w = sampler.sample_roulette()
    dot_v_h = np.dot(view_dir, H_world)

    if w < F:
        # --- REFLECTION ---
        # Critical: Pass H_world as the normal!
        final_dir = calculate_reflection_vector(H_world, incident_dir)
        
        # Origin Offset: Push AWAY from geometric normal
        final_origin = new_origin + (surface_normal * bias)
        
        # Note on PDF: In a combined BSDF, the PDF is usually weighted by the probability
        # of choosing this lobe.
        # pdf_final = pdf_reflection * F
        
    else:
        # --- REFRACTION ---
        final_dir = calculate_refraction_vector(
            surface_normal=H_world,  # Use Microfacet H
            direction=incident_dir,
            refractive_index_incident=ior_incident,
            refractive_index=ior_transmitted
        )
        
        # Safety: If your module detects TIR (returning None), fallback to reflection.
        # (Though our F check above should catch this, floating point errors happen).
        if final_dir is None:
            final_dir = calculate_reflection_vector(H_world, incident_dir)
            final_origin = new_origin + (surface_normal * bias)
        else:
            # Origin Offset: Push THROUGH geometric normal
            final_origin = new_origin - (surface_normal * bias)

        # pdf_final = pdf_refraction * (1 - F)

    return TracingRay(origin=final_origin, orientation=final_dir)

def calculate_throughput_weight(
        light_dir: np.ndarray,
        surface_normal: np.ndarray,
        bsdf_value: np.ndarray,
        pdf: float,
        bias: float = 1e-6
    ) -> np.ndarray: 
    """
    Calculates the weight (color contribution) of a specific ray sample 
    using the Monte Carlo estimator: (BSDF * CosTheta) / PDF.
    
    Args:
        light_dir: The direction of the OUTGOING (sampled) ray.
        surface_normal: The geometric surface normal.
        bsdf_value: The RGB color returned by evaluate_bsdf().
        pdf: The probability density calculated by calculate_pdf().
    """
    
    # 1. Safety Check: Avoid division by zero
    if pdf < bias:
        return np.array([0.0, 0.0, 0.0])

    # 2. Geometry Term (Cosine Law / Foreshortening)
    # We take the Absolute value (|N.L|) because:
    # - Reflection: L is on the same side as N (positive).
    # - Refraction: L is on the opposite side of N (negative).
    # Both attenuate light based on the projected area.
    cos_theta = np.abs(np.dot(surface_normal, light_dir))

    # 3. Calculate Weight
    # Weight = (BSDF * CosTheta) / PDF
    weight = (bsdf_value * cos_theta) / pdf

    return weight

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
