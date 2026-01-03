from math import cos, sin, sqrt
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Union
import bisect

from CommonUtils import clamp, lerp, unit, orthonormal_basis, attenuate_sqr_distance, attenuate_distance_exponential
from PrimaryStructures import Ray, HitInfo
from Sampling import Sampler

@dataclass
class Color:
    """
    A data class representing a color with red, green, and blue components.
    Internal representation uses floats from 0.0 to 1.0.
    """
    r: float
    g: float
    b: float
    a: float = 1.0  # Alpha channel (opacity), defaults to 1.0

    def clamp(self):
        """Clamps internal RGBA values to be between 0.0 and 1.0."""
        self.r = clamp(self.r)
        self.g = clamp(self.g)
        self.b = clamp(self.b)
        self.a = clamp(self.a)

    # --- Constructors ---

    @classmethod
    def from_hex(cls, hex_str: str):
        """Creates a Color from a hex string (e.g., '#FF5733' or 'FF5733')."""
        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6:
            raise ValueError(f"Invalid hex string: {hex_str}")
        
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return cls(r / 255.0, g / 255.0, b / 255.0)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0):
        """Creates a Color from HSV values (0.0 - 1.0)."""
        r, g, b = cls.hsv_to_rgb(h, s, v)
        return cls(r, g, b, a)

    @classmethod
    def from_int_rgb(cls, r: int, g: int, b: int, a: int = 255):
        """Creates a Color from 0-255 integer values."""
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    # --- Output / Conversions ---

    def to_hex(self) -> str:
        """Returns the hex string representation (e.g., '#FF0000')."""
        return '#{:02x}{:02x}{:02x}'.format(
            int(self.r * 255), int(self.g * 255), int(self.b * 255)
        )

    def to_hsv(self) -> Tuple[float, float, float]:
        """Returns a tuple of (Hue, Saturation, Value)."""
        return self.rgb_to_hsv(self.r, self.g, self.b)

    def to_int_rgb(self) -> Tuple[int, int, int]:
        """Returns a tuple of integers (0-255)."""
        return (int(self.r * 255), int(self.g * 255), int(self.b * 255))

    def to_np_array(self, include_alpha=False) -> np.ndarray:
        """Returns a numpy array of the color."""
        if include_alpha:
            return np.array([self.r, self.g, self.b, self.a], dtype=np.float32)
        return np.array([self.r, self.g, self.b], dtype=np.float32)

    # Compatibility aliases (tests and external code may expect different names)
    def to_np_ndarray(self, include_alpha=False) -> np.ndarray:
        """Alias kept for older code/tests expecting `to_np_ndarray`."""
        return self.to_np_array(include_alpha)

    @property
    def red(self) -> float:
        return self.r

    @red.setter
    def red(self, v: float):
        self.r = v

    @property
    def green(self) -> float:
        return self.g

    @green.setter
    def green(self, v: float):
        self.g = v

    @property
    def blue(self) -> float:
        return self.b

    @blue.setter
    def blue(self, v: float):
        self.b = v

    # --- Arithmetic Operations ---

    def __add__(self, other: Union['Color', float]):
        """Adds two colors (additive blending) or adds a scalar to brightness."""
        if isinstance(other, Color):
            return Color(self.r + other.r, self.g + other.g, self.b + other.b, self.a)
        elif isinstance(other, (int, float)):
            return Color(self.r + other, self.g + other, self.b + other, self.a)
        return NotImplemented

    def __sub__(self, other: Union['Color', float]):
        """Subtracts a color or scalar."""
        if isinstance(other, Color):
            return Color(self.r - other.r, self.g - other.g, self.b - other.b, self.a)
        elif isinstance(other, (int, float)):
            return Color(self.r - other, self.g - other, self.b - other, self.a)
        return NotImplemented

    def __mul__(self, scale: float):
        """Multiplies color by a scalar (adjusts brightness)."""
        # Support scalar multiplication
        if isinstance(scale, (int, float)):
            return Color(self.r * scale, self.g * scale, self.b * scale, self.a)
        # Support component-wise color * color multiplication
        if isinstance(scale, Color):
            return Color(self.r * scale.r, self.g * scale.g, self.b * scale.b, self.a * scale.a)
        return NotImplemented

    def __rmul__(self, scale: float):
        """Supports `float * Color` by delegating to __mul__."""
        return self.__mul__(scale)

    def __truediv__(self, scale: float):
        """Divides color by a scalar."""
        if isinstance(scale, (int, float)) and scale != 0:
            return Color(self.r / scale, self.g / scale, self.b / scale, self.a)
        return NotImplemented

    def __repr__(self):
        return f"Color(r={self.r:.2f}, g={self.g:.2f}, b={self.b:.2f}, hex={self.to_hex()})"

@dataclass(slots=True)
class ColorGradient:
    colors: Color
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
        self.colors = np.array([c for p, c in sorted_pairs], dtype=Color)

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
    albedo: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))
    roughness: float = 0.7
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
            emission_color=color,
            emission_intensity=intensity,
            type=MaterialType.EMISSIVE,
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
        accumulated_light = Color(0.0, 0.0, 0.0)

        for light in scene_lights:
            # --- Light Calculation ---
            light_dir = light.get_light_direction(hit_point)
            dist = np.linalg.norm(light.position - hit_point)
            
            # Optimization: Skip lights that are too close (singularities)
            if dist <= bias: continue
            
            # --- B. Visibility Check (Shadows) ---
            # If the light is blocked by another object, we skip it.
            visibility = visibility_function(hit_point, light.position)
            if visibility <= 0.0:
                continue

            # --- C. Lighting Calculations ---
            # 1. Attenuation: Light gets weaker over distance (Inverse Square Law)
            attenuation = attenuate_sqr_distance(dist)
            
            # 2. Cosine Law: Light hits weaker at glancing angles
            NdotL = max(0.0, np.dot(surface_normal, light_dir))
            
            # If light is behind the surface, skip (unless it's translucent, handled separately)
            if NdotL <= 0.0:
                continue

            # 3. Incoming Radiance (Li): Intensity * Attenuation * Visibility
            incoming_intensity = light.intensity * attenuation * visibility

            # Evaluate BRDF using light color + incoming intensity
            brdf_color = self.evaluate_brdf(light.color, incoming_intensity, light_dir, surface_normal, view_dir)
            accumulated_light += brdf_color
            print(accumulated_light)
            
        return accumulated_light

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
        safe_roughness = max(self.data.roughness, bias)
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
        sigma = (Color(1.0, 1.0, 1.0) - self.data.absorption_color) * self.data.absorption_density
        
        # Calculate attenuation per channel
        # e.g. If Red has high sigma, exp(-high * dist) -> 0.0 (Red is blocked)
        attenuation = attenuate_distance_exponential(distance, sigma)
        
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

    def evaluate_brdf(self, light_color: Color, light_intensity: float, light_dir: np.ndarray, surface_normal: np.ndarray, view_dir: np.ndarray, bias: float = 1e-4) -> Color:
        """
        Returns the BRDF value (ratio of radiance).
        """
        H = (view_dir + light_dir)
        # Safety check for degenerate vectors
        h_len = np.linalg.norm(H)
        if h_len < bias:
            return Color(0.0, 0.0, 0.0)
        H = H / h_len

        # Clamp dot products to avoid negative values (lighting from behind surface)
        VdotH = max(np.dot(view_dir, H), bias)

        # Compute PBR Terms
        specular_term = self.get_specular_component(light_color, light_intensity, light_dir, surface_normal, view_dir)
        diffuse_term = self.get_diffuse_component(light_color, light_intensity, light_dir, surface_normal) * (1.0/np.pi)

        return diffuse_term + specular_term

    def __repr__(self):
        return (
            f"Material(albedo={self.data.albedo}, other={self.data})"
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
    unit_incident = unit(incident_dir)
    unit_normal = unit(normal)
    
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
