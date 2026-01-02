import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter, uniform_filter

from Sampling import SampleSettings, Sample
from Luminance import clamp

@dataclass
class BloomSettings:
    enabled: bool = False
    threshold: float = 1.0
    softness: float = 0.5
    intensity: float = 0.8
    radius: int = 4
    fast: bool = True

@dataclass
class ChromaticAberrationSettings:
    enabled: bool = False
    strength: float = 0.005 # strength is usually a small float, not int

@dataclass
class VignetteSettings:
    enabled: bool = False
    strength: float = 0.5
    curve: float = 1.0

@dataclass
class AcesToneMapSettings:
    enabled: bool = True

@dataclass
class CustomToneMapSettings:
    enabled: bool = False
    # ACES approximate constants (Knarkowicz)
    a: float = 2.51
    b: float = 0.03
    c: float = 2.43
    d: float = 0.59
    e: float = 0.14

@dataclass
class GammaSettings:
    enabled: bool = True
    gamma: float = 2.2

@dataclass
class PostProcessingSettings:
    # We use default_factory to ensure every new instance gets its own fresh settings
    bloom: BloomSettings = field(default_factory=BloomSettings)
    chromatic_abberation: ChromaticAberrationSettings = field(default_factory=ChromaticAberrationSettings)
    vignette: VignetteSettings = field(default_factory=VignetteSettings)
    aces_tone_map: AcesToneMapSettings = field(default_factory=AcesToneMapSettings)
    custom_tone_map: CustomToneMapSettings = field(default_factory=CustomToneMapSettings)
    gamma_correction: GammaSettings = field(default_factory=GammaSettings)

class PostProcessingPipeline:
    @staticmethod
    def reconstruct_from_samples(
        width: int,
        height: int,
        pixel_samples_and_colors: List[List[Tuple[Sample, np.ndarray]]], # Using string 'Sample' for forward ref
        settings: SampleSettings
    ) -> np.ndarray:
        """
        Phase 1: Film Reconstruction.
        Converts the raw list of ray samples into a coherent 2D floating-point image
        using the specific PixelFilter (Box, Gaussian, etc.) defined in settings.
        """
        # Initialize floating point buffer (Linear Color Space)
        img_array = np.zeros((height, width, 3), dtype=np.float32)

        # We assume reconstruct_pixel is available from your sampling module
        # If not, ensure it is imported at the top of this file.
        from Sampling import reconstruct_pixel 

        # Iterate through pixels
        # Note: In a highly optimized engine, this loop would be C++ or Numba.
        # For Python, this is the bottleneck, but it is accurate.
        for y in range(height):
            for x in range(width):
                pixel_idx = y * width + x
                
                # Safety check for bounds
                if pixel_idx >= len(pixel_samples_and_colors):
                    continue

                samples_data = pixel_samples_and_colors[pixel_idx]
                
                if not samples_data:
                    # Black pixel if no rays hit
                    continue

                # Unpack the list of tuples [(Sample, ColorArray), ...]
                samples = [s[0] for s in samples_data]
                colors = [s[1] for s in samples_data] # Expecting np.array([r,g,b])

                # Reconstruct single pixel color
                final_color = reconstruct_pixel(x, y, samples, colors, settings)
                
                img_array[y, x] = final_color

        return img_array

    @staticmethod
    def apply_bloom(img_array: np.ndarray, threshold: float = 0.8, softness: float = 0.1, intensity: float = 0.5, radius: int = 4, fast: bool = True) -> np.ndarray:
        """
        Phase 2: Bloom.
        Uses a Separable Box Blur (Two-Pass) to approximate Gaussian Blur efficiently.
        Pass 1: Blur Horizontal. Pass 2: Blur Vertical.
        O(W*H) complexity instead of O(W*H*radius^2).
        """
        # 1. Extract Bright Pass (Soft Threshold)
        # Use simple luminance dot product
        luminance = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        
        # Soft knee threshold calculation
        knee = clamp(softness, 0, 1)
        soft = np.maximum(luminance - threshold + knee, 0)
        soft = soft / (2 * knee + 1e-5) # Normalize
        weight = np.maximum(soft, np.maximum(luminance - threshold, 0) / (np.maximum(luminance, 1e-5)))
        
        # Apply mask to create bright pass
        bright_pass = img_array * weight[..., np.newaxis]
        blurred_pass = np.zeros_like(bright_pass)
        
        # 2. Blur the Bright Pass
        if fast:
            b = bright_pass
            for _ in range(3):
                b = uniform_filter(b, size=radius*2, mode='reflect')
            blurred_pass = b
        else:
            blurred_pass = gaussian_filter(bright_pass, sigma=radius)

        # 3. Additive Blending
        return img_array + (blurred_pass * intensity)

    @staticmethod
    def apply_chromatic_aberration(img_array: np.ndarray, strength: int = 2) -> np.ndarray:
        """
        Phase 2: Lens Imperfections.
        int(strength)s Red and Blue channels in opposite directions.
        """
        if int(strength) == 0: return img_array
        
        out_img = np.copy(img_array)
        
        # int(strength) Red Left
        out_img[:, :-int(strength), 0] = img_array[:, int(strength):, 0]
        out_img[:, -int(strength):, 0] = img_array[:, -1:, 0] # Clamp edge
        
        # int(strength) Blue Right
        out_img[:, int(strength):, 2] = img_array[:, :-int(strength), 2]
        out_img[:, :int(strength), 2] = img_array[:, :1, 2]   # Clamp edge
        
        return out_img

    @staticmethod
    def apply_vignette(img_array: np.ndarray, strength: float = 0.5, curve: float = 1.0) -> np.ndarray:
        """Darkens corners to simulate lens barrel."""
        height, width, _ = img_array.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        radius = np.sqrt(X**2 + Y**2)
        # 1.0 at center, drops to 1-strength at corners
        vignette = 1.0 - np.clip(radius * strength, 0, 1) ** curve
        
        return img_array * vignette[..., np.newaxis]

    @staticmethod
    def aces_tone_map(img_array: np.ndarray) -> np.ndarray:
        """
        Phase 3: Tone Mapping.
        ACES Filmic Tone Mapping approximation (Narkowicz).
        Compresses High Dynamic Range (HDR) values > 1.0 into 0.0-1.0 range nicely.
        """
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return np.clip((img_array * (a * img_array + b)) / (img_array * (c * img_array + d) + e), 0.0, 1.0)

    @staticmethod
    def custom_tone_map(img_array: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
        return np.clip((img_array * (a * img_array + b)) / (img_array * (c * img_array + d) + e), 0.0, 1.0)

    @staticmethod
    def gamma_correct(img_array: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        Phase 4: Gamma Correction.
        Converts Linear Physics space to sRGB Monitor space.
        """
        return np.power(np.maximum(img_array, 0.0), 1.0 / gamma)

    @classmethod
    def process_and_export(
            cls, 
            pixel_samples: List[List[Tuple[Sample, np.ndarray]]], 
            sample_settings: SampleSettings,
            pipeline_settings: PostProcessingSettings
            
        ) -> np.ndarray:
        """
        Master function to run the entire pipeline from Raw Samples -> Saved 8-bit Image.
        """
        width = sample_settings.width
        height = sample_settings.height
        
        # 1. Reconstruct (Raw Samples -> Float Image)
        print(" > Reconstructing image from samples")
        img = cls.reconstruct_from_samples(width, height, pixel_samples, sample_settings)
        
        # 2. Effects (Linear Space)
        print(" > Applying visual effects")
        if pipeline_settings.bloom.enabled:
            print("Applying Bloom...")
            img = cls.apply_bloom(
                img, 
                threshold=pipeline_settings.bloom.threshold,
                intensity=pipeline_settings.bloom.intensity,
                softness=pipeline_settings.bloom.softness,
                radius=pipeline_settings.bloom.radius,
                fast=pipeline_settings.bloom.fast
            )

        if pipeline_settings.chromatic_abberation.enabled:
            print("Applying Chromatic Aberration...")
            img = cls.apply_chromatic_aberration(
                img,
                strength=pipeline_settings.chromatic_abberation.strength
            )

        if pipeline_settings.vignette.enabled:
            print("Applying Vignette...")
            img = cls.apply_vignette(
                img,
                strength=pipeline_settings.vignette.strength,
                curve=pipeline_settings.vignette.curve,
            )

        # 3. Tone Mapping (HDR -> LDR)
        print(" > Tone Mapping")
        if pipeline_settings.aces_tone_map.enabled:
            print("Applying ACES tone mapping...")
            img = cls.aces_tone_map(img)
        elif pipeline_settings.custom_tone_map.enabled:
            print("Applying custom tone mapping...")
            img = cls.custom_tone_map(
                img, 
                a=pipeline_settings.custom_tone_map.a,
                b=pipeline_settings.custom_tone_map.b,
                c=pipeline_settings.custom_tone_map.c,
                d=pipeline_settings.custom_tone_map.d,
                e=pipeline_settings.custom_tone_map.e
            )

        # 4. Gamma Correction
        print(" > Gamma Correcting")
        if pipeline_settings.gamma_correction.enabled:
            img = cls.gamma_correct(
                img,
                gamma=pipeline_settings.gamma_correction.gamma
            )

        # 5. Quantize (Float -> 8-bit Int)
        print(" > Quantizing")
        img_8bit = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        return img_8bit