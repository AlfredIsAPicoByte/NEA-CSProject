import numpy as np
from typing import List, Tuple, Dict, Any, Union
from scipy.ndimage import gaussian_filter, uniform_filter
from Sampling import SampleSettings, Sample

class PostProcessingPipeline:
    @staticmethod
    def reconstruct_from_samples(
        width: int,
        height: int,
        pixel_samples_and_colors: List[List[Tuple['Sample', np.ndarray]]], # Using string 'Sample' for forward ref
        settings: 'SampleSettings'
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
    def apply_bloom(img_array: np.ndarray, threshold: float = 0.8, intensity: float = 0.5, radius: int = 4, fast: bool = True) -> np.ndarray:
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
        knee = 0.1 # Softness
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
    def apply_chromatic_aberration(img_array: np.ndarray, strength: float = 2.0) -> np.ndarray:
        """
        Phase 2: Lens Imperfections.
        Shifts Red and Blue channels in opposite directions.
        """
        shift = int(strength)
        if shift == 0: return img_array
        
        out_img = np.copy(img_array)
        
        # Shift Red Left
        out_img[:, :-shift, 0] = img_array[:, shift:, 0]
        out_img[:, -shift:, 0] = img_array[:, -1:, 0] # Clamp edge
        
        # Shift Blue Right
        out_img[:, shift:, 2] = img_array[:, :-shift, 2]
        out_img[:, :shift, 2] = img_array[:, :1, 2]   # Clamp edge
        
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
    def gamma_correct(img_array: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        Phase 4: Gamma Correction.
        Converts Linear Physics space to sRGB Monitor space.
        """
        return np.power(np.maximum(img_array, 0.0), 1.0 / gamma)

    @classmethod
    def process_and_export(cls, 
                           pixel_samples: List[List[Tuple['Sample', np.ndarray]]], 
                           settings: 'SampleSettings',
                           pipeline_options: Dict[str, Any]
                           ) -> np.ndarray:
        """
        Master function to run the entire pipeline from Raw Samples -> Saved 8-bit Image.
        """
        width = settings.width
        height = settings.height
        
        # 1. Reconstruct (Raw Samples -> Float Image)
        print("Reconstructing image from samples...")
        img = cls.reconstruct_from_samples(width, height, pixel_samples, settings)
        
        # 2. Effects (Linear Space)
        if pipeline_options.get('bloom_enabled', True):
            print("Applying Bloom...")
            img = cls.apply_bloom(
                img, 
                threshold=pipeline_options.get('bloom_threshold', 0.9),
                intensity=pipeline_options.get('bloom_intensity', 0.4),
                radius=pipeline_options.get('bloom_radius', 3)
            )

        if pipeline_options.get('chromatic_aberration_enabled', False):
            print("Applying Chromatic Aberration...")
            img = cls.apply_chromatic_aberration(img, strength=2.0)

        if pipeline_options.get('vignette_enabled', True):
            print("Applying Vignette...")
            img = cls.apply_vignette(img, strength=0.4)

        # 3. Tone Mapping (HDR -> LDR)
        print("Tone Mapping...")
        img = cls.aces_tone_map(img)

        # 4. Gamma Correction
        print("Gamma Correcting...")
        img = cls.gamma_correct(img)

        # 5. Quantize (Float -> 8-bit Int)
        print("Quantizing...")
        img_8bit = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        return img_8bit