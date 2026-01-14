import numpy as np
from scipy.ndimage import gaussian_filter

from src.Data.Ratio import Ratio
from .Pipeline import PostProcessPass

class ACESFilmicToneMapping(PostProcessPass):
    """
    The industry standard for photorealistic tone mapping.
    Maps HDR values to a filmic curve (pleasing contrast and saturation).
    """
    def apply(self, image: np.ndarray) -> np.ndarray:
        # ACES approximation
        # (Input must be linear, this outputs roughly linear 0-1 range that looks 'filmic')
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return np.clip((image * (a * image + b)) / (image * (c * image + d) + e), 0.0, 1.0)

class ReinhardToneMapping(PostProcessPass):
    """
    Simple tone mapping. Good for preserving details but desaturates highlights.
    Formula: Color / (1 + Color)
    """
    def apply(self, image: np.ndarray) -> np.ndarray:
        return image / (1.0 + image)

class GammaCorrection(PostProcessPass):
    """
    Converts Linear Color to sRGB for monitor display.
    """
    def __init__(self, gamma: float = 2.2):
        self.gamma = gamma
        self.inv_gamma = 1.0 / gamma

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Add epsilon to avoid log(0) errors if using logs, 
        # but power is usually safe on 0.
        return np.power(np.clip(image, 0.0, 1.0), self.inv_gamma)

class Exposure(PostProcessPass):
    """
    Adjusts brightness before tone mapping (like ISO on a camera).
    """
    def __init__(self, exposure_value: float = 1.0):
        self.exposure = exposure_value

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image * self.exposure

class AutoExposure(PostProcessPass):
    """
    Calculates average brightness and scales image to target 18% gray.
    (Simple implementation)
    """
    def apply(self, image: np.ndarray) -> np.ndarray:
        # Calculate geometric mean of luminance
        # 1. Convert to Luminance (perceived brightness)
        luminance = np.dot(image, [0.2126, 0.7152, 0.0722])
        avg_lum = np.mean(luminance)
        
        if avg_lum < 1e-4: 
            return image # Avoid divide by zero for black images
            
        # Target middle gray (0.18)
        key_value = 0.18
        exposure = key_value / avg_lum
        
        return image * exposure

class Bloom(PostProcessPass):
    def __init__(self, threshold: float = 1.0, radius: float = 10.0, intensity: float = 0.5, softness: float = 0.5):
        self.threshold = threshold
        self.radius = radius
        self.intensity = intensity
        self.softness = softness

    def apply(self, image: np.ndarray) -> np.ndarray:
        # 1. Isolation (Thresholding)
        luminance = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        
        # Soft knee threshold calculation
        knee = np.clip(self.softness, 0, 1)
        soft = np.maximum(luminance - self.threshold + knee, 0)
        soft = soft / (2 * knee + 1e-5) # Normalize
        weight = np.maximum(soft, np.maximum(luminance - self.threshold, 0) / (np.maximum(luminance, 1e-5)))
        
        # Apply mask to create bright pass
        bright_pass = image * weight[..., np.newaxis]
        
        # 2. Blur (Gaussian)
        # sigma is the standard deviation for Gaussian kernel
        # We blur each channel (R, G, B) independently.
        # The 'order=0' means simple gaussian smoothing.
        blurred_glow = np.zeros_like(bright_pass)

        for i in range(3): # Loop R, G, B
            blurred_glow[:, :, i] = gaussian_filter(
                bright_pass[:, :, i], 
                sigma=self.radius
            )

        # 3. Additive Blending
        # Original + (Glow * Intensity)
        return image + (blurred_glow * self.intensity)

class Vignette(PostProcessPass):
    def __init__(self, intensity: float = 0.5, softness: float = 0.4, curve: float = 1.0):
        self.intensity = intensity
        self.softness = softness
        self.curve = curve

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        
        # 1. Create a coordinate grid from -1 to 1
        y, x = np.ogrid[:height, :width]
        
        # Normalize coordinates to range [-0.5, 0.5]
        x = (x / width) - 0.5
        y = (y / height) - 0.5
        
        # 2. Calculate Distance from center
        # Correct for aspect ratio to ensure circular vignette
        try:
            aspect = width / height
        except ZeroDivisionError:
            aspect = 1.0
            
        x_corrected = x * aspect
        
        # Calculate radius from center
        radius = np.sqrt(x_corrected**2 + y**2)
        
        # Normalize radius roughly so edges are near 1.0
        # (multiplying by ~1.5 to 2.0 usually scales it nicely to the corners)
        radius = radius * 1.5 
        
        # 3. Compute Falloff
        vignette = 1.0 - (self.intensity * (radius * (1.0 + self.softness))) ** self.curve
        
        # Clamp
        vignette = np.clip(vignette, 0.0, 1.0)
        
        # 4. Multiply
        return image * vignette[..., np.newaxis]

class ChromaticAberration(PostProcessPass):
    def __init__(self, intensity: float = 0.005):
        self.intensity = intensity # Percentage of screen width to shift

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width, channels = image.shape
        
        # Calculate shift in pixels
        shift_amount = int(width * self.intensity)
        
        # Early exit if shift is negligible
        if shift_amount == 0:
            return image

        # Split channels
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        
        # Apply Shifts using np.roll (Linear shift)
        # Shift Red channel LEFT
        r_shifted = np.roll(r, shift=-shift_amount, axis=1)
        # (Optional) Fix the "wrap around" effect of roll by blacking out the edge
        r_shifted[:, -shift_amount:] = 0 
        
        # Green stays centered
        g_shifted = g 
        
        # Shift Blue channel RIGHT
        b_shifted = np.roll(b, shift=shift_amount, axis=1)
        # (Optional) Fix wrap around
        b_shifted[:, :shift_amount] = 0
        
        # Merge channels back
        # Ensure we use np.stack (new array) instead of modifying in place
        return np.stack([r_shifted, g_shifted, b_shifted], axis=2)