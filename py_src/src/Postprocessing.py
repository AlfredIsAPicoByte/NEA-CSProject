import numpy as np

class PostProcessingPipeline:
    @staticmethod
    def apply_bloom(img_array: np.ndarray, threshold: float = 0.8, intensity: float = 0.5, radius: int = 2) -> np.ndarray:
        """
        Simple Bloom effect: Extracts bright pixels, blurs them, and adds them back.
        Note: Real bloom uses Gaussian blur; this uses a fast box blur approximation.
        """
        height, width, _ = img_array.shape
        
        # 1. Extract bright parts (Thresholding)
        # We work on a copy to avoid altering original yet
        bright_pass = np.copy(img_array)
        
        # Calculate luminance (perceived brightness)
        luminance = np.dot(bright_pass[..., :3], [0.299, 0.587, 0.114])
        
        # Mask: Keep only pixels brighter than threshold
        mask = luminance < threshold
        bright_pass[mask] = 0  # Darken everything else
        
        # 2. Blur the bright pass (Box Blur approximation using integral images or simple convolution)
        # For simplicity in pure NumPy without Scipy/OpenCV:
        # We shift the image in 4 directions and average them to simulate a blur
        blurred = np.copy(bright_pass)
        
        # Accumulate shifts for blur effect
        for y_off in range(-radius, radius + 1):
            for x_off in range(-radius, radius + 1):
                if x_off == 0 and y_off == 0: continue
                
                # Roll circles pixel values around
                shifted = np.roll(bright_pass, shift=(y_off, x_off), axis=(0, 1))
                
                # Zero out the wrap-around artifacts (simple cleanup)
                if y_off > 0: shifted[:y_off, :] = 0
                if y_off < 0: shifted[y_off:, :] = 0
                if x_off > 0: shifted[:, :x_off] = 0
                if x_off < 0: shifted[:, x_off:] = 0
                
                blurred += shifted

        # Normalize the blur stack
        blurred = blurred / ((2 * radius + 1) ** 2)

        # 3. Additive blending (Screen-like blend)
        # Original + (Blurred Highlights * Intensity)
        final_img = img_array + (blurred * intensity)
        
        return final_img

    @staticmethod
    def apply_chromatic_aberration(img_array: np.ndarray, strength: float = 2.0) -> np.ndarray:
        """
        Simulates lens fringing by shifting Red and Blue channels in opposite directions.
        """
        height, width, _ = img_array.shape
        out_img = np.copy(img_array)
        
        # Shift Red channel LEFT
        r_shift = int(strength)
        red_channel = np.roll(img_array[..., 0], shift=-r_shift, axis=1)
        # Fix wrap-around artifact
        red_channel[:, -r_shift:] = img_array[:, -r_shift:, 0]
        
        # Shift Blue channel RIGHT
        b_shift = int(strength)
        blue_channel = np.roll(img_array[..., 2], shift=b_shift, axis=1)
        # Fix wrap-around artifact
        blue_channel[:, :b_shift] = img_array[:, :b_shift, 2]
        
        out_img[..., 0] = red_channel
        out_img[..., 2] = blue_channel
        
        return out_img

    @staticmethod
    def apply_vignette(img_array: np.ndarray, strength: float = 0.5, curve: float = 1.0) -> np.ndarray:
        """
        Darkens the corners of the image.
        """
        height, width, _ = img_array.shape
        
        # Create coordinate grid (-1 to 1)
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from center (radius)
        radius = np.sqrt(X**2 + Y**2)
        
        # Create vignette mask (1 at center, 0 at corners)
        # Formula: 1 - strength * radius^curve
        mask = 1.0 - np.clip(radius * strength, 0, 1) ** curve
        
        # Expand mask to 3 channels for multiplication
        mask = np.dstack((mask, mask, mask))
        
        return img_array * mask

    @staticmethod
    def aces_tone_map(img_array: np.ndarray) -> np.ndarray:
        """Vectorized ACES Tone Mapping for numpy arrays."""
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        return (img_array * (a * img_array + b)) / (img_array * (c * img_array + d) + e)

    @staticmethod
    def gamma_correct(img_array: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Vectorized Gamma Correction."""
        return np.power(np.maximum(img_array, 0), 1.0 / gamma)