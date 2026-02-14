import sys, os
import numpy as np
from PIL import Image

class Film:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Store arrays with shape (height, width, channels) for row-major (y,x) indexing
        self.accum_color = np.zeros((height, width, 3), dtype=np.float32)
        self.accum_weight = np.zeros((height, width), dtype=np.float32)

    def add_pixel_batch(self, x: int, y: int, color_sum: np.ndarray, weighted_sum: float):
        # Bounds check: valid pixel coordinates are 0 <= x < width and 0 <= y < height
        if 0 <= x < self.width and 0 <= y < self.height:
            # Use row-major indexing: [y, x]
            self.accum_color[y, x] += color_sum[:3]
            self.accum_weight[y, x] += weighted_sum
    
    def get_image(self) -> np.ndarray:
        natural_mask = self.accum_weight > 0

        result = np.zeros_like(self.accum_color)

        result[natural_mask] = (
            self.accum_color[natural_mask] / self.accum_weight[natural_mask][..., np.newaxis]
        )

        return result

    @classmethod
    def save(cls, pixles: np.ndarray, filename: str, verbose: bool = False):    
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # 1. Quantization (0.0-1.0 float -> 0-255 uint8)
        # Clip to ensure no math errors pushed us outside range
        final_pixels = (np.clip(pixles, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        # 2. Save
        img = Image.fromarray(final_pixels, 'RGB')
        img.save(filename)
        if verbose: print(f" > Saved image to {filename}")

    def __repr__(self):
        return f"Film({self.width}x{self.height}, accum_pixels={self.accum_color.shape}, accum_weight={self.accum_weight.shape}, first_pixel={self.accum_color[0, 0]}, first_weight={self.accum_weight[0, 0]}, center_pixel={self.accum_color[self.height//2, self.width//2]}, center_weight={self.accum_weight[self.height//2, self.width//2]}, last_pixel={self.accum_color[-1, -1]}, last_weight={self.accum_weight[-1, -1]})"