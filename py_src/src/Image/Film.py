import sys, os
import numpy as np
from typing import List, Tuple
from PIL import Image

from src.Utilities.Sampling import Sample
from src.Data.Color import Color

class Film:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.accum_color = np.zeros((width, height, 3), dtype=np.float32)
        self.accum_weight = np.zeros((width, height), dtype=np.float32)

    def add_pixle_batch(self, x: int, y: int, color_sum: np.ndarray, weighted_sum: float):
        if 0 <= x <= self.width and 0 <= y <= self.height:
            self.accum_color[x, y] += color_sum
            self.accum_weight[x, y] += weighted_sum
    
    def get_image(self) -> np.ndarray:
        natural_mask = self.accum_weight > 0

        result = np.zeros_like(self.accum_color)

        result[natural_mask] = (
            self.accum_color[natural_mask] / self.accum_weight[natural_mask][..., np.newaxis]
        )

        return result

    @classmethod
    def save(cls, pixles: np.ndarray, filename: str):    
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # 1. Quantization (0.0-1.0 float -> 0-255 uint8)
        # Clip to ensure no math errors pushed us outside range
        final_pixels = (np.clip(pixles, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        # 2. Save
        img = Image.fromarray(final_pixels, 'RGB')
        img.save(filename)
        print(f" > Saved to {filename}")