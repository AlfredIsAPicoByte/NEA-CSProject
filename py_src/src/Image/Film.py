import numpy as np
from typing import List, Tuple

from src.Utilities.Sampling import Sample
from src.Data.Color import Color

class Film:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # A flat list of lists.
        # Index = y * width + x
        # Content = List of (Sample, ColorArray) tuples
        self.samples: List[List[Tuple[Sample, np.ndarray]]] = [[] for _ in range(width * height)]

    def add_sample(self, x: int, y: int, sample: Sample, color: Color):
        """
        Thread-safe storage of a raw sample.
        """
        # Convert Color object to simple numpy array [r, g, b] for the pipeline
        col_array = np.array([color.r, color.g, color.b], dtype=np.float32)
        
        idx = y * self.width + x
        self.samples[idx].append((sample, col_array))

    def get_raw_data(self):
        """Pass this to the PostProcessingPipeline."""
        return self.samples