from __future__ import annotations
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List, Union

class PixelFilter(Enum):
    BOX = 0
    TENT = 1
    GAUSSIAN = 2

@dataclass
class SampleSettings:
    width: int = 800
    height: int = 600
    samples_per_pixel: int = 1
    filter_type: PixelFilter = PixelFilter.BOX
    filter_width: float = 2.0  # Radius in pixels for the filter

    def __post_init__(self):
        # 1. Validate Dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Resolution must be positive. Got {self.width}x{self.height}")

        # 2. Validate Sampling
        if self.samples_per_pixel < 1:
            raise ValueError(f"Samples per pixel must be >= 1. Got {self.samples_per_pixel}")
        
        # 3. Validate Filter Width
        if self.filter_width <= 0:
            raise ValueError(f"Filter width must be positive. Got {self.filter_width}")

        # 4. Optional: Type Checking (Dataclasses don't enforce types by default)
        if not isinstance(self.filter_type, PixelFilter):
            raise TypeError(f"filter_type must be a PixelFilter Enum, got {type(self.filter_type)}")
    
@dataclass
class Sample:
    u: float
    v: float
    w: float = 1.0

def evaluate_filter_weight(
settings: SampleSettings, 
    dist_x: np.ndarray, 
    dist_y: np.ndarray
) -> np.ndarray:
    """
    Calculates weights for a batch of samples using NumPy broadcasting.
    """
    # 1. Mask out samples beyond filter radius
    # (Box usually ignores radius, but we clamp for safety)
    mask = (np.abs(dist_x) <= settings.filter_width) & (np.abs(dist_y) <= settings.filter_width)
    
    # Default weights (0.0 outside, placeholder inside)
    weights = np.zeros_like(dist_x)
    
    ftype = settings.filter_type

    if ftype == PixelFilter.BOX:
        # Box is 1.0 everywhere inside the mask
        weights[mask] = 1.0

    elif ftype == PixelFilter.TENT:
        # 1.0 at center, 0.0 at radius
        wx = 1.0 - (np.abs(dist_x[mask]) / settings.filter_width)
        wy = 1.0 - (np.abs(dist_y[mask]) / settings.filter_width)
        # Clip negative values just in case
        weights[mask] = np.maximum(0.0, wx) * np.maximum(0.0, wy)

    elif ftype == PixelFilter.GAUSSIAN:
        alpha = 2.0
        fw_sq = settings.filter_width**2
        
        # Helper for exp calc
        def gaussian_1d(d):
            return np.exp(-alpha * (d * d)) - math.exp(-alpha * fw_sq)

        gx = gaussian_1d(dist_x[mask])
        gy = gaussian_1d(dist_y[mask])
        weights[mask] = np.maximum(0.0, gx * gy)

    return weights

def reconstruct_pixel(
    pixel_x: int,
    pixel_y: int,
    samples: List[Sample],
    colors: List[np.ndarray], # RGB numpy arrays like [1.0, 0.5, 0.0]
    settings: SampleSettings
) -> np.ndarray:
    """
    Combines many samples into one final pixel color using the chosen filter.
    """
    if not samples:
        return np.array([0.0, 0.0, 0.0])

    # 1. Convert lists to NumPy arrays for speed
    # (N,) arrays for coordinates
    u_coords = np.array([s.u for s in samples])
    v_coords = np.array([s.v for s in samples])
    sample_weights = np.array([s.w for s in samples])
    
    # (N, 4) array for colors
    color_stack = np.stack(colors) 

    # 2. Calculate Distances (Pixel Units)
    # Center of the pixel is +0.5
    center_u = (pixel_x + 0.5) / settings.width
    center_v = (pixel_y + 0.5) / settings.height
    
    dist_x = (u_coords - center_u) * settings.width
    dist_y = (v_coords - center_v) * settings.height

    # 3. Calculate Weights (Vectorized)
    filter_weights = evaluate_filter_weight(settings, dist_x, dist_y)
    
    # Combine filter weight with inherent sample weight
    final_weights = filter_weights * sample_weights

    # 4. Accumulate
    total_weight = np.sum(final_weights)
    
    if total_weight <= 0.0:
        return np.array([0.0, 0.0, 0.0, 0.0])
        
    # Broadcasting: (N, 1) * (N, 4) -> Sum over axis 0 -> (4,)
    weighted_colors = color_stack * final_weights[:, np.newaxis]
    final_color = np.sum(weighted_colors, axis=0)
    
    return final_color / total_weight

class Sampler:
    """
    Base Sampler. Can act as a persistent wrapper for an RNG.
    Compatible with both Pixel Generation (settings-aware) and Monte Carlo (rng-aware).
    """
    def __init__(
        self, 
        input_source: Union[SampleSettings, np.random.Generator, None] = None, 
        seed: Optional[int] = None
    ):
        # Handle Polymorphic Initialization
        self.settings = SampleSettings()
        self._rng: np.random.Generator = None # type: ignore

        # Case 1: Passed an existing RNG (High Perf mode for Raytracer)
        if isinstance(input_source, np.random.Generator):
            self._rng = input_source
        
        # Case 2: Passed Settings (Standard mode for Pixel Generation)
        elif isinstance(input_source, SampleSettings):
            self.settings = input_source
            self._rng = np.random.default_rng(seed)
            
        # Case 3: Default
        else:
            self._rng = np.random.default_rng(seed)

    def start_pixel(self, x: int, y: int) -> None:
        """Reset internal state for a new pixel (used by Stratified/Halton)."""
        pass

    def random_float(self) -> float:
        """Alias for next_1d, used by Raytracer."""
        return self._rng.random()

    def next_1d(self) -> float:
        return self._rng.random()

    def next_2d(self) -> Tuple[float, float]:
        return (self._rng.random(), self._rng.random())

    def clone(self, seed: Optional[int] = None) -> "Sampler":
        # Cloning usually resets to a generic RandomSampler unless overridden
        return RandomSampler(self.settings, seed)

    def set_samples_per_pixel(self, spp: int) -> None:
        self.settings.samples_per_pixel = max(spp, 1)

    def get_samples_per_pixel(self, x: int, y: int) -> List[Sample]:
        """Generate all samples for a specific pixel."""
        self.start_pixel(x, y)
        out = []
        for i in range(self.settings.samples_per_pixel):
            u, v = self.sample_pixel(x, y, i)
            out.append(Sample(u, v))
        return out

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> Tuple[float, float]:
        """Get the specific u,v for sample index i in pixel x,y."""
        return self.next_2d()
        
class RandomSampler(Sampler):
    """Pure random sampling (White Noise). Good for high sample counts."""
    def sample_pixel(self, x: int, y: int, sample_idx: int) -> Tuple[float, float]:
        # Return normalized coordinates in [0,1] across the full image
        return (
            (x + self._rng.random()) / float(self.settings.width),
            (y + self._rng.random()) / float(self.settings.height)
        )

class StratifiedSampler(Sampler):
    """
    Jittered Grid Sampling. Reduces noise by ensuring samples are well-distributed.
    
    """
    def __init__(self, settings: SampleSettings = SampleSettings(), seed: Optional[int] = None):
        super().__init__(settings, seed)
        self._rebuild_grid()

    def _rebuild_grid(self):
        # Determine grid size (NxN)
        self._grid_side = max(1, int(math.ceil(math.sqrt(self.settings.samples_per_pixel))))

    def set_samples_per_pixel(self, spp: int) -> None:
        super().set_samples_per_pixel(spp)
        self._rebuild_grid()

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> Tuple[float, float]:
        # Map linear index to grid coordinates
        # e.g., index 5 in a 3x3 grid might be row 1, col 2
        n = self._grid_side
        i = x + sample_idx % n
        j = y + sample_idx // n
        
        # If we run out of grid slots (spp > n*n), fall back to random
        if j >= n:
            return (self._rng.random(), self._rng.random())
            
        # Jitter within the grid cell
        jitter_x = self._rng.random()
        jitter_y = self._rng.random()
        
        # Normalize to 0..1 range
        u = (i + jitter_x) / n
        v = (j + jitter_y) / n
        
        return u, v

class HaltonSampler(Sampler):
    """Low Discrepancy Sequence. Good for progressive rendering."""
    
    @staticmethod
    def _halton(index: int, base: int) -> float:
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> Tuple[float, float]:
        # We add 1 because Halton(0) is always 0, which can be problematic at edges
        # We also mix in x/y to decorrelate adjacent pixels slightly (simple scramble)
        idx = sample_idx + 1 + (x * 499) + (y * 503) 
        u = self._halton(idx, 2)
        v = self._halton(idx, 3)
        return u, v

class AdaptiveSampler(Sampler):
    """Placeholder for an adaptive sampler implementation."""
    def __init__(self, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None):
        super().__init__(sample_settings, seed)
        # Implementation would go here

    def start_pixel(self, x: int, y: int) -> None:
        pass

    def next_1d(self) -> float:
        return np.random()

    def next_2d(self) -> Tuple[float, float]:
        return (np.random(), np.random())

    def clone(self, seed: Optional[int] = None) -> "AdaptiveSampler":
        return AdaptiveSampler(self.settings, seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> tuple[float, float]:
        return (np.random(), np.random())

# Registry and factory
_SAMPLER_REGISTRY: dict[str, type[Sampler]] = {
    "random": RandomSampler,
    "stratified": StratifiedSampler,
    "halton": HaltonSampler,
    "adaptive": AdaptiveSampler,
}

def create_sampler(name: str, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None) -> Sampler:
    cls = _SAMPLER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown sampler: {name}")
    return cls(sample_settings, seed)

class SamplingManager:
    """Factory and Manager for sampling strategies."""
    def __init__(self, sample_settings: SampleSettings, sampler_name: str = "random", seed: Optional[int] = None):
        self.settings = sample_settings
        self.sampler = create_sampler(sampler_name, sample_settings, seed)

    @property
    def _sampler(self):
        """Backward-compatible alias for the internal sampler instance."""
        return self.sampler

    def get_samples_per_pixel(self, x: int, y: int) -> List[Sample]:
        # Delegate to the strategy
        return self.sampler.get_samples_per_pixel(x, y)