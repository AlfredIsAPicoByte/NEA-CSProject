from __future__ import annotations
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List

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
    dist_x: float, 
    dist_y: float
) -> float:
    """
    Calculates the weight of a sample based on its distance from the pixel center.
    dist_x, dist_y are in 'pixel units' (not normalized 0-1).
    """
    # 1. Check bounds (if sample is outside filter radius, weight is 0)
    # Note: Box filters usually ignore radius or use 0.5, but generic filters use a radius.
    if abs(dist_x) > settings.filter_width or abs(dist_y) > settings.filter_width:
        return 0.0

    ftype = settings.filter_type

    # --- BOX FILTER ---
    # All samples within the pixel square get equal weight.
    if ftype == PixelFilter.BOX:
        # Standard Box is usually just 1.0 inside the boundary
        return 1.0

    # --- TENT FILTER (Triangle) ---
    # Linear falloff: 1.0 at center, 0.0 at radius
    elif ftype == PixelFilter.TENT:
        wx = max(0.0, 1.0 - abs(dist_x) / settings.filter_width)
        wy = max(0.0, 1.0 - abs(dist_y) / settings.filter_width)
        return wx * wy

    # --- GAUSSIAN FILTER ---
    # Exponential falloff: Creates very smooth images
    elif ftype == PixelFilter.GAUSSIAN:
        alpha = 2.0  # Controls "pointiness" of the bell curve
        exp_x = math.exp(-alpha * (dist_x * dist_x)) - math.exp(-alpha * (settings.filter_width**2))
        exp_y = math.exp(-alpha * (dist_y * dist_y)) - math.exp(-alpha * (settings.filter_width**2))
        return max(0.0, exp_x * exp_y)

    return 1.0

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
    
    # Accumulators
    final_color = np.array([0.0, 0.0, 0.0])
    total_weight = 0.0
    
    # Center of the pixel in normalized coordinates (0.0 to 1.0)
    # We add 0.5 to target the *center* of the pixel grid cell
    center_u = (pixel_x + 0.5) / settings.width
    center_v = (pixel_y + 0.5) / settings.height

    for sample, color in zip(samples, colors):
        # 1. Calculate distance from pixel center in PIXEL UNITS
        #    (We multiply by width/height to convert 0..1 back to 0..800)
        dist_x = (sample.u - center_u) * settings.width
        dist_y = (sample.v - center_v) * settings.height

        # 2. Get the filter weight (how much this sample contributes)
        #    We multiply by sample.w to respect adaptive sampling weights if they exist
        filter_w = evaluate_filter_weight(settings, dist_x, dist_y)
        combined_weight = filter_w * sample.w

        # 3. Accumulate
        if combined_weight > 0:
            final_color += color * combined_weight
            total_weight += combined_weight

    # 4. Normalize
    #    If total_weight is 0 (e.g. no samples hit), return black or ambient
    if total_weight > 0:
        return final_color / total_weight
    else:
        return np.array([0.0, 0.0, 0.0])

class Sampler(ABC):
    """
    Sampler abstraction used by render algorithms.
    - start_pixel: set pixel/tile context (used for per-pixel seeding, stratification offsets)
    - next_1d/next_2d: supply samples in [0,1)
    - clone: produce independent sampler for thread/worker
    """
    def __init__(self, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None):
        self.seed = seed
        self.settings = sample_settings

    @abstractmethod
    def start_pixel(self, x: int, y: int) -> None:
        ...

    @abstractmethod
    def next_1d(self) -> float:
        ...

    @abstractmethod
    def next_2d(self) -> Tuple[float, float]:
        ...

    @abstractmethod
    def clone(self, seed: Optional[int] = None) -> "Sampler":
        ...

    @abstractmethod
    def sample_pixel(self, x: int, y: int, sample_idx: int) -> tuple[float, float]:
        ...
    
    def set_samples_per_pixel(self, spp: int) -> None:
        self.settings.samples_per_pixel = max(spp, 1)

    def get_samples_per_pixel(self, x: int, y:int) -> List[Sample]:
        """Utility to get all samples for a pixel as Sample(u,v) list."""
        self.start_pixel(x, y)
        out: List[Sample] = []
        for _ in range(self.settings.samples_per_pixel):
            u, v = self.next_2d()
            out.append(Sample(u, v))
        return out

    def get_samples_for_region(self, region: Tuple[int, int, int, int]) -> List[Sample]:
        """Utility to get all samples for a region (x0,y0,x1,y1) as Sample(u,v) list."""
        x0, y0, x1, y1 = region
        out: List[Sample] = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                self.start_pixel(x, y)
                for _ in range(self.samples_per_pixel):
                    u, v = self.next_2d()
                    out.append(Sample(u, v))
        return out

    def get_sample_by_index(self, index: int, image_width: int, image_height: int) -> Sample:
        """Utility to get a single sample by linear index over the image."""
        pixel_index = index // self.settings.samples_per_pixel
        sample_index = index % self.settings.samples_per_pixel

        x = pixel_index % image_width
        y = pixel_index // image_width
        self.start_pixel(x, y)
        for _ in range(sample_index + 1):
            u, v = self.next_2d()
        return Sample(u, v)

class RandomSampler(Sampler):
    """Independent RNG sampler, deterministic with base seed + pixel coords."""
    def __init__(self, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None):
        super().__init__(sample_settings, seed)
        self._rng = np.random.default_rng(seed)
        self._x = 0
        self._y = 0

    def start_pixel(self, x: int, y: int) -> None:
        self._x = x; self._y = y

    def next_1d(self) -> float:
        return self._rng.random()

    def next_2d(self) -> Tuple[float, float]:
        return (self._rng.random(), self._rng.random())

    def clone(self, seed: Optional[int] = None) -> "RandomSampler":
        return RandomSampler(self.settings, seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> tuple[float, float]:
        self.start_pixel(x, y)
        for _ in range(sample_idx + 1):
            dx, dy = self.next_2d()
        return dx, dy

class StratifiedSampler(Sampler):
    """Stratified 2D sampler (square grid) with per-cell jitter."""
    def __init__(self, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None):
        super().__init__(sample_settings, seed)
        self._rng = np.random.default_rng(seed)
        self._x = 0
        self._y = 0
        self._current = 0
        self._rebuild_grid()

    def _rebuild_grid(self):
        n = max(1, int(math.ceil(math.sqrt(self.settings.samples_per_pixel))))
        self._nx = n
        self._ny = n

    def set_samples_per_pixel(self, spp: int) -> None:
        super().set_samples_per_pixel(spp)
        self._rebuild_grid()

    def start_pixel(self, x: int, y: int) -> None:
        self._x = x; self._y = y
        self._current = 0

    def next_1d(self) -> float:
        return self._rng.random()

    def next_2d(self) -> Tuple[float, float]:
        i = self._current % self._nx
        j = self._current // self._nx
        if j >= self._ny:
            # exhausted, fall back to random
            return (self._rng.random(), self._rng.random())
        u = (i + self._rng.random()) / self._nx
        v = (j + self._rng.random()) / self._ny
        self._current += 1
        return (u, v)

    def clone(self, seed: Optional[int] = None) -> "StratifiedSampler":
        return StratifiedSampler(self.settings, seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> tuple[float, float]:
        self.start_pixel(x, y)
        n = max(1, int(math.ceil(math.sqrt(self.settings.samples_per_pixel))))
        i = sample_idx % n
        j = sample_idx // n
        if j >= n:
            return (self._rng.random(), self._rng.random())
        dx = (i + self._rng.random()) / n
        dy = (j + self._rng.random()) / n
        return dx, dy

class HaltonSampler(Sampler):
    def __init__(self, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None):
        super().__init__(sample_settings, seed)
        self._index = 0

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

    def start_pixel(self, x: int, y: int) -> None:
        # per-pixel offset can be applied; keep simple
        self._index = 1

    def next_1d(self) -> float:
        v = self._halton(self._index, 2)
        self._index += 1
        return v

    def next_2d(self) -> Tuple[float, float]:
        v = (self._halton(self._index, 2), self._halton(self._index, 3))
        self._index += 1
        return v

    def clone(self, seed: Optional[int] = None) -> "HaltonSampler":
        return HaltonSampler(self.samples_per_pixel, seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> tuple[float, float]:
        # Use sample_idx+1 to avoid zero index in Halton
        u = self._halton(sample_idx + 1, 2)
        v = self._halton(sample_idx + 1, 3)
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
    """
    High-level manager combining settings and sampler use.
    """
    def __init__(self, sample_settings: 'SampleSettings', sampler_name: str | None = None, seed: Optional[int] = None, precompute: bool = False):
        self.settings = sample_settings
        self.seed = seed
        self.sampler_name = sampler_name if sampler_name is not None else "random"
        
        self._spp = self.settings.samples_per_pixel
        
        self._sampler = create_sampler(self.sampler_name, sample_settings, seed)

        self._precompute = bool(precompute)
        self.precomputed_samples: List[Sample] = []
        
        if self._precompute:
            self._do_precompute()

    def _do_precompute(self):
        """Generates all samples upfront. Warning: High Memory Usage."""
        print(f"Precomputing samples for {self.settings.width}x{self.settings.height} image...")
        self.precomputed_samples = []

        # Use a single local sampler instance for the loop
        # We typically clone once at the start to ensure thread safety if this runs in parallel
        if hasattr(self._sampler, "clone"):
             local_sampler = self._sampler.clone(self.seed)
        else:
             local_sampler = self._sampler

        for y in range(self.settings.height):
            for x in range(self.settings.width):
                local_sampler.start_pixel(x, y)
                for _ in range(self._spp):
                    u, v = local_sampler.next_2d()
                    self.precomputed_samples.append(Sample(u, v))

    def get_samples_per_pixel(self, x: int, y: int) -> List[Sample]:
        """
        Returns the list of samples (u,v) for a specific pixel coordinate.
        """
        # --- PATH A: Precomputed (Fast access, High RAM) ---
        if self._precompute:
            # Calculate flat index
            # FIX: Corrected typo 'samples_per_pixel' -> 'samples_per_pixel'
            pixel_index = (y * self.settings.width) + x
            start_idx = pixel_index * self._spp
            
            # Safety Check
            if start_idx + self._spp > len(self.precomputed_samples):
                return [] 
                
            return self.precomputed_samples[start_idx : start_idx + self._spp]

        # --- PATH B: On-Demand (Low RAM, Slightly Slower) ---
        
        # Avoid cloning per-pixel if possible. 
        # Ideally, samplers are lightweight. If 'clone' is expensive, this line is a bottleneck.
        # However, for thread safety in a renderer, we often MUST clone or use thread-local storage.
        if hasattr(self._sampler, "clone"):
            current_sampler = self._sampler.clone(self.seed)
        else:
            current_sampler = self._sampler

        # Reset sampler state for this specific pixel
        try:
            current_sampler.start_pixel(x, y)
        except Exception:
            pass

        out: List[Sample] = []
        for _ in range(self._spp):
            try:
                u, v = current_sampler.next_2d()
            except Exception:
                # 4. FIX: Correct Numpy Syntax
                u = np.random.random()
                v = np.random.random()
            
            out.append(Sample(u, v))
            
        return out

    def __repr__(self):
        return f"SamplingManager(sampler={self.sampler_name}, spp={self._spp})"