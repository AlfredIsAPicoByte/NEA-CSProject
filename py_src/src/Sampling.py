from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List
import random
import math

class PixelFilter(Enum):
    BOX = 0
    TENT = 1
    GAUSSIAN = 2

@dataclass
class SampleSettings:
    width: int = 800               # image width in pixels
    height: int = 600              # image height in pixels
    filter_type: PixelFilter = PixelFilter.BOX
    filter_width: float = 1.0          # size of the filter kernel
    size: int = 16               # total samples per pixel (may be square for stratified)

@dataclass
class Sample:
    u: float                      # horizontal sample position in [0,1)
    v: float                      # vertical sample position in [0,1)
    w: float = 1.0                # sample weight (for adaptive sampling)
    width: int = 1
    height: int = 1

    def __init__(self, u: float, v: float, w: float = 1.0, width: int = 1, height: int = 1):
        self.u = u
        self.v = v
        self.w = w
        self.width = width
        self.height = height
    
    def get_sample_index(self, image_width: int, image_height: int) -> int:
        """Convert normalized (u,v) to a linear sample index."""
        x = min(int(self.u * image_width), image_width - 1)
        y = min(int(self.v * image_height), image_height - 1)
        return y * image_width + x

    def get_pixel_coords(self, image_width: int, image_height: int) -> Tuple[int, int]:
        """Convert normalized (u,v) to pixel coordinates."""
        x = min(int(self.u * image_width), image_width - 1)
        y = min(int(self.v * image_height), image_height - 1)
        return (x, y)

class Sampler(ABC):
    """
    Sampler abstraction used by render algorithms.
    - start_pixel: set pixel/tile context (used for per-pixel seeding, stratification offsets)
    - next_1d/next_2d: supply samples in [0,1)
    - clone: produce independent sampler for thread/worker
    """
    def __init__(self, samples_per_pixel: int = 1):
        self.samples_per_pixel = samples_per_pixel

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

    def get_samples_for_pixel(self, x: int, y:int) -> List[Sample]:
        """Utility to get all samples for a pixel as Sample(u,v) list."""
        self.start_pixel(x, y)
        out: List[Sample] = []
        for _ in range(self.samples_per_pixel):
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
        total_pixels = image_width * image_height
        pixel_index = index // self.samples_per_pixel
        sample_index = index % self.samples_per_pixel
        x = pixel_index % image_width
        y = pixel_index // image_width
        self.start_pixel(x, y)
        for _ in range(sample_index + 1):
            u, v = self.next_2d()
        return Sample(u, v)

    def sample_pixel(self, x: int, y: int, sample_idx: int, width: int, height: int) -> tuple[float, float]:
        """
        Returns a subpixel offset (dx, dy) in [0, 1) for the given pixel and sample index.
        Default implementation uses get_samples_for_pixel.
        """
        samples = self.get_samples_for_pixel(x, y)
        if sample_idx < len(samples):
            return (samples[sample_idx].u, samples[sample_idx].v)
        # fallback: random
        return (random.random(), random.random())

class RandomSampler(Sampler):
    """Independent RNG sampler, deterministic with base seed + pixel coords."""
    def __init__(self, samples_per_pixel: int = 1, seed: Optional[int] = None):
        super().__init__(samples_per_pixel)
        self._base_seed = 0 if seed is None else int(seed)
        self._rng = random.Random(self._base_seed)
        self._x = 0
        self._y = 0

    def start_pixel(self, x: int, y: int) -> None:
        self._x = x; self._y = y
        # Build a deterministic integer seed from base_seed and pixel coords.
        # random.seed only accepts None, int, float, str, bytes, bytearray.
        # Use a simple integer hashing with primes and mask to 32-bit to keep it stable.
        try:
            base = int(self._base_seed)
        except Exception:
            base = 0
        seed_val = (base * 73856093) ^ (x * 19349663) ^ (y * 83492791)
        seed_val &= 0xFFFFFFFF
        self._rng.seed(seed_val)

    def next_1d(self) -> float:
        return self._rng.random()

    def next_2d(self) -> Tuple[float, float]:
        return (self._rng.random(), self._rng.random())

    def clone(self, seed: Optional[int] = None) -> "RandomSampler":
        return RandomSampler(self.samples_per_pixel, seed if seed is not None else self._base_seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int, width: int, height: int) -> tuple[float, float]:
        self.start_pixel(x, y)
        for _ in range(sample_idx + 1):
            dx, dy = self.next_2d()
        return dx, dy

class StratifiedSampler(Sampler):
    """Stratified 2D sampler (square grid) with per-cell jitter."""
    def __init__(self, samples_per_pixel: int = 1, seed: Optional[int] = None):
        super().__init__(samples_per_pixel)
        self._base_seed = 0 if seed is None else int(seed)
        self._rng = random.Random(self._base_seed)
        self._x = 0
        self._y = 0
        self._current = 0
        self._rebuild_grid()

    def _rebuild_grid(self):
        n = max(1, int(math.ceil(math.sqrt(self.samples_per_pixel))))
        self._nx = n
        self._ny = n

    def set_samples_per_pixel(self, spp: int) -> None:
        super().set_samples_per_pixel(spp)
        self._rebuild_grid()

    def start_pixel(self, x: int, y: int) -> None:
        self._x = x; self._y = y
        self._current = 0
        self._rng.seed((self._base_seed, x, y))

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
        return StratifiedSampler(self.samples_per_pixel, seed if seed is not None else self._base_seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int, width: int, height: int) -> tuple[float, float]:
        self.start_pixel(x, y)
        n = max(1, int(math.ceil(math.sqrt(self.samples_per_pixel))))
        i = sample_idx % n
        j = sample_idx // n
        if j >= n:
            return (self._rng.random(), self._rng.random())
        dx = (i + self._rng.random()) / n
        dy = (j + self._rng.random()) / n
        return dx, dy

# Simple quasi-random (Halton) generator for demonstration
class HaltonSampler(Sampler):
    def __init__(self, samples_per_pixel: int = 1, seed: Optional[int] = None):
        super().__init__(samples_per_pixel)
        self._index = 0
        self._base_seed = 0 if seed is None else int(seed)

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
        return HaltonSampler(self.samples_per_pixel, seed if seed is not None else self._base_seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int, width: int, height: int) -> tuple[float, float]:
        # Use sample_idx+1 to avoid zero index in Halton
        u = self._halton(sample_idx + 1, 2)
        v = self._halton(sample_idx + 1, 3)
        return u, v

class AdaptiveSampler(Sampler):
    """Placeholder for an adaptive sampler implementation."""
    def __init__(self, samples_per_pixel: int = 1, seed: Optional[int] = None):
        super().__init__(samples_per_pixel)
        # Implementation would go here

    def start_pixel(self, x: int, y: int) -> None:
        pass

    def next_1d(self) -> float:
        return random.random()

    def next_2d(self) -> Tuple[float, float]:
        return (random.random(), random.random())

    def clone(self, seed: Optional[int] = None) -> "AdaptiveSampler":
        return AdaptiveSampler(self.samples_per_pixel, seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int, width: int, height: int) -> tuple[float, float]:
        return (random.random(), random.random())

# Registry and factory
_SAMPLER_REGISTRY: dict[str, type[Sampler]] = {
    "random": RandomSampler,
    "stratified": StratifiedSampler,
    "halton": HaltonSampler,
    "adaptive": AdaptiveSampler,
}

def create_sampler(name: str, samples_per_pixel: int = 1, seed: Optional[int] = None) -> Sampler:
    cls = _SAMPLER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown sampler: {name}")
    return cls(samples_per_pixel, seed)

class SamplingManager:
    """
    High-level manager combining settings and sampler use.
    Use get_samples_for_pixel(x,y) to obtain a list of Sample(u,v).
    """
    def __init__(self, sample_settings: SampleSettings, sampler_name: str | None = None, seed: Optional[int] = None, precompute: bool = False):
        self.sample_settings = sample_settings
        self.seed = seed
        self.sampler_name = sampler_name if sampler_name is not None else "random"
        self._spp = max(1, self.sample_settings.size)
        self._sampler = create_sampler(self.sampler_name, self._spp, seed)
        # Precompute samples if requested (disabled by default to save memory)
        self.samples: List[Tuple[float, float]] = []
        self._precompute = bool(precompute)
        if self._precompute:
            self._precompute_samples()
    
    def _precompute_samples(self):
        self.samples = []
        sampler = self._sampler.clone(self.seed)
        for y in range(self.sample_settings.height):
            for x in range(self.sample_settings.width):
                sampler.start_pixel(x, y)
                for _ in range(self._spp):
                    u, v = sampler.next_2d()
                    self.samples.append((u, v))

    def get_precomputed_samples(self, count: int = 0, offset: int = 0) -> List[Tuple[float, float]]:
        """Return a slice of the precomputed sample list."""
        if count <= 0:
            return self.samples[offset:]
        return self.samples[offset:offset + count]

    # Provide the interface the renderer expects: get_samples_for_pixel(x,y) -> List[Sample]
    def get_samples_for_pixel(self, x: int, y: int) -> List[Sample]:
        # If precomputed, slice out samples for this pixel
        if self._precompute:
            start = (y * self.sample_settings.width + x) * self._spp
            slice_uv = self.samples[start:start + self._spp]
            return [Sample(u, v) for (u, v) in slice_uv]

        # Otherwise generate on-demand from the underlying sampler (clone per-call for safety)
        sampler = self._sampler.clone(self.seed) if hasattr(self._sampler, "clone") else self._sampler
        try:
            sampler.start_pixel(x, y)
        except Exception:
            # best-effort
            pass

        out: List[Sample] = []
        for _ in range(self._spp):
            try:
                u, v = sampler.next_2d()
            except Exception:
                u = random.random()
                v = random.random()
            out.append(Sample(u, v))
        return out

    def __repr__(self):
        return f"SamplingManager(sampler={self.sampler_name}, spp={self._spp}, settings={self.sample_settings})"