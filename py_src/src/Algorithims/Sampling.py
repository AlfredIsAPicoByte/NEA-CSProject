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
    
    pixels_covered: int = 1        # number of pixels this sample covers (for adaptive sampling)

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

    def set_samples_per_pixel(self, spp: int) -> None:
        self.samples_per_pixel = spp

    def get_samples_per_pixel(self, x: int, y:int) -> list[Sample]:
        """Utility to get all samples for a pixel as Sample(u,v) list."""
        self.start_pixel(x, y)
        out: list[Sample] = []
        for _ in range(self.samples_per_pixel):
            u, v = self.next_2d()
            out.append(Sample(u, v))
        return out

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
        # combine seed with pixel coords for reproducibility
        self._rng.seed((self._base_seed, x, y))

    def next_1d(self) -> float:
        return self._rng.random()

    def next_2d(self) -> Tuple[float, float]:
        return (self._rng.random(), self._rng.random())

    def clone(self, seed: Optional[int] = None) -> "RandomSampler":
        return RandomSampler(self.samples_per_pixel, seed if seed is not None else self._base_seed)

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

    def __repr__(self):
        return f"SamplingManager(sampler={self.sampler_name}, spp={self._spp}, settings={self.sample_settings})"