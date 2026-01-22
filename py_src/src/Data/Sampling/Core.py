from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List, Union, Dict, Type, TypeVar


class PixelFilter(Enum):
    BOX = 0
    TENT = 1
    GAUSSIAN = 2

@dataclass
class SampleSettings:
    width: int = 800
    height: int = 600

    samples_per_pixel: int = 1
    
    # Adaptive Settings
    min_samples: int = 4            # Minimum rays before checking variance
    noise_threshold: float = 0.05   # Variance limit (lower = higher quality)

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
    """Calculates weights for a batch of samples using NumPy broadcasting."""
    # Mask out samples beyond filter radius
    mask = (np.abs(dist_x) <= settings.filter_width) & (np.abs(dist_y) <= settings.filter_width)
    
    weights = np.zeros_like(dist_x)
    ftype = settings.filter_type

    if ftype == PixelFilter.BOX:
        weights[mask] = 1.0

    elif ftype == PixelFilter.TENT:
        # 1.0 at center, 0.0 at radius
        wx = 1.0 - (np.abs(dist_x[mask]) / settings.filter_width)
        wy = 1.0 - (np.abs(dist_y[mask]) / settings.filter_width)
        weights[mask] = np.maximum(0.0, wx) * np.maximum(0.0, wy)

    elif ftype == PixelFilter.GAUSSIAN:
        alpha = 2.0
        fw_sq = settings.filter_width**2
        
        # Gaussian function: exp(-alpha * d^2) - exp(-alpha * r^2)
        shift = math.exp(-alpha * fw_sq)
        
        gx = np.exp(-alpha * dist_x[mask]**2) - shift
        gy = np.exp(-alpha * dist_y[mask]**2) - shift
        weights[mask] = np.maximum(0.0, gx * gy)

    return weights

def reconstruct_pixel(
    pixel_x: int,
    pixel_y: int,
    samples: List[Sample],
    colors: List[np.ndarray], 
    settings: SampleSettings
) -> np.ndarray:
    """
    Combines many samples into one final pixel color using the chosen filter.
    """
    if not samples:
        return np.array([0.0, 0.0, 0.0])

    u_coords = np.array([s.u for s in samples])
    v_coords = np.array([s.v for s in samples])
    sample_weights = np.array([s.w for s in samples])
    color_stack = np.stack(colors) 

    # Calculate Distances from Pixel Center (in Pixel Units)
    center_u = (pixel_x + 0.5) / settings.width
    center_v = (pixel_y + 0.5) / settings.height
    
    dist_x = (u_coords - center_u) * settings.width
    dist_y = (v_coords - center_v) * settings.height

    # Calculate Weights
    filter_weights = evaluate_filter_weight(settings, dist_x, dist_y)
    final_weights = filter_weights * sample_weights

    total_weight = np.sum(final_weights)
    
    if total_weight <= 0.0:
        return np.array([0.0, 0.0, 0.0])
        
    weighted_colors = color_stack * final_weights[:, np.newaxis]
    final_color = np.sum(weighted_colors, axis=0)
    
    return final_color / total_weight

class Sampler:
    """
    Base Sampler. Acts as the source of entropy for the entire engine.
    """
    def __init__(
        self, 
        input_source: Union[SampleSettings, np.random.Generator, None] = None, 
        seed: Optional[int] = None
    ):
        self.settings = SampleSettings()
        self._rng: np.random.Generator = None # type: ignore

        if isinstance(input_source, np.random.Generator):
            self._rng = input_source
        elif isinstance(input_source, SampleSettings):
            self.settings = input_source
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core Entropy Methods (The "Raw Data")
    # ------------------------------------------------------------------
    
    def next_1d(self) -> float:
        """Returns a random float in [0, 1)."""
        return self._rng.random()

    def next_2d(self) -> Tuple[float, float]:
        """Returns a tuple of two random floats in [0, 1)."""
        return (self._rng.random(), self._rng.random())

    # ------------------------------------------------------------------
    # 1. Pixel & Screen Space (Already Implemented)
    # ------------------------------------------------------------------

    def start_pixel(self, x: int, y: int) -> None:
        """Reset internal state for a new pixel."""
        pass

    def get_samples_per_pixel(self, x: int, y: int) -> List[Sample]:
        self.start_pixel(x, y)
        out = []
        for i in range(self.settings.samples_per_pixel):
            u, v = self.sample_pixel(x, y, i)
            out.append(Sample(u, v))
        return out

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> Tuple[float, float]:
        return self.next_2d()

    # ------------------------------------------------------------------
    # 2. Camera Effects
    # ------------------------------------------------------------------

    def sample_lens(self) -> Tuple[float, float]:
        """
        Returns (u, v) for Depth of Field calculation.
        These are usually mapped to a disk to find the ray origin on the lens.
        """
        # TEMP: Just return random numbers. 
        # FUTURE: A stratified sampler would advance its dimension index here.
        return self.next_2d()

    def sample_time(self) -> float:
        """
        Returns t for Motion Blur.
        Usually mapped to [shutter_open, shutter_close].
        """
        return self.next_1d()

    # ------------------------------------------------------------------
    # 3. Lighting & Materials (Next Event Estimation)
    # ------------------------------------------------------------------

    def sample_light_selection(self) -> float:
        """
        Returns w to decide WHICH light to sample in a scene with multiple lights.
        """
        return self.next_1d()

    def sample_light_position(self) -> Tuple[float, float]:
        """
        Returns (u, v) to pick a point ON the chosen light source (Area Light).
        """
        return self.next_2d()

    def sample_bsdf(self) -> Tuple[float, float]:
        """
        Returns (u, v) for the material scattering direction (GGX, Diffuse, etc).
        This is the u, v passed into your 'sample_microfacet_glass' function.
        """
        return self.next_2d()
    
    def sample_wavelength(self) -> float:
        """
        Returns lambda for Spectral Rendering (for later).
        """
        return self.next_1d()
    
    def sample_unit_sphere(self) -> np.ndarray:
        """
        Returns a random normalized vector on the surface of a unit sphere.
        Uses the standard Spherical Coordinate method.
        """
        # 1. Pick two random numbers
        u1 = np.random.random()
        u2 = np.random.random()

        # 2. Calculate spherical coordinates
        # z goes from -1 to 1
        z = 1.0 - 2.0 * u1 
        
        # r is the radius of the slice at height z
        r = math.sqrt(max(0.0, 1.0 - z * z))
        
        # phi is the angle around the Z axis
        phi = 2.0 * math.pi * u2

        # 3. Convert to Cartesian (x, y, z)
        x = r * math.cos(phi)
        y = r * math.sin(phi)

        return np.array([x, y, z], dtype=np.float32)

    def sample_cosine_hemisphere(self, normal: np.ndarray) -> np.ndarray:
        """
        Returns a random direction on the hemisphere oriented around 'normal'.
        The probability of choosing a direction is proportional to the cosine 
        of the angle with the normal (Cosine Importance Sampling).
        
        Crucial for efficient Diffuse/Lambertian rendering.
        """
        # 1. Generate a sample in Local Tangent Space (where Normal is 0,0,1)
        # We use Malley's Method (Concentric Disk Sampling -> Project to Hemisphere)
        # Or simpler Polar method:
        
        u1 = np.random.random()
        u2 = np.random.random()
        
        # r = sqrt(u1) ensures area-preserving mapping on the disk
        r = math.sqrt(u1)
        theta = 2.0 * math.pi * u2
        
        # Local coordinates (on the disk)
        local_x = r * math.cos(theta)
        local_y = r * math.sin(theta)
        
        # Project up to the hemisphere surface
        # z = sqrt(1 - x^2 - y^2) = sqrt(1 - r^2) = sqrt(1 - u1)
        local_z = math.sqrt(max(0.0, 1.0 - u1))

        local_vector = np.array([local_x, local_y, local_z], dtype=np.float32)

        # 2. Transform Local Space -> World Space (Align with actual Normal)
        return self._align_to_normal(local_vector, normal)

    def _align_to_normal(self, sample_dir: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        Helper: Builds an Orthonormal Basis (ONB) around the normal 
        and transforms the sample direction to world space.
        """
        # Ensure normal is normalized
        n = normal / np.linalg.norm(normal)
        
        # 1. Find a Tangent vector (T) not parallel to N
        # If N is close to world-up (0,1,0), use world-X, else use world-Y
        if abs(n[0]) > 0.9:
            ref_axis = np.array([0.0, 1.0, 0.0])
        else:
            ref_axis = np.array([1.0, 0.0, 0.0])
            
        # T = normalize(cross(N, ref))
        tangent = np.cross(n, ref_axis)
        tangent = tangent / np.linalg.norm(tangent)
        
        # 2. Find Bitangent (B)
        # B = cross(N, T)
        bitangent = np.cross(n, tangent)
        
        # 3. Transform: sample.x * T + sample.y * B + sample.z * N
        return (sample_dir[0] * tangent) + (sample_dir[1] * bitangent) + (sample_dir[2] * n)

    # ------------------------------------------------------------------
    # 4. Volumetrics
    # ------------------------------------------------------------------

    def sample_scattering_distance(self) -> float:
        """
        Returns w to decide how far a ray travels into a fog/medium before hitting a particle.
        Usually used in: dist = -log(1 - w) / sigma_t
        """
        return self.next_1d()

    def sample_phase_function(self) -> Tuple[float, float]:
        """
        Returns (u, v) to decide the new direction after a volumetric collision (Henyey-Greenstein).
        """
        return self.next_2d()

    # ------------------------------------------------------------------
    # 5. Optimization & Termination
    # ------------------------------------------------------------------

    def sample_roulette(self) -> float:
        """
        Returns w for Russian Roulette path termination.
        Compared against the throughput (e.g., if w > throughput, kill ray).
        """
        return self.next_1d()
        
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
        self.settings.samples_per_pixel = spp
        self._rebuild_grid()

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> Tuple[float, float]:
        # Map linear index to grid coordinates
        # e.g., index 5 in a 3x3 grid might be row 1, col 2
        n = self._grid_side
        
        col = sample_idx % n
        row = sample_idx // n
        
        # If we have more samples than grid slots, fallback to random inside the pixel
        if row >= n:
            u_local = self._rng.random()
            v_local = self._rng.random()
        else:
            # Jitter within the specific grid cell
            jitter_x = self._rng.random()
            jitter_y = self._rng.random()
            
            u_local = (col + jitter_x) / n
            v_local = (row + jitter_y) / n
        
        # Convert local pixel offset (0..1) to global normalized coordinates
        return (
            (x + u_local) / float(self.settings.width),
            (y + v_local) / float(self.settings.height)
        )

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
        # Create a unique index for this sample instance to decorrelate pixels
        # Adding 1 ensures we skip index 0
        idx = sample_idx + 1 + (x * 499) + (y * 503) 
        
        # Get Halton values in [0, 1]
        h_x = self._halton(idx, 2)
        h_y = self._halton(idx, 3)
        
        # Apply as offsets to the integer pixel coordinates
        return (
            (x + h_x) / float(self.settings.width),
            (y + h_y) / float(self.settings.height)
        )  

class AdaptiveSampler(Sampler):
    """
    Adaptive Sampler.
    
    Acts as a Random Sampler but provides a method `has_converged()`
    that the Integrator can call to stop sampling early if noise is low.
    """
    def __init__(self, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None):
        super().__init__(sample_settings, seed)
        
        # We use a simple random strategy for the coordinates themselves
        # to avoid alignment artifacts if we stop early.
        self._base_sampler = RandomSampler(sample_settings, seed)

    def sample_pixel(self, x: int, y: int, sample_idx: int) -> Tuple[float, float]:
        # Delegate coordinate generation to Random Sampler
        return self._base_sampler.sample_pixel(x, y, sample_idx)

    def has_converged(self, colors: List[np.ndarray]) -> bool:
        """
        Checks if the current batch of colors has sufficiently low variance.
        Call this inside your rendering loop.
        """
        n = len(colors)
        
        # 1. Don't stop before the minimum
        if n < self.settings.min_samples:
            return False
            
        # 2. Compute Variance (Simplified for RGB: Luminance Variance)
        # We check the standard error of the mean: sigma / sqrt(N)
        # If the error interval is smaller than threshold, we are good.
        
        # Convert list to array for calculation (can be optimized to online algorithm later)
        color_stack = np.stack(colors) # Shape (N, 3)
        
        # Calculate Luminance: R*0.2126 + G*0.7152 + B*0.0722
        luminance = np.dot(color_stack, np.array([0.2126, 0.7152, 0.0722]))
        
        mean_lum = np.mean(luminance)
        if mean_lum == 0.0:
            return True # Pitch black converges instantly

        # Sample Variance
        variance = np.var(luminance, ddof=1)
        
        # Heuristic: We want the noise relative to the brightness to be low.
        # Metric: RMSE / Mean < Threshold
        rmse = math.sqrt(variance / n) 
        
        # Avoid division by zero for very dark pixels
        relative_error = rmse / (mean_lum + 1e-4)

        return relative_error < self.settings.noise_threshold

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


T = TypeVar("T", bound=Sampler)

_SAMPLER_REGISTRY: Dict[str, Type[Sampler]] = {
    "random": RandomSampler,
    "stratified": StratifiedSampler,
    "halton": HaltonSampler,
    "adaptive": AdaptiveSampler,
}

def register_sampler(name: str):
    def _decorator(cls: Type[T]) -> Type[T]:
        _SAMPLER_REGISTRY[name] = cls
        return cls
    return _decorator

def create_sampler(name: str, sample_settings: SampleSettings = SampleSettings(), seed: Optional[int] = None) -> Sampler:
    """
    Instantiate a registered sampler by name.
    Use this to avoid hard imports at call sites.
    
    :param name: Description
    :type name: str
    :param sample_settings: Description
    :type sample_settings: SampleSettings
    :param seed: Description
    :type seed: Optional[int]
    :return: Description
    :rtype: Sampler
    """
    if name not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler '{name}'. Registered: {list(_SAMPLER_REGISTRY.keys())}"
        )

    cls = _SAMPLER_REGISTRY[name]

    return cls(sample_settings, seed)

def list_samplers() -> Dict[str, Type[Sampler]]:
    return dict(_SAMPLER_REGISTRY)