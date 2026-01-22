from __future__ import annotations
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.Data.Scene import Scene
from src.Data.Sampling.Core import Sampler, RandomSampler
from src.Image.Film import Film
from src.Utilities.Memory.Core import get_process_id, get_memory_mb

@dataclass
class RenderStats:
    """
    Statistics collected during rendering for performance analysis and debugging.
    """
    memory_usage: float = 0.0  # in MB
    pixels_processed: int = 0
    nan_errors: int = 0
    
    # --- Timing ---
    time_taken_seconds: float = 0.0
    _start_time: float = field(default=0.0, repr=False)

    def start_timer(self):
        self._start_time = time.perf_counter()

    def stop_timer(self):
        self.time_taken_seconds = time.perf_counter() - self._start_time
        
    def update_memory(self):
        self.memory_usage = get_memory_mb(get_process_id())

    def __add__(self, other: "RenderStats") -> "RenderStats":
        new_stats = RenderStats()

        # Max/Avg specific fields
        new_stats.time_taken_seconds = max(self.time_taken_seconds, other.time_taken_seconds)
        new_stats.memory_usage = max(self.memory_usage, other.memory_usage)

        return new_stats

    def format_report(self) -> str:
        """
        Generates a formatted string report suitable for saving to a .txt file.
        """
        lines = []
        lines.append(f"=== Rendering Stats ===")
        lines.append(f"Time: {self.time_taken_seconds:.3f}s")
        lines.append(f"Mem: {self.memory_usage:.2f}MB")
        lines.append(f"-------------------------")
        lines.append(f"Diagnostics:")
        lines.append(f"  - NaN Errors:      {self.nan_errors}")
        
        return "\n".join(lines)
    
    def print_verbose_report(self):
        print()
        print(self.format_report())

@dataclass(slots=True)
class AlgorithmSettings:
    image_width: int
    image_height: int

    film: Film = field(default_factory=lambda: Film(0, 0))

class Algorithm(ABC):
    """
    Abstract base for rendering algorithms (ray marcher, path tracer, rasterizer, ...).
    Implementations should be side-effect free where possible and avoid global state.
    """
    settings_type = AlgorithmSettings
    def __init__(self, settings: AlgorithmSettings):
        self.settings = settings
        self.stats = RenderStats()

    def setup(self, scene: Scene) -> None:
        """
        Genric setup called once per-scene before rendering begins.
        """
        pass

    @abstractmethod
    def render_tile(
        self,
        scene: Scene,
        sampler: Sampler,
        tile_x: int,
        tile_y: int,
        width: int,
        height: int,
    ) -> None:
        """
        Resolves a single tile of the image.
        
        :param scene: The scene to render
        :type scene: Scene
        :param sampler: The sampler to use for pixel sampling
        :type sampler: Sampler
        :param tile_x: The x-coordinate of the tile's top-left corner
        :type tile_x: int
        :param tile_y: The y-coordinate of the tile's top-left corner
        :type tile_y: int
        :param width: The width of the tile
        :type width: int
        :param height: The height of the tile
        :type height: int
        """
        ...
    
    def generate_film(
            self,
            scene: Scene,
            sampler: Optional[Sampler] = None,
            region: Optional[Tuple[int, int, int, int]] = None,
        ) -> None:
        """
        Generates a film for the given scene using the specified sampler and region.

        :param scene: The scene to render
        :type scene: Scene
        :param sampler: The sampler to use for pixel sampling
        :type sampler: Optional[Sampler]
        :param region: The region to render (x, y, width, height)
        :type region: Optional[Tuple[int, int, int, int]]
        """
        self.reset_stats()
        self.setup(scene)

        sampler = sampler or RandomSampler()
        region = region or (0, 0, self.settings.image_width, self.settings.image_height)

        self.render_tile(scene, sampler, *region)

        self.settings.film = Film(0, 0)

    def reset_stats(self) -> None:
        """Resets the rendering statistics."""
        self.stats = RenderStats()

class RenderManager:
    """
    Manages the rendering process using a specified algorithm.
    """
    pass