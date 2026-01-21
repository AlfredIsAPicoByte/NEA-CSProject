from __future__ import annotations
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.Data.Scene import Scene
from src.Data.Sampling import Sampler, RandomSampler
from src.Image.Film import Film
from src.Utilities.Memory.Core import get_process_id, get_memory_mb

@dataclass
class RenderStats:
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

    raw_film: Film = field(default_factory=lambda: Film(0, 0))
    final_film: Film = field(default_factory=lambda: Film(0, 0))

class Algorithm(ABC):
    """
    Abstract base for rendering algorithms (ray marcher, path tracer, rasterizer, ...).
    Implementations should be side-effect free where possible and avoid global state.
    """
    def __init__(self, settings: AlgorithmSettings):
        self.settings = settings
        self.stats = RenderStats()

    def setup(self, scene: Scene) -> None:
        """
        Docstring for setup
        
        :param self: Description
        :param scene: Description
        :type scene: Scene
        """
        pass

    @abstractmethod
    def render_tile(
        self,
        scene: Scene,
        sampler: Sampler,
        x: int,
        y: int,
        h: int,
        h: int,
    ) -> None:
        """
        Docstring for render_tile
        
        :param scene: Description
        :type scene: Scene
        :param sampler: Description
        :type sampler: Sampler
        :param x: Description
        :type x: int
        :param y: Description
        :type y: int
        :param h: Description
        :type h: int
        :param h: Description
        :type h: int
        """
        ...
    
    def generate_film(
            self,
            scene: Scene,
            sampler: Optional[Sampler] = None,
            region: Optional[Tuple[int, int, int, int]] = None,
        ) -> Film:
        """
        Docstring for generate_film
        
        :param scene: Description
        :type scene: Scene
        :param sampler: Description
        :type sampler: Optional[Sampler]
        :param region: Description
        :type region: Optional[Tuple[int, int, int, int]]
        :return: Description
        :rtype: Film
        """
        self.reset_stats()
        self.setup(scene)

        sampler = sampler or RandomSampler()
        region = region or (0, 0, self.settings.image_width, self.settings.image_height)

        self.render_tile(scene, sampler, *region)

        return Film(0, 0)

    def reset_stats(self) -> None:
        self.stats = RenderStats()

class RenderManager:
    pass