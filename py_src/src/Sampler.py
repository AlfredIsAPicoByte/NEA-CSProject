from py_src.src import Luminance, PrimaryStructures
from src import Algorithims
from enum import Enum
import numpy as np

"""

"""

class SamplerType(Enum):
    MONTE_CARLO = 0
    STRATIFIED = 1
    QUASI_MONTE_CARLO = 2
    ADAPTIVE = 3 # Not implemented yet

class SampleSettings:
    position: int = 1
    size: int = 16
    sampler_type: SamplerType = SamplerType.STRATIFIED

    def __init__(self):
        pass

class RenderMode(Enum):
    RAYTRACING = 0
    RASTERIZATION = 1

class RenderSettings:
    image_scale: float = 1.0
    background_color: Luminance.Color = Luminance.Color(0, 0, 0)
    ambient_light: Luminance.Color = Luminance.Color(0.1, 0.1, 0.1)
    render_mode: RenderMode = RenderMode.RAYTRACING

    def __init__(self, width: int, height: int, raysPerPixel: int):
        self.width = width
        self.height = height
        self.rays_per_pixel = raysPerPixel

class PixelData: 
    def __init__(self, color: Luminance.Color):
        self.color = color

class Sample:
    def __init__(self, u: float, v: float, rays: list[PrimaryStructures.Ray] = []):
        self.u = u
        self.v = v

        self.rays = rays
        self.pixle_data: list[PixelData] = []

class SamplingManager:
    def __init__(self, settings: SampleSettings, renderSettings: RenderSettings):
        self.sample_settings = settings
        self.render_settings = renderSettings
        
        self.samples: list[Sample] = []
        self.GenerateSamples()
    
    def GenerateSamples(self):
        if self.settings.sampler_type == SamplerType.MONTE_CARLO:
            for _ in range(self.settings.size):
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
                self.samples.append((x, y))
        elif self.settings.sampler_type == SamplerType.STRATIFIED:
            n = int(np.sqrt(self.settings.size))
            for i in range(n):
                for j in range(n):
                    x = (i + np.random.uniform(0, 1)) / n
                    y = (j + np.random.uniform(0, 1)) / n
                    self.samples.append((x, y))
        elif self.settings.sampler_type == SamplerType.QUASI_MONTE_CARLO:
            # Using Halton sequence for quasi-random sampling
            def halton(index, base):
                result = 0
                f = 1 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i //= base
                    f /= base
                return result
            
            for i in range(self.settings.size):
                x = halton(i + 1, 2)
                y = halton(i + 1, 3)
                self.samples.append((x, y))
        elif self.settings.sampler_type == SamplerType.ADAPTIVE:
            # Placeholder for adaptive sampling logic
            # This would typically involve more complex logic based on scene analysis
            print("Adaptive sampling not implemented, defaulting to Monte Carlo")
            for _ in range(self.settings.size):
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
                self.samples.append((x, y))
    
    def GetSamples(self, range: int = 1, offset: int = 0) -> list[tuple[float, float]]:
        return self.samples[offset:offset + range]
    
    def __repr__(self):
        return f"Sampler(settings={self.sample_settings}, render_settings={self.render_settings}, samples={self.samples})"