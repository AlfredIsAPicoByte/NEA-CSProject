import sys
import os
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

sys.path.insert(0, current_dir)

from test_scenes import get_minimal_scene
from src.Rendering.RayTracing.Core import RayTracer
from src.Rendering.RayTracing.Intersections import AnalyticalIntersection
from src.Rendering.RayTracing.Shading import RecursiveLambertShading
from src.Rendering.RayTracing.Interactions import StandardInteraction
from src.Data.Sampling.Core import SamplingManager, SampleSettings, PixelFilter
from src.Data.Camera import Camera

def test_minimal_scene_not_all_background():
    scene = get_minimal_scene(16, 9)
    sample_settings = SampleSettings(16, 9, 1, PixelFilter.BOX, 2)
    sampling_manager = SamplingManager(sample_settings, 'halton')

    raytracer = RayTracer(
        max_depth=4,
        sampling_manager=sampling_manager,
        camera=Camera(),
        intersection_strategy=AnalyticalIntersection(max_distance=100),
        interaction_strategy=StandardInteraction(),
        shading_strategy=RecursiveLambertShading(),
        custom_background=scene.get_background_color([0.0, 0.0, 0.0, -1.0]),
        enable_scene_background=True
    )

    pixels = raytracer.render(scene)
    bg = scene.get_background_color([0.0, 0.0, -1.0])

    # Assert at least one pixel differs from background
    assert any((float(p.r), float(p.g), float(p.b)) != (float(bg.r), float(bg.g), float(bg.b)) for p in pixels), "All pixels match the background; objects were not rendered."

if __name__ == '__main__':
    sys.exit(pytest.main(["-v", __file__]))
