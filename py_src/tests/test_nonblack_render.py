import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from test_scenes import get_minimal_scene
from src.Raytracing import Raytracer, JitterRayGenerator, RayMarchingIntersection, StandardInteraction, RecursiveLambertShading
from src.Sampling import SamplingManager, SampleSettings, PixelFilter


def test_minimal_scene_not_all_background():
    scene = get_minimal_scene(16, 9)
    sample_settings = SampleSettings(16, 9, 1, PixelFilter.BOX, 2)
    sampling_manager = SamplingManager(sample_settings, 'halton')

    raytracer = Raytracer(
        max_depth=4,
        sampling_manager=sampling_manager,
        ray_generator=JitterRayGenerator(),
        intersection_strategy=RayMarchingIntersection(max_distance=100),
        interaction_strategy=StandardInteraction(),
        shading_strategy=RecursiveLambertShading(),
        custom_background=scene.get_background_color([0.0, 0.0, 0.0, -1.0]),
        enable_scene_background=True
    )

    pixels = raytracer.render(scene)
    bg = scene.get_background_color([0.0, 0.0, -1.0])

    # Assert at least one pixel differs from background
    assert any((float(p.r), float(p.g), float(p.b)) != (float(bg.r), float(bg.g), float(bg.b)) for p in pixels), "All pixels match the background; objects were not rendered."