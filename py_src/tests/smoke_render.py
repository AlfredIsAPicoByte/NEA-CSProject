import sys
import os

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(repo, 'py_src'))

from tests.test_scenes import get_minimal_scene
from src.Rendering.Raytracing import RayTracer, TracingStats
from src.Rendering.Intersections import BVHIntersection
from src.Rendering.Interactions import TerminalInteraction
from src.Rendering.RayTracing.Shading import LambertShading, AmbienceSettings, ShadowSettings, BackgroundSettings
from src.Image.Film import Film
from src.Data.Sampling.Core import SamplingManager, SampleSettings

def main():
    width, height = 64, 32
    scene = get_minimal_scene(width, height)

    intersection = BVHIntersection(max_distance=500, max_steps=128)
    interactor = TerminalInteraction()
    shading = LambertShading(
        ambience_settings=AmbienceSettings(False),
        shadow_settings=ShadowSettings(False),
        background_settings=BackgroundSettings(True, None, None)
    )

    sampling_manager = SamplingManager(SampleSettings(samples_per_pixel=1), "random")

    raytracer = RayTracer(
        max_recursions=1,
        sampling_manager=sampling_manager,
        intersection_strategy=intersection,
        interaction_strategy=interactor,
        shading_strategy=shading,
    )

    film = raytracer.render(scene, sampler=None, tile_size=8)

    img = film.get_image()
    out = os.path.join(repo, 'images', 'testing', 'smoke_test.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    Film.save(img, out)
    print('Smoke render saved to', out)


if __name__ == '__main__':
    main()
