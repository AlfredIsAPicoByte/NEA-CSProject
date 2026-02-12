import sys
import os
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
py_src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Ensure py_src is on sys.path so local packages (tests, src) import correctly
sys.path.insert(0, py_src_root)

from tests import test_scenes
# Correct imports (module names are case-sensitive)
from src.Rendering.RayTracing.Core import RayTracer, TracingStats, RayTracingSettings
from src.Rendering.RayTracing.Intersections import BVHIntersection
from src.Rendering.RayTracing.Shading import LambertShading, AmbienceSettings, ShadowSettings, BackgroundSettings, PhysicalShadingSettings
from src.Image.Film import Film
from src.Data.Sampling.Core import SamplingManager, SampleSettings

SCENE_FUNCS = [
    name for name in dir(test_scenes) if name.startswith('get_')
]

print('Found scenes:', SCENE_FUNCS)

from src.Rendering.RayTracing.Intersections import IntersectionSettings

intersection = BVHIntersection(IntersectionSettings(max_distance=500, max_steps=128))
# Build shading settings correctly
shading_settings = PhysicalShadingSettings(AmbienceSettings(False), ShadowSettings(False), BackgroundSettings(False))
shading = LambertShading(shading_settings)

sampling_manager = SamplingManager(SampleSettings(samples_per_pixel=1), "random")

failures = []

for name in SCENE_FUNCS:
    func = getattr(test_scenes, name)
    try:
        print('\n--- Testing', name)
        scene = func(32, 16)

        # Construct a per-scene RayTracer using current camera resolution
        rt_settings = RayTracingSettings(
            image_width=scene.camera.width,
            image_height=scene.camera.height,
            sampling_manager=sampling_manager,
            max_recursions=0,
            intersection_strategy=intersection,
            shading_strategy=shading,
            use_tiling=True,
            tile_size=8,
            debug_mode=False,
            verbose_logging=False
        )
        raytracer = RayTracer(rt_settings)

        # Generate the film
        raytracer.generate_film(scene)
        film = raytracer.settings.film
        img = film.get_image()

        out_dir = os.path.join(project_root, 'images', 'testing', 'scene_tests')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{name}.png')
        Film.save(img, out_path)
        print('Rendered', name, '->', out_path)
    except Exception:
        tb = traceback.format_exc()
        print('ERROR rendering', name)
        print(tb)
        failures.append((name, tb))

print('\nDone. Failures:', len(failures))
for n, tb in failures:
    print('\n--', n)
    print(tb)

if failures:
    sys.exit(1)
else:
    sys.exit(0)
