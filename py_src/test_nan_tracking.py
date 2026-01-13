import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.Scene import Scene
from src.Geometry import VObject, Sphere
from src.PrimaryStructures import TracingRay as Ray, Transform
from src.Raytracing import Raytracer, TracingStats
from src.Sampling import RandomSampler
from src.Shading import ShadingStrategy, AmbienceSettings, ShadowSettings, BackgroundSettings
from src.Luminance import Color

class NaNShading(ShadingStrategy):
    def __init__(self):
        super().__init__(AmbienceSettings(), ShadowSettings(), BackgroundSettings())
    
    def shade(self, *args, **kwargs) -> Color:
        return Color(np.nan, np.nan, np.nan)


def test_nan_tracking():
    scene = Scene()
    sph = Sphere()
    obj = VObject(transform=Transform.identity(), shape=sph)
    scene.add_object(obj)

    ray = Ray(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))

    from src.Shading import LambertShading, AmbienceSettings, ShadowSettings, BackgroundSettings
    from src.Intersections import BVHIntersection

    tracer = Raytracer(
        intersection_strategy=BVHIntersection(max_distance=100, max_steps=10),
        shading_strategy=LambertShading(AmbienceSettings(), ShadowSettings(), BackgroundSettings())
    )

    tracer.shader = NaNShading()
    tracer.stats = TracingStats()

    sampler = RandomSampler()

    c = tracer._trace_ray(scene, ray, recursions_left=1, sampler=sampler)

    # stats.nan_errors must have incremented
    assert tracer.stats.nan_errors > 0, f"Expected nan_errors > 0, got {tracer.stats.nan_errors}"

    # returned color should be finite after sanitization
    vals = np.array([c.r, c.g, c.b])
    assert np.all(np.isfinite(vals)), f"Returned color still contains non-finite values: {vals}"

    # Force logging threshold low and ensure the tracer updates last-logged counter
    tracer._nan_log_threshold = 1
    tracer._last_nan_logged = 0
    c2 = tracer._trace_ray(scene, ray, recursions_left=1, sampler=sampler)
    assert tracer._last_nan_logged >= tracer.stats.nan_errors - 0, "Expected last-nan-logged to be updated after threshold crossing"

if __name__ == '__main__':
    test_nan_tracking()
    print('NaN tracking test passed')
