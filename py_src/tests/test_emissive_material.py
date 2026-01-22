import numpy as np
from src.Data.Transform import Transform
from src.Geometry.Primitive import Primitive
from src.Geometry.Core import Sphere
from src.Material.Factory import MaterialFactory
from src.Rendering.RayTracing.Core import RayTracer
from src.Rendering.RayTracing.Intersections import AnalyticalIntersection
from src.Rendering.RayTracing.Interactions import StandardInteraction
from src.Rendering.RayTracing.Shading import LambertShading
from src.Data.Ray import TracingRay
from src.Data.Scene import Scene
from src.Data.Sampling.Core import RandomSampler, SampleSettings
from src.Data.Color import Color


def test_emissive_render_direct():
    # Single emissive sphere at origin
    sphere = Primitive(name='E', transform=Transform(position=np.array([0.0, 0.0, 0.0])), shape=Sphere(), material=MaterialFactory.create_emissive(Color.from_hex('#FFAA22'), 1.0))

    scene = Scene(camera=None)
    scene.add_object(sphere)

    tracer = RayTracer(max_recursions=1, intersection_strategy=AnalyticalIntersection(), interaction_strategy=StandardInteraction(), shading_strategy=LambertShading())

    # Ray from z=-5 to +Z
    ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
    sampler = RandomSampler(SampleSettings())
    color = tracer._trace_ray(scene, ray, tracer.max_recursions, sampler)

    assert color is not None
    # Should match emissive color at least by non-zero green/blue components
    assert color.r > 0.9
    assert color.g > 0.0
    assert color.b > 0.0

