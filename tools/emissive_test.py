import sys, os
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(repo, 'py_src'))

import numpy as np
from tests.test_scenes import get_emissive_scene
from src.Rendering.Raytracing import RayTracer, TracingStats
from src.Rendering.Intersections import BVHIntersection
from src.Rendering.Interactions import TerminalInteraction
from src.Rendering.Shading import LambertShading, AmbienceSettings, ShadowSettings, BackgroundSettings
from src.Utilities.Sampling import SamplingManager, SampleSettings

scene = get_emissive_scene(32, 16)
# Print emissive materials
for obj in scene.objects:
    mat = getattr(obj, 'material', None)
    if mat and getattr(mat, 'type', None) and mat.type.name == 'EMISSIVE':
        print('Found emissive', obj.name, mat.data.emission_color, mat.data.emission_intensity)
        print('get_emissive_component ->', mat.get_emissive_component())

intersection = BVHIntersection()
interactor = TerminalInteraction()
shading = LambertShading(ambience_settings=AmbienceSettings(False), shadow_settings=ShadowSettings(False))

raytracer = RayTracer(max_recursions=1, intersection_strategy=intersection, interaction_strategy=interactor, shading_strategy=shading)

# Directly test shading on center ray
from src.Utilities.Sampling import RandomSampler, SampleSettings
sampler = RandomSampler(SampleSettings())
# Ray from camera center to scene center (deterministic ray)
cam = scene.camera
cx, cy = cam.width//2, cam.height//2
u = (cx + 0.5) / float(cam.width)
v = (cy + 0.5) / float(cam.height)
_r = cam.generate_ray(u, v)
from src.Data.Ray import TracingRay
ray = TracingRay(origin=_r.origin, orientation=_r.orientation, pixel_x=cx, pixel_y=cy, sample_u=0.5, sample_v=0.5)
print('Testing trace for center ray (u,v)=', (u,v), '->', ray)
hit = scene.get_closest_intersection(ray)
print('Hit info:', hit)
print('Hit obj material:', getattr(hit.obj, 'material', None))
color = raytracer._trace_ray(scene, ray, raytracer.max_recursions, sampler)
print('Trace output color:', color)

film = raytracer.render(scene, sampler=sampler, tile_size=8)
img = film.get_image()
print('Image shape', img.shape)
# Print central pixel
cx, cy = scene.camera.width//2, scene.camera.height//2
print('Center pixel', img[cy, cx])
print('Raw accum_color', film.accum_color[cy, cx])
print('Raw accum_weight', film.accum_weight[cy, cx])
# Save
from src.Image.Film import Film
out = os.path.join(repo, 'images','testing','emissive_test.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
Film.save(img, out)
print('Saved to', out)
