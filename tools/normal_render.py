import sys, os
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(repo, 'py_src'))

from tests.test_scenes import get_minimal_scene
from src.Rendering.Raytracing import Raytracer
from src.Rendering.Intersections import BVHIntersection
from src.Rendering.Interactions import TerminalInteraction
from src.Rendering.Shading import NormalShading
from src.Utilities.Sampling import SampleSettings, RandomSampler
from src.Image.Film import Film
import numpy as np

scene = get_minimal_scene(64, 64)
intersection = BVHIntersection()
interactor = TerminalInteraction()
shading = NormalShading()

raytracer = Raytracer(max_recursions=1, intersection_strategy=intersection, interaction_strategy=interactor, shading_strategy=shading)
film = raytracer.render(scene, sampler=RandomSampler(SampleSettings(samples_per_pixel=1)), tile_size=16)
img = film.get_image()
print('Image shape', img.shape)
# Save
out = os.path.join(repo, 'images', 'testing', 'normal_test.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
Film.save(img, out)
print('Saved to', out)
