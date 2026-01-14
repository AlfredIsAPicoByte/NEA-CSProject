import sys, os
import numpy as np

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(repo, 'py_src'))

from src.Geometry.Primitive import Primitive
from src.Geometry.Core import Sphere
from src.Data.Transform import Transform
from src.Utilities.Scene import Scene
from src.Data.Ray import Ray

# Sphere at origin
sphere = Primitive(name="S", transform=Transform(position=np.array([0.0,0.0,0.0])), shape=Sphere())
scene = Scene(camera=None)
scene.add_object(sphere)

sphere.update_world_matrices()

# Fire rays to +X, +Y, +Z from far away
rays = [
    Ray(origin=np.array([-5.0,0.0,0.0]), orientation=np.array([1.0,0.0,0.0])), # hits -x side
    Ray(origin=np.array([0.0,-5.0,0.0]), orientation=np.array([0.0,1.0,0.0])), # hits -y side
    Ray(origin=np.array([0.0,0.0,-5.0]), orientation=np.array([0.0,0.0,1.0])), # hits -z side
]

for i, r in enumerate(rays):
    hit = scene.get_closest_intersection(r)
    print(f"Ray {i}: hit={hit.hit}")
    if hit.hit:
        print(' point=', hit.point)
        print(' normal=', hit.normal)
        print(' normal mapped color=', ((hit.normal*0.5)+0.5))
