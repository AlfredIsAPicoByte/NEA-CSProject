import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

sys.path.insert(0, current_dir)

from src.Data.Transform import Transform
from src.Data.Ray import Ray
from py_src.src.Geometry.SDF import Sphere
from py_src.src.Geometry.Node import SceneNode
from src.Data.Scene import Scene


def test_world_transform_applied_to_normals():
    # Parent with non-uniform scale to introduce a difference between local and world transform
    parent_transform = Transform(position=np.zeros(3), rotation=np.zeros(3), scale=np.array([2.0, 1.0, 1.0]))
    parent = SceneNode(transform=parent_transform, name="parent")

    # Child sphere at origin (local)
    child_transform = Transform.identity()
    sphere = Sphere()
    child = SceneNode(transform=child_transform, shape=sphere, name="child")

    parent.add_child(child)

    scene = Scene()
    scene.add_object(parent)

    # Cast a ray directly along +Z toward the origin from z = -5
    ray = Ray(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))

    hit = scene.get_closest_intersection(ray)

    assert hit.hit, "Expected to hit the sphere"
    assert hit.point is not None
    assert hit.normal is not None

    # Recompute expected normal using world_transform pipeline
    world_transform = child.transform
    local_pt = world_transform.inverse_transform_point(hit.point)
    local_normal = sphere.get_normal(local_pt)
    expected_world_normal = world_transform.transform_normal(local_normal)

    # Normalize both
    def unit(v):
        n = np.linalg.norm(v)
        return v / (n if n != 0 else 1.0)

    got = unit(hit.normal)
    expected = unit(expected_world_normal)

    assert np.allclose(got, expected, atol=1e-6), f"Normal mismatch: got={got}, expected={expected}"

if __name__ == '__main__':
    sys.exit(pytest.main(["-v", __file__]))
