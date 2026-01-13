import numpy as np
import sys, os
# Ensure local 'src' folder is on path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from PrimaryStructures import Transform
from Geometry import Primitive, Sphere
from Scene import Scene
from PrimaryStructures import Ray


def test_world_transform_applied_to_normals():
    # Parent with non-uniform scale to introduce a difference between local and world transform
    parent_transform = Transform(position=np.zeros(3), rotation=np.zeros(3), scale=np.array([2.0, 1.0, 1.0]))
    parent = Primitive(transform=parent_transform, name="parent")

    # Child sphere at origin (local)
    child_transform = Transform.identity()
    sphere = Sphere()
    child = Primitive(transform=child_transform, shape=sphere, name="child")

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
    world_transform = child.world_transform
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
    test_world_transform_applied_to_normals()
    print('Test passed')
