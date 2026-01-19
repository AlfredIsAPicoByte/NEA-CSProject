import numpy as np
from src.Data.Transform import Transform
from src.Geometry.Primitive import Primitive
from src.Geometry.Primitive import Primitive
from src.Geometry.Primitive import Primitive
from src.Geometry.Primitive import Primitive
from src.Geometry.Primitive import Primitive
from src.Geometry.Primitive import Primitive
from src.Geometry.Primitive import Primitive
from src.Geometry.Core import Sphere
from src.Rendering.Intersections import BVHIntersection


def test_bvh_respects_world_transforms():
    # Create two spheres separated along X axis
    p1 = Primitive(name="A", transform=Transform(position=np.array([0.0, 0.0, 0.0])), shape=Sphere())
    p2 = Primitive(name="B", transform=Transform(position=np.array([5.0, 0.0, 0.0])), shape=Sphere())

    p1.update_matrices()
    p2.update_matrices()

    bvh = BVHIntersection()
    root = bvh._build_bvh([p1, p2])

    # Root box should span from near 0 to near 5 (with padding)
    min_x = root.box.min_point[0]
    max_x = root.box.max_point[0]

    assert min_x < 1.0, f"Expected min_x < 1, got {min_x}"
    assert max_x > 4.0, f"Expected max_x > 4, got {max_x}"
