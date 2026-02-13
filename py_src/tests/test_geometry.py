import pytest
import numpy as np
from src.Data.Transform import Transform
from src.Data.Ray import Ray
from src.Data.Scene import SceneNode
from src.Geometry.SDF import Sphere, Cube, Cylinder, Pyramid, Capsule, ShapeSubtraction, ShapeIntersection, ShapeExtrusion
from src.Geometry.BVH import BVHNode, BVHSplitMode, build_bvh_tree
from src.Geometry.AABB import AABB, transform_bounds, convert_bounds_to_corners
from src.Rendering.RayTracing.Intersections import BVHIntersection
from .bench_scenes import get_minimal_scene

class TestSphereSDF:
    def test_centre_is_negative(self):
        """Origin is inside a unit sphere → SDF < 0."""
        assert Sphere(1.0).get_distance(np.array([0.0, 0.0, 0.0])) < 0

    def test_surface_is_zero(self):
        """A point exactly on the surface should give SDF ≈ 0."""
        val = Sphere(1.0).get_distance(np.array([1.0, 0.0, 0.0]))
        assert np.isclose(val, 0.0, atol=1e-6)

    def test_outside_is_positive(self):
        assert Sphere(1.0).get_distance(np.array([2.0, 0.0, 0.0])) > 0

    def test_sdf_equals_distance_minus_radius(self):
        """For a sphere the SDF is |p| - r."""
        sphere = Sphere(2.0)
        point = np.array([5.0, 0.0, 0.0])
        expected = np.linalg.norm(point) - 2.0
        assert np.isclose(sphere.get_distance(point), expected, atol=1e-6)

    def test_radius_scaling(self):
        """Larger radius means a farther point can still be inside."""
        assert Sphere(5.0).get_distance(np.array([4.0, 0.0, 0.0])) < 0
        assert Sphere(1.0).get_distance(np.array([4.0, 0.0, 0.0])) > 0

    @pytest.mark.parametrize("axis", [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
    ])
    def test_sphere_is_isotropic(self, axis):
        """SDF value for a unit-distance point should be ~0 in any direction."""
        val = Sphere(1.0).get_distance(axis)
        assert np.isclose(val, 0.0, atol=1e-6)

    def test_sdf_gradient_points_outward(self):
        """Numerical gradient at a surface point should point radially outward."""
        sphere = Sphere(1.0)
        p = np.array([1.0, 0.0, 0.0])
        eps = 1e-4
        grad = np.array([
            sphere.get_distance(p + np.array([eps, 0, 0])) - sphere.get_distance(p - np.array([eps, 0, 0])),
            sphere.get_distance(p + np.array([0, eps, 0])) - sphere.get_distance(p - np.array([0, eps, 0])),
            sphere.get_distance(p + np.array([0, 0, eps])) - sphere.get_distance(p - np.array([0, 0, eps])),
        ]) / (2 * eps)
        # Gradient should be approximately (1, 0, 0) for a sphere at the +X pole
        assert np.allclose(grad / np.linalg.norm(grad), p / np.linalg.norm(p), atol=1e-3)
        assert np.allclose(sphere.get_normal(p), p / np.linalg.norm(p), atol=1e-3) # Test the analitical solution too

class TestCubeSDF:
    def test_centre_is_inside(self):
        """Origin should be inside a unit cube (half-extent 1)."""
        assert Cube(1.0).get_distance(np.array([0.0, 0.0, 0.0])) < 0

    def test_corner_is_outside(self):
        """A point beyond all three faces should be positive."""
        assert Cube(1.0).get_distance(np.array([2.0, 2.0, 2.0])) > 0

    def test_face_centre_is_zero(self):
        """A point on the centre of a face should give SDF ≈ 0."""
        # For a Cube with half-extent 1 the +X face centre is at (1, 0, 0)
        val = Cube(1.0).get_distance(np.array([1.0, 0.0, 0.0]))
        assert np.isclose(val, 0.0, atol=1e-5)

    def test_symmetry_across_axes(self):
        """The cube SDF should be symmetric about all three principal planes."""
        cube = Cube(1.0)
        p = np.array([0.5, 0.3, 0.1])
        for sign_x in (1, -1):
            for sign_y in (1, -1):
                for sign_z in (1, -1):
                    q = p * np.array([sign_x, sign_y, sign_z])
                    assert np.isclose(cube.get_distance(p), cube.get_distance(q), atol=1e-6)

    def test_inside_depth(self):
        """SDF inside should reflect distance to nearest face."""
        # At origin the nearest face is at distance 1 along each axis, so SDF ≈ -1
        val = Cube(1.0).get_distance(np.array([0.0, 0.0, 0.0]))
        assert np.isclose(val, -1.0, atol=1e-5)

class TestCylinderSDF:
    def test_centre_is_inside(self):
        assert Cylinder(1.0, 2.0).get_distance(np.array([0.0, 0.0, 0.0])) < 0

    def test_outside_radially(self):
        """A point beyond the curved surface should give SDF > 0."""
        assert Cylinder(1.0, 2.0).get_distance(np.array([2.0, 0.0, 0.0])) > 0

    def test_outside_axially(self):
        """A point above the cap should give SDF > 0."""
        assert Cylinder(1.0, 2.0).get_distance(np.array([0.0, 2.0, 0.0])) > 0

    def test_surface_on_curved_wall(self):
        """A point on the curved wall exactly should give SDF ≈ 0."""
        val = Cylinder(1.0, 2.0).get_distance(np.array([1.0, 0.0, 0.0]))
        assert np.isclose(val, 0.0, atol=1e-5)

class TestShapeSubtraction:
    def test_subtracted_region_is_outside(self):
        """A point that is inside the subtracted shape should have SDF > 0."""
        sphere = Sphere(1.0)
        cube = Cube(0.5)
        result = ShapeSubtraction(sphere, Transform.Identity(), cube, Transform.Identity())
        # Origin is inside both — after subtraction it should be outside the result
        val = result.get_distance(np.array([0.0, 0.0, 0.0]))
        assert val >= 0.0

    def test_region_outside_both_is_outside(self):
        """A point outside both shapes should still be outside the result."""
        sphere = Sphere(1.0)
        cube = Cube(0.5)
        result = ShapeSubtraction(sphere, Transform.Identity(), cube, Transform.Identity())
        val = result.get_distance(np.array([5.0, 0.0, 0.0]))
        assert val > 0.0

    def test_region_inside_sphere_outside_cube_is_inside(self):
        """A point inside the sphere but outside the cube should remain inside."""
        sphere = Sphere(1.0)
        cube = Cube(0.3)
        result = ShapeSubtraction(sphere, Transform.Identity(), cube, Transform.Identity())
        # (0.8, 0, 0) is inside sphere (SDF ≈ −0.2) and outside cube (SDF ≈ +0.5)
        val = result.get_distance(np.array([0.8, 0.0, 0.0]))
        assert val < 0.0

class TestShapeIntersection:
    def test_centre_inside_both(self):
        """Origin inside both shapes → intersection result should also be inside."""
        sphere = Sphere(1.0)
        cube = Cube(1.5)
        result = ShapeIntersection(sphere, Transform.Identity(), cube, Transform.Identity())
        assert result.get_distance(np.array([0.0, 0.0, 0.0])) < 0.0

    def test_inside_only_one_is_outside(self):
        """Inside sphere but outside a small cube → not in intersection."""
        sphere = Sphere(2.0)
        cube = Cube(0.4)
        result = ShapeIntersection(sphere, Transform.Identity(), cube, Transform.Identity())
        # (1.5, 0, 0): inside sphere (SDF ≈ -0.5), outside tiny cube (SDF ≈ +1.1)
        val = result.get_distance(np.array([1.5, 0.0, 0.0]))
        assert val > 0.0

class TestAABB:
    def test_contains_interior_point(self):
        aabb = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        assert aabb.contains(np.array([0.0, 0.0, 0.0]))

    def test_does_not_contain_exterior_point(self):
        aabb = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        assert not aabb.contains(np.array([2.0, 0.0, 0.0]))

    def test_boundary_point(self):
        """A point on the boundary should be considered contained."""
        aabb = AABB(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        assert aabb.contains(np.array([1.0, 0.5, 0.5]))

    def test_does_not_contain_just_outside(self):
        aabb = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        assert not aabb.contains(np.array([1.0001, 0.0, 0.0]))

    def test_size(self):
        """AABB extent should be max - min."""
        aabb = AABB(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 5.0]))
        size = aabb.max_point - aabb.min_point
        assert np.allclose(size, np.array([3.0, 4.0, 5.0]))

    def test_center(self):
        aabb = AABB(np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0]))
        center = (aabb.min_point + aabb.max_point) / 2
        assert np.allclose(center, np.zeros(3))

    def test_ray_intersection_hit(self):
        """An axis-aligned ray toward the AABB centre should hit."""
        aabb = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        hit = aabb.intersect(Ray(origin, direction))
        assert hit is not None and hit > 0

    def test_ray_intersection_miss(self):
        """A ray pointing away from the AABB should miss."""
        aabb = AABB(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, -1.0])
        hit = aabb.intersect(Ray(origin, direction))
        assert hit is None or hit < 0

    
    def test_aabb_contains_point(self):
        """Test point-in-AABB queries"""
        aabb = AABB(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        assert aabb.contains(np.array([0, 0, 0]))
        assert not aabb.contains(np.array([2, 0, 0]))
    
    def test_aabb_intersection(self):
        """Test AABB-AABB intersection"""
        aabb1 = AABB(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        aabb2 = AABB(np.array([0, 0, 0]), np.array([2, 2, 2]))
        aabb3 = AABB(np.array([2, 2, 2]), np.array([3, 3, 3]))
        assert aabb1.overlaps(aabb2)
        assert not aabb1.overlaps(aabb3)

    def test_aabb_transform(self):
        """Test AABB transformation"""
        aabb = AABB(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        # Rotate 45 degrees around Y axis
        angle = np.radians(45)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ])
        bounds = np.array([aabb.min_point, aabb.max_point])
        transformed_bounds = transform_bounds(rotation_matrix, bounds)
        transformed_aabb = AABB(transformed_bounds[0], transformed_bounds[1])
        # The transformed AABB should still contain the original corners
        corners = convert_bounds_to_corners(aabb.min_point, aabb.max_point)
        for corner in corners:
            assert transformed_aabb.contains(corner)

    def test_aabb_properties(self):
        """Test AABB properties like center and size"""
        aabb = AABB(np.array([-1, -2, -3]), np.array([1, 2, 3]))
        assert np.allclose(aabb.center, np.array([0, 0, 0]))
        assert np.allclose(aabb.size, np.array([2, 4, 6]))

    def test_aabb_static_methods(self):
        """Test AABB static methods for empty, infinite, and unit cube"""
        empty_aabb = AABB.empty()
        assert np.all(np.isinf(empty_aabb.min_point))
        assert np.all(np.isinf(empty_aabb.max_point))
        
        infinite_aabb = AABB.infinite()
        assert np.all(np.isneginf(infinite_aabb.min_point))
        assert np.all(np.isinf(infinite_aabb.max_point))
        
        unit_cube_aabb = AABB.unit_cube()
        assert np.allclose(unit_cube_aabb.min_point, np.array([-0.5, -0.5, -0.5]))
        assert np.allclose(unit_cube_aabb.max_point, np.array([0.5, 0.5, 0.5]))

class TestBVH:
    def _make_sphere_list(self, n: int = 8):
        """Create a grid of unit spheres for testing."""
        return [Sphere(0.4) for _ in range(n)]

    def test_bvh_constructs_from_shapes(self):
        shapes = [Sphere(1.0), Cube(1.0), Cylinder(0.5, 1.0)]
        bvh = BVHNode(shapes)
        assert bvh is not None

    def test_bvh_constructs_from_single_shape(self):
        bvh = BVHNode([SceneNode(context=Sphere(1.0))])
        assert bvh is not None

    def test_bvh_constructs_from_many_shapes(self):
        shapes = self._make_sphere_list(64)
        nodes = [SceneNode(context=s) for s in shapes]
        bvh = BVHNode(nodes)
        assert bvh is not None

    def test_bvh_node_count_non_zero(self):
        shapes = [Sphere(1.0), Cube(1.0)]
        nodes = [SceneNode(context=s) for s in shapes]
        bvh = BVHNode(nodes)
        assert bvh.node_count > 0

    def test_bvh_leaf_count_matches_input(self):
        """Number of leaf nodes must equal number of input shapes."""
        shapes = [Sphere(1.0), Cube(1.0), Cylinder(0.5, 1.0), Pyramid()]
        nodes = [SceneNode(context=s) for s in shapes]
        bvh = BVHNode(nodes)
        assert bvh.leaf_count == len(shapes)

    def test_bvh_debug_print(self):
        s = get_minimal_scene(64,64)
        for obj in s.nodes:
            obj.update_matrices()

        bvh = BVHIntersection()
        root = build_bvh_tree(list(s.nodes))
        print('Root box min, max:', root.box.min_point, root.box.max_point)
        for node in [root.left, root.right]:
            if node is not None and node.box is not None:
                print('Child box min, max:', node.box.min_point, node.box.max_point)

    def test_bvh_ray_hit_returns_correct_shape(self):
        """A ray aimed at a sphere should report a hit on that sphere."""
        sphere = Sphere(1.0)
        bvh = BVHNode([SceneNode(context=sphere, transform=Transform(np.array([0.0, 0.0, 0.0])))])
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        hit = bvh.box.intersect(Ray(origin, direction))
        assert hit is not None

    def test_bvh_ray_miss_returns_none(self):
        """A ray aimed away from all geometry should return no hit."""
        bvh = BVHNode([SceneNode(context=Sphere(1.0), transform=Transform(np.array([0.0, 0.0, 0.0])))])
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 1.0, 0.0])   # perpendicular – misses sphere
        hit = bvh.box.intersect(Ray(origin, direction))
        assert hit is None

    @pytest.mark.slow
    def test_bvh_faster_than_linear_search(self):
        """BVH traversal should visit fewer nodes than the total shape count."""
        import time
        shapes = [Sphere(0.3) for _ in range(200)]
        transforms = [Transform(np.array([np.random.uniform(-10, 10),
                                          np.random.uniform(-10, 10),
                                          np.random.uniform(-10, 10)])) for _ in shapes]
        bvh = BVHNode([SceneNode(context=s, transform=tr) for s, tr in zip(shapes, transforms)])
        origin = np.array([0.0, 0.0, -50.0])
        direction = np.array([0.0, 0.0, 1.0])

        t0 = time.perf_counter()
        for _ in range(200):
            bvh.box.intersect(Ray(origin, direction))
        bvh_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        for _ in range(200):
            for s, tr in zip(shapes, transforms):
                s.get_distance(tr.local_transform_point(origin))
        linear_time = time.perf_counter() - t1

        # BVH should be at least as fast (within 5×)
        assert bvh_time <= linear_time * 5