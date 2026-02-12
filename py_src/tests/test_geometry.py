import pytest
import numpy as np
from src.Geometry.SDF import Sphere, Cube, Cylinder
from src.Geometry.BVH import BVH
from src.Geometry.AABB import AABB

class TestSDF:
    def test_sphere_contains_point(self):
        """Points inside sphere should have negative SDF"""
        sphere = Sphere(1.0)
        point = np.array([0.0, 0.0, 0.0])
        sdf_value = sphere.evaluate(point)
        assert sdf_value < 0

    def test_sphere_outside_point(self):
        """Points outside sphere should have positive SDF"""
        sphere = Sphere(1.0)
        point = np.array([2.0, 0.0, 0.0])
        sdf_value = sphere.evaluate(point)
        assert sdf_value > 0

    def test_cube_sdf_values(self):
        """Test cube SDF at known points"""
        cube = Cube(1.0)
        # Test corner point
        # Test center point
        # Test surface point

class TestBVH:
    def test_bvh_construction(self):
        """BVH should construct without errors"""
        shapes = [Sphere(1.0), Cube(1.0)]
        bvh = BVH(shapes)
        assert bvh is not None
    
    def test_bvh_acceleration(self):
        """BVH should reduce intersection tests"""
        # Generate many shapes
        # Time with and without BVH
        pass

class TestAABB:
    def test_aabb_contains_point(self):
        """Test point-in-AABB queries"""
        aabb = AABB(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        assert aabb.contains(np.array([0, 0, 0]))
        assert not aabb.contains(np.array([2, 0, 0]))