import pytest
import numpy as np

from src.Data.Ray import Ray
from src.Geometry.SDF import Sphere

IMG_OUTPUT_DIR = 'images/testing'

class TestRayIntersection:
    def test_ray_sphere_intersection(self):
        """Simple ray-sphere intersection"""
        ray = Ray(np.array([0, 0, -5]), np.array([0, 0, 1]))
        sphere = Sphere(1.0)
        # Test that intersection is at z=1
        
    def test_ray_misses_geometry(self):
        """Ray that misses should return no hit"""
        ray = Ray(np.array([0, 0, -5]), np.array([0, 0, 1]))
        sphere = Sphere(1.0)
        # sphere positioned so ray misses
        # assert no hit

class TestShading:
    def test_diffuse_shading(self):
        """Test diffuse shading with simple light and normal"""
        # Set up normal, light direction, material properties
        # Compute shading
        # Assert expected color output

    def test_specular_shading(self):
        """Test specular shading with simple light and view direction"""
        # Set up view direction, light direction, normal, material properties
        # Compute shading
        # Assert expected color output

class TestRendering:
    @pytest.mark.slow
    def test_rendering_pipeline(self):
        """Test basic rendering pipeline with simple scene"""
        # Set up camera, geometry, materials
        # Render scene
        # Assert that output image has expected properties (e.g. non-zero pixels)
        pass
    
    @pytest.mark.slow    
    def test_rendering_consistency(self): 
        """Multiple renders of same scene should produce same output"""
        # Set up scene 
        # Render twice
        # Assert outputs are identical pass
        pass