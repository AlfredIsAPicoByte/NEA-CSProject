import pytest
import numpy as np
from src.Data.Transform import Transform
from src.Data.Color import Color, ColorGradient
from src.Data.Ray import Ray
from src.Data.Camera import Camera, CameraType

class TestTransform:
    def test_identity_transform(self):
        """Identity transform should have no effect"""
        t = Transform.Identity()
        point = np.array([1.0, 2.0, 3.0])
        result = t.apply(point)
        assert np.allclose(result, point)
    
    def test_translation(self):
        """Test translation operations"""
        t = Transform(np.array([1.0, 2.0, 3.0]))
        point = np.array([0.0, 0.0, 0.0])
        result = t.apply(point)
        assert np.allclose(result, np.array([1.0, 2.0, 3.0]))
    
    def test_transform_composition(self):
        """Test combining multiple transforms"""
        t1 = Transform(np.array([1.0, 0.0, 0.0]))
        t2 = Transform(np.array([0.0, 1.0, 0.0]))
        # Verify composition works as expected

class TestColor:
    def test_color_from_hex(self):
        """Test hex to RGB conversion"""
        c = Color.from_hex("#FF0000")
        assert c.r == 1.0 and c.g == 0.0 and c.b == 0.0
    
    def test_color_blending(self):
        """Test color interpolation"""
        c1 = Color(1.0, 0.0, 0.0)  # Red
        c2 = Color(0.0, 0.0, 1.0)  # Blue
        # Test blend at t=0.5

class TestCamera:
    def test_camera_projection(self):
        """Test camera projection of a point"""
        cam = Camera(position=np.array([0, 0, 0]), look_at=np.array([0, 0, -1]), up=np.array([0, 1, 0]), fov=90, width=800, height=600)
        point = np.array([0, 0, -5])
        projected = cam.project(point)
        # Assert projected coordinates are correct for center of view
        assert np.allclose(projected, np.array([400.0, 300.0]))