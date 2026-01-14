import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

sys.path.insert(0, current_dir)

from src.Data.Transform import Transform
from src.Data.Ray import Ray
from src.Data.Color import Color, ColorGradient
from src.Data.Ratio import Ratio
from src.Geometry.Core import Sphere
from src.Geometry.Primitive import Primitive
from src.Lighting.Core import LightSource
from src.Material.Core import PBRMaterial
from src.Rendering.Raytracing import Raytracer
from src.Rendering.Shading import LambertShading
from src.Utilities.Camera import Camera, CameraType
from src.Utilities.Scene import Scene

def test_ray_structure():
    """Test basic Ray structure functionality."""
    origin = np.zeros(3)
    direction = np.array([1, 0, 0])
    ray = Ray(origin, direction)

    # Use numpy's built-in assertion tools for arrays
    np.testing.assert_allclose(ray.origin, origin, err_msg="Ray origin mismatch")
    np.testing.assert_allclose(ray.direction, direction, err_msg="Ray direction mismatch")

    point_at_5 = ray.point_at(5)
    expected_point = np.array([5, 0, 0])
    np.testing.assert_allclose(point_at_5, expected_point, err_msg="Ray point_at calculation incorrect")

# We use parametrize to run the same test logic on different inputs
@pytest.mark.parametrize("shape_name, args", [
    ("Circle", [[0, 0], 1]),
    ("Triangle", [[0, 0], [1, 0], [0, 1]]),
    ("Sphere", [[0, 0, 0], 1]),
    ("Cube", [[0, 0, 0], 1]),
])
def test_shape_creation(shape_name, args):
    """Dynamically tests that all registered shapes can be created via Factory."""
    # Import factory dynamically to match your original logic, or import directly if preferred
    module_name = "src.Geometry"
    factory_class_name = f"{shape_name}Factory"
    
    module = __import__(module_name, fromlist=[factory_class_name])
    factory_class = getattr(module, factory_class_name)
    
    factory = factory_class()
    shape = factory.create(*args)
    
    assert shape is not None, f"Factory {shape_name} returned None"

def test_transform_operations():
    pos = np.array([1.0, 2.0, 3.0])
    rot = np.array([0.0, 0.0, 0.0])
    scale = np.array([1.0, 1.0, 1.0])
    
    t = Transform(pos, rot, scale)
    
    # Test Translation
    t.translate(np.array([1.0, 0.0, 0.0]))
    expected_pos = np.array([2.0, 2.0, 3.0])
    np.testing.assert_allclose(t.position, expected_pos, err_msg="Translation failed")

    # Test Rotation
    t.rotate(np.pi/2, np.array([0, 1, 0]))
    expected_rot = np.array([0, np.pi/2, 0])
    np.testing.assert_allclose(t.rotation, expected_rot, err_msg="Rotation failed")

    # Test Scaling
    t.enlarge(np.array([2.0, 2.0, 2.0]))
    expected_scale = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(t.scale, expected_scale, err_msg="Scaling failed")

def test_ratios():
    r = Ratio(16, 9)
    assert r.width == 16
    assert r.height == 9
    assert abs(r.value - (16/9)) < 1e-6

@pytest.mark.parametrize("t, expected_point, should_match", [
    (0,   [0, 0, 0], True),
    (1,   [0, 1, 0], True),
    (5,   [0, 5, 0], True),
    (-3,  [0, -3, 0], True),
    (100, [0, 50, 0], False), # Intentionally incorrect in your data
    (-12, [0, -6, 0], False),
])
def test_ray_check_points(t, expected_point, should_match):
    ray = Ray(np.zeros(3), np.array([0, 1, 0]))
    point = ray.point_at(t)
    
    is_close = np.allclose(point, np.array(expected_point))
    
    if should_match:
        assert is_close, f"Point at t={t} was {point}, expected {expected_point}"
    else:
        assert not is_close, f"Point at t={t} matched {expected_point} but shouldn't have"

def test_Primitive_creation():
    transform = Transform.identity()
    shape = Sphere()
    
    obj = Primitive(shape, transform)
    
    assert obj.shape == shape
    assert obj.transform == transform

def test_color_math():
    c1 = Color(0.2, 0.4, 0.6)
    c2 = Color(0.1, 0.2, 0.3)
    
    # Addition
    added = c1 + c2
    np.testing.assert_allclose([added.r, added.g, added.b], [0.3, 0.6, 0.9])
    
    # Scaling
    scaled = c1 * 2
    np.testing.assert_allclose([scaled.r, scaled.g, scaled.b], [0.4, 0.8, 1.2])

def test_camera_logic():
    transform = Transform.identity()
    cam = Camera(transform, 90, 0.1, 1000, 1440, 810)
    
    cam.aspect_ratio.simplify()
    assert cam.aspect_ratio.width == 16 and cam.aspect_ratio.height == 9 # Assuming internally simplified

    # Test Resize
    cam.resize_aspect(Ratio(4, 3), 110)
    cam.aspect_ratio.simplify()
    assert cam.aspect_ratio.width == 4 and cam.aspect_ratio.height == 3

def test_ray_shape_intersection():
    sphere = Sphere()
    
    ray_hit = Ray(np.zeros(3), np.array([1, 0, 0])) # Originates inside
    ray_miss = Ray(np.array([2, 2, 2]), np.array([1, 0, 0]))
    
    assert sphere.check_ray_intersection(ray_hit) == True
    assert sphere.check_ray_intersection(ray_miss) == False

def test_background_gradient():
    cam = Camera(Transform(np.array([0,0,-3]), np.zeros(3), np.ones(3)), 60, 0.1, 100, 8, 8, CameraType.PERSPECTIVE)
    grad = ColorGradient([Color.from_hex("#000033"), Color.from_hex("#87CEEB")], np.array([0.0, 1.0]))
    scene = Scene(name="bg_test", camera=cam, background_color=grad)
    
    up_color = scene.get_background_color([0.0, 1.0, 0.0])
    down_color = scene.get_background_color([0.0, -1.0, 0.0])
    
    # Helper to get array from Color object
    def get_rgb(c):
        if hasattr(c, "to_np_ndarray"): return c.to_np_ndarray()[:3]
        return np.array([c.red, c.green, c.blue])

    uc = get_rgb(up_color)
    dc = get_rgb(down_color)
    
    # Assert they are not equal (gradient works)
    assert not np.allclose(uc, dc), "Gradient should vary by direction"
    
    # Assert they are not magenta (error color)
    magenta = np.array([1.0, 0.0, 1.0])
    assert not np.allclose(uc, magenta), "Returned fallback magenta"

def test_ambient_lighting():
    # Setup scene
    cam = Camera(
        transform=Transform(np.array([0,0,-5]), np.zeros(3), np.ones(3)),
        fov=60,
        near=0.1,
        far=100,
        resolution_width=4, resolution_height=4,
        camera_type=CameraType.PERSPECTIVE
    )
    light = LightSource(position=np.array([10,10,-10]), color=Color(1.0, 1.0, 1.0), intensity=0.5)
    material = PBRMaterial.create_diffuse(
        albedo=Color(0.8, 0.8, 0.8),
        roughness=0.5,
    )
    sphere = Primitive(Sphere(), Transform.identity())
    # Attach PBR data to shape for compatibility
    sphere.material = material
    scene = Scene(name="ambient_test", camera=cam, objects=[sphere], lights=[
        light
    ], background_color=Color(0.0, 0.0, 0.0))
    raytracer = Raytracer(
        max_depth=2,
        sampling_manager=None,
        ray_generator=None,
        intersection_strategy=None,
        interaction_strategy=None,
        shading_strategy=LambertShading(),
        custom_background=scene.get_background_color([0.0, 0.0, -1.0]),
        enable_scene_background=True
    )
    pixels = raytracer.render(scene)
    # Check center pixel
    if scene.camera is None:
        raise ValueError("Camera is missing!")
    center_index = int((scene.camera.resolution_height // 2) * scene.camera.resolution_width + (scene.camera.resolution_width // 2))
    center_pixel = pixels[center_index]
    rgb = np.array([center_pixel.r, center_pixel.g, center_pixel.b])

    # Ensure it isn't Black
    assert not np.allclose(rgb, 0.0), f"Result was black {rgb}, expected ambient gray"
    assert not np.allclose(rgb, 1.0), f"Result was white {rgb}, expected ambient gray"

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))