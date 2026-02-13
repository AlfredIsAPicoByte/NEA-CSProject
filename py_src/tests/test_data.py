import pytest
import numpy as np
from src.Data.Transform import Transform
from src.Data.Color import Color, ColorGradient
from src.Data.Ray import Ray
from src.Data.Camera import Camera, CameraType
from src.Data.Scene import Scene, SceneNode


class TestTransform:
    def test_identity_transform(self):
        """Identity transform should leave any point unchanged."""
        t = Transform.Identity()
        point = np.array([1.0, 2.0, 3.0])
        assert np.allclose(t.local_transform_point(point), point)
        assert np.allclose(t.world_transform_point(point), point)

    def test_translation(self):
        """Translating the origin by [1, 2, 3] should yield [1, 2, 3]."""
        t = Transform.Identity()
        t.translate(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(t.position, np.array([1.0, 2.0, 3.0]))

    def test_translation_additive(self):
        """Translation should add to an existing point."""
        t = Transform(np.array([1.0, 0.0, 0.0]))
        point = np.array([2.0, 3.0, 4.0])
        result = t.translate(point)
        assert np.allclose(result, np.array([3.0, 3.0, 4.0]))

    def test_translation_negative(self):
        """Negative translation should move in the opposite direction."""
        t = Transform(np.array([-1.0, -2.0, -3.0]))
        t.translate(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(t.position, np.array([0.0, 0.0, 0.0]))

    def test_rotation_90_degrees_y(self):
        """90° rotation around Y should map +X to +Z (right-hand convention)."""
        t = Transform(np.zeros(3), rotation=np.array([0.0, np.deg2rad(90), 0.0]))
        point = np.array([1.0, 0.0, 0.0])
        result = t.apply(point)
        # After 90° CCW around Y: +X → −Z
        assert np.allclose(result, np.array([0.0, 0.0, 1.0]), atol=1e-6)

    def test_uniform_scale(self):
        """Uniform scale by 2 should double all coordinates."""
        t = Transform(np.zeros(3), scale=np.full(3, 2.0))
        point = np.array([1.0, 1.0, 1.0])
        assert np.allclose(t.apply(point), np.array([2.0, 2.0, 2.0]))

    def test_non_uniform_scale(self):
        """Non-uniform scale should scale each axis independently."""
        t = Transform(np.zeros(3), scale=np.array([2.0, 3.0, 4.0]))
        point = np.array([1.0, 1.0, 1.0])
        assert np.allclose(t.apply(point), np.array([2.0, 3.0, 4.0]))

    def test_transform_composition_translation(self):
        """Two sequential translations should accumulate."""
        t1 = Transform(np.array([1.0, 0.0, 0.0]))
        t2 = Transform(np.array([0.0, 1.0, 0.0]))
        origin = np.array([0.0, 0.0, 0.0])
        result = t2.apply(t1.apply(origin))
        assert np.allclose(result, np.array([1.0, 1.0, 0.0]))

    def test_identity_position_is_zero(self):
        """Identity transform position should be zero."""
        t = Transform.Identity()
        assert np.allclose(t.position, np.zeros(3))

    def test_look_at_changes_rotation(self):
        """look_at should reorient the transform toward the target."""
        t = Transform(np.array([0.0, 0.0, -5.0]))
        t.look_at(np.array([0.0, 0.0, 0.0]))
        # After look_at the forward vector should point from position toward target
        forward = t.apply(np.array([0.0, 0.0, 1.0])) - t.position
        direction = np.array([0.0, 0.0, 0.0]) - np.array([0.0, 0.0, -5.0])
        direction /= np.linalg.norm(direction)
        assert np.allclose(forward / np.linalg.norm(forward), direction, atol=1e-5)

    def test_scale_one_is_identity_scale(self):
        """Scale of 1 should not change coordinates."""
        t = Transform(np.zeros(3), scale=np.ones(3))
        point = np.array([3.0, -2.0, 7.0])
        assert np.allclose(t.apply(point), point)


class TestColor:
    def test_from_hex_red(self):
        c = Color.from_hex("#FF0000")
        assert np.isclose(c.r, 1.0) and np.isclose(c.g, 0.0) and np.isclose(c.b, 0.0)

    def test_from_hex_white(self):
        c = Color.from_hex("#FFFFFF")
        assert np.isclose(c.r, 1.0) and np.isclose(c.g, 1.0) and np.isclose(c.b, 1.0)

    def test_from_hex_black(self):
        c = Color.from_hex("#000000")
        assert np.isclose(c.r, 0.0) and np.isclose(c.g, 0.0) and np.isclose(c.b, 0.0)

    def test_from_hex_arbitrary(self):
        """#4080C0 should map to approximately (0.251, 0.502, 0.753)."""
        c = Color.from_hex("#4080C0")
        assert np.isclose(c.r, 0x40 / 255, atol=1e-3)
        assert np.isclose(c.g, 0x80 / 255, atol=1e-3)
        assert np.isclose(c.b, 0xC0 / 255, atol=1e-3)

    def test_color_channels_in_range(self):
        """All channels of any Color should be in [0, 1]."""
        c = Color(0.3, 0.6, 0.9)
        for ch in (c.r, c.g, c.b):
            assert 0.0 <= ch <= 1.0

    def test_color_equality(self):
        c1 = Color(0.5, 0.5, 0.5)
        c2 = Color(0.5, 0.5, 0.5)
        assert np.isclose(c1.r, c2.r) and np.isclose(c1.g, c2.g) and np.isclose(c1.b, c2.b)

    def test_from_wavelength_red_range(self):
        """Wavelength ~670 nm should produce a predominantly red colour."""
        c = Color.from_wavelength(670)
        assert c.r > c.g and c.r > c.b

    def test_from_wavelength_blue_range(self):
        """Wavelength ~450 nm should produce a predominantly blue colour."""
        c = Color.from_wavelength(450)
        assert c.b > c.r

    def test_from_kelvin_warm(self):
        """Low colour temperature (~2000 K) should be warmer (more red) than cool."""
        warm = Color.from_kelvin(2000)
        cool = Color.from_kelvin(10000)
        assert warm.r >= cool.r

    def test_from_kelvin_cool(self):
        """High colour temperature (~10 000 K) should be cooler (more blue)."""
        warm = Color.from_kelvin(2000)
        cool = Color.from_kelvin(10000)
        assert cool.b >= warm.b

    def test_from_hsl_pure_red(self):
        """HSL(0°, 100%, 50%) = pure red."""
        c = Color.from_hsl(0.0, 100.0, 50.0)
        assert np.isclose(c.r, 1.0, atol=1e-3)
        assert np.isclose(c.g, 0.0, atol=1e-3)
        assert np.isclose(c.b, 0.0, atol=1e-3)

    def test_from_hsv_pure_blue(self):
        """HSV(240°, 100%, 100%) = pure blue."""
        c = Color.from_hsv(240.0, 1.0, 1.0)
        assert c.b > c.r and c.b > c.g

    def test_from_cmyk_black(self):
        """CMYK(0, 0, 0, 1) should be black."""
        c = Color.from_cmyk(0.0, 0.0, 0.0, 1.0)
        assert np.isclose(c.r, 0.0, atol=1e-3)
        assert np.isclose(c.g, 0.0, atol=1e-3)
        assert np.isclose(c.b, 0.0, atol=1e-3)


class TestColorGradient:
    def test_gradient_at_zero(self):
        """Gradient evaluated at t=0 should return the first colour."""
        colors = [Color(1.0, 0.0, 0.0), Color(0.0, 0.0, 1.0)]
        positions = np.array([0.0, 1.0])
        grad = ColorGradient(colors, positions)
        c = grad.get_color(0.0)
        assert np.isclose(c.r, 1.0, atol=1e-3) and np.isclose(c.b, 0.0, atol=1e-3)

    def test_gradient_at_one(self):
        """Gradient evaluated at t=1 should return the last colour."""
        colors = [Color(1.0, 0.0, 0.0), Color(0.0, 0.0, 1.0)]
        positions = np.array([0.0, 1.0])
        grad = ColorGradient(colors, positions)
        c = grad.get_color(1.0)
        assert np.isclose(c.b, 1.0, atol=1e-3) and np.isclose(c.r, 0.0, atol=1e-3)

    def test_gradient_midpoint(self):
        """Midpoint of a red→blue gradient should be ~(0.5, 0, 0.5)."""
        colors = [Color(1.0, 0.0, 0.0), Color(0.0, 0.0, 1.0)]
        positions = np.array([0.0, 1.0])
        grad = ColorGradient(colors, positions)
        c = grad.get_color(0.5)
        assert np.isclose(c.r, 0.5, atol=0.05) and np.isclose(c.b, 0.5, atol=0.05)

    def test_gradient_multi_stop(self):
        """Three-stop gradient should interpolate between the correct pair."""
        colors = [Color(1.0, 0.0, 0.0), Color(0.0, 1.0, 0.0), Color(0.0, 0.0, 1.0)]
        positions = np.array([0.0, 0.5, 1.0])
        grad = ColorGradient(colors, positions)
        # At t=0.5 we should be at the middle stop (pure green)
        c = grad.get_color(0.5)
        assert np.isclose(c.g, 1.0, atol=0.05)


class TestCamera:
    def _make_cam(self, width=800, height=600, fov=90.0):
        cam_transform = Transform(np.array([0.0, 0.0, 0.0]))
        return Camera(
            cam_transform,
            fov=fov, near=0.1, far=1000.0,
            resolution_width=width, resolution_height=height,
            camera_type=CameraType.PERSPECTIVE
        )

    def test_camera_resolution_stored(self):
        cam = self._make_cam(320, 240)
        assert cam.resolution_width == 320 and cam.resolution_height == 240

    def test_camera_fov_stored(self):
        cam = self._make_cam(fov=75.0)
        assert np.isclose(cam.fov, 75.0)

    def test_camera_near_far_stored(self):
        cam_transform = Transform(np.zeros(3))
        cam = Camera(cam_transform, fov=60.0, near=0.5, far=500.0,
                     resolution_width=100, resolution_height=100)
        assert np.isclose(cam.near, 0.5) and np.isclose(cam.far, 500.0)

    def test_camera_default_type_perspective(self):
        cam = self._make_cam()
        assert cam.camera_type == CameraType.PERSPECTIVE

    def test_camera_rays_have_unit_direction(self):
        """Every generated ray direction should be normalised."""
        cam = self._make_cam(32, 32, fov=60.0)
        rays = cam.generate_rays()           # expected: ndarray of shape (H, W, ?) or similar
        # Tolerant iteration regardless of exact shape
        dirs = np.array(rays).reshape(-1, 3)
        norms = np.linalg.norm(dirs, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_centre_ray_points_forward(self):
        """The centre ray of a camera looking along +Z should point in +Z."""
        cam_transform = Transform(np.array([0.0, 0.0, 0.0]))
        cam = Camera(cam_transform, fov=90.0, near=0.1, far=100.0,
                     resolution_width=101, resolution_height=101,
                     camera_type=CameraType.PERSPECTIVE)
        # Look along +Z
        cam.transform.look_at(np.array([0.0, 0.0, 1.0]))
        rays = cam.generate_rays()
        cx, cy = 50, 50          # centre pixel
        centre_dir = np.array(rays).reshape(101, 101, 3)[cy, cx]
        centre_dir /= np.linalg.norm(centre_dir)
        assert np.allclose(centre_dir, np.array([0.0, 0.0, 1.0]), atol=0.05)

class TestScene:
    def test_scene_initialisation(self):
        """A new Scene should have empty geometry and materials."""
        scene = Scene()
        assert len(scene.nodes) == 0
        assert scene.version == 0

    def test_scene_node_addition(self):
        """Adding a node to the scene should increase the node count."""
        scene = Scene()
        node = SceneNode(name="TestNode")
        scene.add_node(node)
        assert len(scene.nodes) == 1
        assert scene.nodes[0].name == "TestNode"

    def test_scene_version_increment(self):
        """Adding a node should increment the scene version."""
        scene = Scene()
        initial_version = scene.version
        node = SceneNode(name="VersionTestNode")
        scene.add_node(node)
        assert scene.version == initial_version + 1

    def test_scene_node_hierarchy(self):
        """Nodes should be able to have child nodes, forming a hierarchy."""
        parent_node = SceneNode(name="Parent")
        child_node = SceneNode(name="Child")
        parent_node.add_child(child_node)
        assert len(parent_node.children) == 1
        assert parent_node.children[0].name == "Child"

    def test_scene_node_transform(self):
        """A SceneNode should have a Transform that can be modified."""
        node = SceneNode(name="TransformTest")
        initial_position = node.transform.position.copy()
        node.transform.position += np.array([1.0, 2.0, 3.0])
        assert np.allclose(node.transform.position, initial_position + np.array([1.0, 2.0, 3.0]))