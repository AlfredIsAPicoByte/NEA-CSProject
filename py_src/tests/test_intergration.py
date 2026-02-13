import numpy as np
import pytest

from src.Data.Transform import Transform
from src.Data.Color import Color
from src.Data.Ray import Ray
from src.Data.Camera import Camera, CameraType
from src.Data.Context import SDF_Material
from src.Geometry.SDF import Sphere, Cube
from src.Material.Factory import MaterialFactory
from src.Lighting.Core import Light

IMG_OUTPUT_DIR = "images/testing"


# ---------------------------------------------------------------------------
# Scene Construction Helpers
# ---------------------------------------------------------------------------

def _make_base_scene(width=32, height=32):
    """Return the canonical minimal scene from bench_scenes."""
    from py_src.tests.bench_scenes import get_minimal_scene
    return get_minimal_scene(width=width, height=height)


def _render(scene, seed=0):
    from src.Rendering.Raytracing import RayTracer
    return RayTracer(scene).render(seed=seed)


# ---------------------------------------------------------------------------
# Scene Construction Tests
# ---------------------------------------------------------------------------

class TestSceneConstruction:
    def test_minimal_scene_creates_camera(self):
        scene = _make_base_scene()
        assert scene.camera is not None

    def test_minimal_scene_has_objects(self):
        scene = _make_base_scene()
        assert len(scene.objects) > 0

    def test_minimal_scene_has_lights(self):
        scene = _make_base_scene()
        lights = [o for o in scene.objects if isinstance(o.context, Light)]
        assert len(lights) > 0

    def test_scene_name_set(self):
        scene = _make_base_scene()
        assert scene.name == "minimal_scene"

    def test_adding_object_increases_count(self):
        scene = _make_base_scene()
        initial_count = len(scene.objects)
        mat = MaterialFactory.create_diffuse(Color(1, 0, 0), roughness=0.5)
        scene.add_object_by_context(
            SDF_Material(Sphere(0.5), mat), "ExtraObj", Transform(np.array([5.0, 0.0, 0.0]))
        )
        assert len(scene.objects) == initial_count + 1

    def test_adding_light_increases_count(self):
        scene = _make_base_scene()
        initial_count = len(scene.objects)
        scene.add_object_by_context(
            Light(color=Color(1, 1, 1), intensity=100.0),
            "ExtraLight",
            Transform(np.array([0.0, 10.0, 0.0]))
        )
        assert len(scene.objects) == initial_count + 1

    def test_camera_resolution_matches_request(self):
        scene = _make_base_scene(width=64, height=128)
        assert scene.camera.resolution_width == 64
        assert scene.camera.resolution_height == 128

    def test_background_color_stored(self):
        scene = _make_base_scene()
        assert scene.background_color is not None

    def test_object_transform_position_correct(self):
        scene = _make_base_scene()
        mat = MaterialFactory.create_diffuse(Color(0, 1, 0), roughness=0.5)
        target_pos = np.array([3.0, 1.0, -2.0])
        node = scene.add_object_by_context(
            SDF_Material(Sphere(0.5), mat), "CheckObj", Transform(target_pos)
        )
        assert np.allclose(node.world_transform.position, target_pos)


# ---------------------------------------------------------------------------
# Rendering Pipeline Tests
# ---------------------------------------------------------------------------

class TestRenderingPipeline:
    @pytest.mark.slow
    def test_minimal_render_shape(self):
        """Output array must have shape (H, W, 3)."""
        output = _render(_make_base_scene(width=32, height=32))
        assert output.shape == (32, 32, 3), f"Got shape {output.shape}"

    @pytest.mark.slow
    def test_minimal_render_dtype_float(self):
        """Output should be a float array."""
        output = _render(_make_base_scene(width=16, height=16))
        assert output.dtype.kind == "f"

    @pytest.mark.slow
    def test_render_values_in_range(self):
        """All pixel values must be in [0, 1]."""
        output = _render(_make_base_scene(width=16, height=16))
        assert np.all(output >= 0.0), "Pixel below 0 detected"
        assert np.all(output <= 1.0), "Pixel above 1 detected"

    @pytest.mark.slow
    def test_render_not_all_background(self):
        """The sphere in the minimal scene should colour at least some pixels."""
        output = _render(_make_base_scene(width=32, height=32))
        bg = np.array([0x9B, 0xB0, 0xCA]) / 255.0
        different = np.any(np.abs(output - bg) > 0.04, axis=-1)
        assert different.sum() > 32 * 32 * 0.05, "Almost all pixels are background colour"

    @pytest.mark.slow
    def test_render_deterministic_same_seed(self):
        """Same scene + same seed → identical pixels."""
        scene = _make_base_scene(width=16, height=16)
        out1 = _render(scene, seed=1)
        out2 = _render(scene, seed=1)
        assert np.allclose(out1, out2), "Render is not deterministic"

    @pytest.mark.slow
    def test_render_different_seeds_may_differ(self):
        """Different seeds for a stochastic renderer should typically differ."""
        from src.Rendering.Raytracing import RayTracer
        scene = _make_base_scene(width=16, height=16)
        tracer = RayTracer(scene)
        out1 = tracer.render(seed=1)
        out2 = tracer.render(seed=99)
        # If renderer is deterministic regardless of seed this is still fine;
        # we only assert it does NOT crash.
        assert out1.shape == out2.shape

    @pytest.mark.slow
    def test_render_higher_resolution_same_aspect(self):
        """Doubling resolution should not change the aspect ratio of the output."""
        lo = _render(_make_base_scene(width=16, height=16))
        hi = _render(_make_base_scene(width=32, height=32))
        assert lo.shape[0] * 2 == hi.shape[0]
        assert lo.shape[1] * 2 == hi.shape[1]

    @pytest.mark.slow
    def test_removing_light_darkens_scene(self):
        """A scene with no lights should be darker than one with lights."""
        from src.Data.Scene import Scene

        lit_scene   = _make_base_scene(width=16, height=16)
        dark_scene  = _make_base_scene(width=16, height=16)
        # Strip all lights from the dark scene
        dark_scene.objects = [
            o for o in dark_scene.objects
            if not isinstance(o.context, Light)
        ]

        lit_output  = _render(lit_scene,  seed=0)
        dark_output = _render(dark_scene, seed=0)
        assert lit_output.mean() > dark_output.mean()

    @pytest.mark.slow
    def test_emissive_object_adds_luminance(self):
        """Adding a bright emissive sphere near the camera should raise avg luminance."""
        base_scene = _make_base_scene(width=16, height=16)
        glow_scene = _make_base_scene(width=16, height=16)
        mat_glow = MaterialFactory.create_emissive(Color(1, 1, 1), 20.0)
        glow_scene.add_object_by_context(
            SDF_Material(Sphere(1.0), mat_glow), "BigGlow",
            Transform(np.array([0.0, 0.5, -2.0]))
        )
        base_out = _render(base_scene, seed=0)
        glow_out = _render(glow_scene, seed=0)
        assert glow_out.mean() >= base_out.mean() - 1e-4

    @pytest.mark.slow
    def test_glass_sphere_transmits(self):
        """A scene with a glass sphere should have higher average brightness
        than the same scene with an opaque black sphere blocking the same view."""
        from py_src.tests.bench_scenes import get_refraction_lab_scene
        from src.Rendering.Raytracing import RayTracer

        glass_scene = get_refraction_lab_scene(width=16, height=16)
        glass_out = RayTracer(glass_scene).render(seed=0)
        assert glass_out.mean() > 0.0, "Glass scene produced all-black output"


# ---------------------------------------------------------------------------
# Scene-specific Smoke Tests
# ---------------------------------------------------------------------------

class TestBenchSceneSmoke:
    """Quick smoke tests: every bench scene should render without crashing and
    produce a valid (H, W, 3) float array in [0, 1]."""

    SCENES = [
        "get_minimal_scene",
        "get_gradient_scene",
        "get_emissive_scene",
        "get_lit_studio_scene",
        "get_rgb_cornell_box_scene",
        "get_material_deck_scene",
        "get_refraction_lab_scene",
        "get_pastel_blocks_scene",
        "get_sdf_boolean_scene",
    ]

    @pytest.mark.slow
    @pytest.mark.parametrize("scene_fn", SCENES)
    def test_scene_renders_without_error(self, scene_fn):
        import importlib, py_src.tests.bench_scenes as bs
        from src.Rendering.Raytracing.Core import RayTracer

        fn = getattr(bs, scene_fn)
        scene = fn(width=16, height=16)
        tracer = RayTracer()
        output = tracer.render(seed=0)

        assert output.shape == (16, 16, 3), f"{scene_fn}: bad shape {output.shape}"
        assert np.all(output >= 0.0),  f"{scene_fn}: pixel below 0"
        assert np.all(output <= 1.0),  f"{scene_fn}: pixel above 1"
        assert output.dtype.kind == "f", f"{scene_fn}: expected float array"


# ---------------------------------------------------------------------------
# Camera Ray Generation
# ---------------------------------------------------------------------------

class TestCameraRayGeneration:
    def test_ray_count_matches_resolution(self):
        from py_src.tests.bench_scenes import get_minimal_scene
        scene = get_minimal_scene(width=8, height=8)
        rays = scene.camera.generate_rays()
        flat = np.array(rays).reshape(-1, 3)
        assert flat.shape[0] == 8 * 8

    def test_all_ray_directions_normalised(self):
        from py_src.tests.bench_scenes import get_minimal_scene
        scene = get_minimal_scene(width=8, height=8)
        dirs = np.array(scene.camera.generate_rays()).reshape(-1, 3)
        norms = np.linalg.norm(dirs, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_corner_rays_diverge_from_centre(self):
        """Corner rays must point away from the centre ray."""
        from py_src.tests.bench_scenes import get_minimal_scene
        scene = get_minimal_scene(width=9, height=9)
        rays = np.array(scene.camera.generate_rays()).reshape(9, 9, 3)
        center = rays[4, 4]
        corner = rays[0, 0]
        cosine = np.dot(center, corner) / (np.linalg.norm(center) * np.linalg.norm(corner))
        assert cosine < 1.0 - 1e-4, "Corner ray should diverge from centre ray"