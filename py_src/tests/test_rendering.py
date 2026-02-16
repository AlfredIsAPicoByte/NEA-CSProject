import pytest
import numpy as np

from src.Data.Ray import Ray
from src.Data.Color import Color
from src.Data.Transform import Transform
from src.Data.Sampling.Core import Sampler
from src.Geometry.SDF import Sphere, Cube, Cylinder
from src.Material.Factory import MaterialFactory
from src.Utilities.Common import unit

IMG_OUTPUT_DIR = "images/testing"

def _march_sphere(ray: Ray, sphere: Sphere, max_steps: int = 128, tol: float = 1e-4):
    """Minimal sphere-tracing loop used to validate ray-SDF intersection logic."""
    t = 0.0
    for _ in range(max_steps):
        p = ray.origin + t * ray.direction
        d = sphere.get_distance(p)
        if d < tol:
            return t, p
        t += d
        if t > 1000.0:
            break
    return None, None

class TestRay:
    def test_ray_direction_normalised(self):
        ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 4.0]))
        assert np.isclose(np.linalg.norm(ray.direction), 1.0, atol=1e-6)

    def test_ray_at_t_zero_is_origin(self):
        ray = Ray(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0]))
        assert np.allclose(ray.point_at(0.0), ray.origin)

    def test_ray_at_t_positive(self):
        ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        assert np.allclose(ray.point_at(5.0), np.array([0.0, 0.0, 5.0]))

    def test_ray_at_negative_t(self):
        ray = Ray(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]))
        result = ray.point_at(-3.0)
        assert np.allclose(result, np.array([0.0, 0.0, 2.0]))

    def test_ray_stores_origin(self):
        origin = np.array([1.0, -2.0, 3.0])
        ray = Ray(origin, np.array([0.0, 1.0, 0.0]))
        assert np.allclose(ray.origin, origin)

class TestRaySphereIntersection:
    def test_direct_hit_returns_positive_t(self):
        """Centre ray aimed at a unit sphere at the origin should register a hit."""
        ray = Ray(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]))
        sphere = Sphere(1.0)
        t, p = _march_sphere(ray, sphere)
        assert t is not None and t > 0.0

    def test_direct_hit_point_on_surface(self):
        """The hit point should lie on the sphere surface (|p| ≈ radius)."""
        ray = Ray(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]))
        sphere = Sphere(1.0)
        t, p = _march_sphere(ray, sphere)
        assert p is not None
        assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-3)

    def test_hit_from_side(self):
        """A ray travelling along +X aimed at origin should also register a hit."""
        ray = Ray(np.array([-5.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        sphere = Sphere(1.0)
        t, p = _march_sphere(ray, sphere)
        assert t is not None and t > 0.0

    def test_miss_returns_none(self):
        """A ray offset clearly outside the sphere's cross-section should miss."""
        ray = Ray(np.array([0.0, 5.0, -5.0]), np.array([0.0, 0.0, 1.0]))
        sphere = Sphere(1.0)
        t, p = _march_sphere(ray, sphere)
        assert t is None

    def test_ray_inside_sphere_hits_from_inside(self):
        """A ray originating inside the sphere should find a forward hit."""
        ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        sphere = Sphere(2.0)
        t, p = _march_sphere(ray, sphere)
        assert t is not None and t > 0.0

    def test_hit_point_matches_t_value(self):
        """Hit point reconstructed from t should match the marched position."""
        ray = Ray(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]))
        sphere = Sphere(1.0)
        t, p = _march_sphere(ray, sphere)
        assert t is not None
        assert np.allclose(ray.point_at(t), p, atol=1e-3)

    def test_larger_sphere_hit_at_greater_t(self):
        """A bigger sphere blocks the ray sooner when facing the same ray origin."""
        ray = Ray(np.array([0.0, 0.0, -10.0]), np.array([0.0, 0.0, 1.0]))
        t_small, _ = _march_sphere(ray, Sphere(1.0))
        t_large, _ = _march_sphere(ray, Sphere(3.0))
        assert t_large is not None and t_small is not None
        assert t_large < t_small  # larger sphere reached sooner from behind

class TestSurfaceNormals:
    def _numerical_normal(self, sdf, p, eps=1e-4):
        grad = np.array([
            sdf.get_distance(p + np.array([eps, 0, 0])) - sdf.get_distance(p - np.array([eps, 0, 0])),
            sdf.get_distance(p + np.array([0, eps, 0])) - sdf.get_distance(p - np.array([0, eps, 0])),
            sdf.get_distance(p + np.array([0, 0, eps])) - sdf.get_distance(p - np.array([0, 0, eps])),
        ])
        return grad / (np.linalg.norm(grad) + 1e-12)

    def test_sphere_normal_at_x_axis(self):
        """Normal at (1, 0, 0) on a unit sphere should be ~(1, 0, 0)."""
        sphere = Sphere(1.0)
        n = self._numerical_normal(sphere, np.array([1.0, 0.0, 0.0]))
        assert np.allclose(n, np.array([1.0, 0.0, 0.0]), atol=1e-3)

    def test_sphere_normal_at_y_axis(self):
        sphere = Sphere(1.0)
        n = self._numerical_normal(sphere, np.array([0.0, 1.0, 0.0]))
        assert np.allclose(n, np.array([0.0, 1.0, 0.0]), atol=1e-3)

    def test_sphere_normal_isunit_length(self):
        sphere = Sphere(1.0)
        n = self._numerical_normal(sphere, np.array([1.0, 0.0, 0.0]))
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-5)

    def test_cube_face_normal_along_x(self):
        """Normal at the +X face centre of a unit cube should be ~(1, 0, 0)."""
        cube = Cube(1.0)
        n = self._numerical_normal(cube, np.array([1.0, 0.0, 0.0]))
        assert np.allclose(n, np.array([1.0, 0.0, 0.0]), atol=1e-2)

    @pytest.mark.parametrize("sdf, point, expected_normal", [
        (Sphere(1.0), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (Sphere(1.0), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (Cube(1.0), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (Cube(1.0), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (Cylinder(1.0, 2.0), np.array([1.0, 2e-4, 2e-4]), np.array([1., 2e-4/np.sqrt(2e-8), 2e-4/np.sqrt(2e-8)])),
        (Cylinder(1.0, 2.0), np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    ])
    def test_numerical_normal(self, sdf, point, expected_normal):
        """Numerical normal should approximate the expected normal for simple shapes."""
        n = self._numerical_normal(sdf, point)
        assert np.allclose(n, expected_normal, atol=1e-1), f"Normal {n} differs from expected {expected_normal}"  

class TestDiffuseShading:
    def _lambertian(self, normal, light_dir, albedo):
        """Simple Lambertian shading: albedo * max(dot(N, L), 0)."""
        return albedo * max(float(np.dot(normal, light_dir)), 0.0)

    def test_facing_light_is_brighter(self):
        normal    = unit(np.array([0.0, 1.0, 0.0]))
        light_dir = unit(np.array([0.0, 1.0, 0.0]))  # directly above
        shade = self._lambertian(normal, light_dir, 1.0)
        assert np.isclose(shade, 1.0, atol=1e-6)

    def test_grazing_angle_is_dimmer(self):
        normal     = unit(np.array([0.0, 1.0, 0.0]))
        grazing    = unit(np.array([1.0, 0.01, 0.0]))
        shade = self._lambertian(normal, grazing, 1.0)
        assert shade < 0.1

    def test_back_facing_is_zero(self):
        normal    = unit(np.array([0.0, 1.0, 0.0]))
        behind    = unit(np.array([0.0, -1.0, 0.0]))
        shade = self._lambertian(normal, behind, 1.0)
        assert np.isclose(shade, 0.0, atol=1e-6)

    def test_albedo_scales_output(self):
        normal    = unit(np.array([0.0, 1.0, 0.0]))
        light_dir = unit(np.array([0.0, 1.0, 0.0]))
        assert np.isclose(
            self._lambertian(normal, light_dir, 0.5),
            0.5 * self._lambertian(normal, light_dir, 1.0),
            atol=1e-6
        )

    def test_45_degree_light(self):
        """Light at 45° should give cos(45°) ≈ 0.707."""
        normal    = unit(np.array([0.0, 1.0, 0.0]))
        light_dir = unit(np.array([1.0, 1.0, 0.0]))
        shade = self._lambertian(normal, light_dir, 1.0)
        assert np.isclose(shade, np.cos(np.deg2rad(45)), atol=1e-5)

class TestSpecularShading:
    def _blinn_phong(self, normal, view_dir, light_dir, shininess):
        half = unit(view_dir + light_dir)
        return max(float(np.dot(normal, half)), 0.0) ** shininess

    def test_on_lobe_is_maximum(self):
        normal = unit(np.array([0.0, 1.0, 0.0]))
        view   = unit(np.array([0.0, 1.0, 0.0]))
        light  = unit(np.array([0.0, 1.0, 0.0]))
        val = self._blinn_phong(normal, view, light, shininess=32)
        assert np.isclose(val, 1.0, atol=1e-5)

    def test_higher_shininess_narrower_peak(self):
        normal    = unit(np.array([0.0, 1.0, 0.0]))
        view      = unit(np.array([0.0, 1.0, 0.0]))
        off_lobe  = unit(np.array([1.0, 1.0, 0.0]))
        low_shin  = self._blinn_phong(normal, view, off_lobe, shininess=4)
        high_shin = self._blinn_phong(normal, view, off_lobe, shininess=128)
        assert low_shin >= high_shin

    def test_specular_zero_when_light_behind(self):
        normal    = unit(np.array([0.0, 1.0, 0.0]))
        view      = unit(np.array([0.0, 1.0, 0.0]))
        behind    = unit(np.array([0.0, -1.0, 0.0]))
        val = max(float(np.dot(normal, unit(view + behind))), 0.0) ** 32
        assert val < 0.1

class TestRenderingPipeline:
    @pytest.mark.slow
    def test_render_output_shape(self):
        """Rendered output should have the expected pixel dimensions."""
        from py_src.tests.bench_scenes import get_minimal_scene
        from src.Rendering.RayTracing.Core import RayTracer, RayTracingSettings

        scene = get_minimal_scene(width=16, height=16)
        tracer = RayTracer(RayTracingSettings(16, 16))
        tracer.generate_film(scene)
        output = tracer.settings.film.get_image()

        assert output.shape == (16, 16, 3), f"Unexpected shape {output.shape}"

    @pytest.mark.slow
    def test_render_values_in_range(self):
        """All pixel values should be in [0, 1]."""
        from py_src.tests.bench_scenes import get_minimal_scene
        from src.Rendering.RayTracing.Core import RayTracer, RayTracingSettings

        scene = get_minimal_scene(width=16, height=16)
        tracer = RayTracer(RayTracingSettings(16, 16))
        tracer.generate_film(scene)
        output = tracer.settings.film.get_image()

        assert np.all(output >= 0.0), "Pixel value below 0"
        assert np.all(output <= 1.0), "Pixel value above 1"

    @pytest.mark.slow
    def test_render_has_non_background_pixels(self):
        """A sphere in front of the camera should colour at least some pixels."""
        from py_src.tests.bench_scenes import get_minimal_scene
        from src.Rendering.RayTracing.Core import RayTracer, RayTracingSettings

        scene = get_minimal_scene(width=32, height=32)
        tracer = RayTracer(RayTracingSettings(32, 32))
        tracer.generate_film(scene)
        output = tracer.settings.film.get_image()

        bg = np.array([0x9B, 0xB0, 0xCA]) / 255.0  # scene background
        # At least 10 % of pixels must differ from the background
        diff = np.any(np.abs(output - bg) > 0.05, axis=-1)
        assert diff.sum() > 0.10 * 32 * 32

    @pytest.mark.slow
    def test_render_is_deterministic(self):
        """Two renders of the same scene with the same seed must be identical."""
        from py_src.tests.bench_scenes import get_minimal_scene
        from src.Rendering.RayTracing.Core import RayTracer, RayTracingSettings

        scene = get_minimal_scene(width=16, height=16)
        tracer = RayTracer(RayTracingSettings(16, 16))
        tracer.generate_film(scene, Sampler(seed=0))
        out1 = tracer.settings.film.get_image()
        tracer.generate_film(scene, Sampler(seed=0))
        out2 = tracer.settings.film.get_image()

        assert np.allclose(out1, out2), "Non-deterministic output for identical seeds"

    @pytest.mark.slow
    def test_emissive_object_brightens_scene(self):
        """Adding an emissive object should increase average luminance."""
        from py_src.tests.bench_scenes import get_minimal_scene
        from src.Rendering.RayTracing import RayTracer, RayTracingSettings
        from src.Data.Scene import Scene
        from src.Data.Context import SDF_Material
        from src.Geometry.SDF import Sphere as _Sphere

        base_scene = get_minimal_scene(width=16, height=16)
        tracer = RayTracer(RayTracingSettings(16, 16))
        tracer.generate_film(base_scene, Sampler(seed=0))
        base_output = tracer.settings.film.get_image()

        emissive_scene = get_minimal_scene(width=16, height=16)
        mat_glow = MaterialFactory.create_emissive(Color(1, 1, 1), 10.0)
        emissive_scene.add_object_by_context(
            SDF_Material(_Sphere(0.5), mat_glow), "Glow",
            Transform(np.array([0.0, 3.0, 0.0]))
        )
        tracer.generate_film(emissive_scene, Sampler(seed=0))
        emissive_output = tracer.settings.film.get_image()

        assert emissive_output.mean() >= base_output.mean() - 1e-4