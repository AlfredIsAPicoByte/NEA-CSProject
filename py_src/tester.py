from src import Geometry, Reflections, Refractions, Sampler, Camera, Luminance, PrimaryStructures, Projection
from src.Algorithims import Raytracing
import numpy as np

import inspect
from typing import Callable, Tuple, Any

# --- Custom Text Definitions ---
# Define these globally or pass them to run_tests if you want them customizable per run.
# For simplicity, we'll keep them here.
PASS_TEXT = ("✅", "SUCCESS")
FAIL_TEXT = ("❌" ,"FAILED")
ERROR_TEXT = ("🚫" ,"CRITCAL ERROR")
UNKNOWN_TEXT = ("⁉️", "UNKNOWN")

def return_message_handler(test_logic: Callable[[], Any], title: str, expected: bool = True) -> Tuple[bool, str]:
    """
    A handler for any test defined with a function retuning true and filse alongside custom responsed for expected failues

    :param test_logic: A function containing the actual test assertions/code.
    :param title: A brief description of what the test is checking.
    :param expected_result: True if the test is expected to pass, False if it's expected to fail.
    :return: A tuple of (result, message). Raises Exception on error.
    """
    try:
        out: Tuple[bool, str|None] = test_logic()

        # Normalize outputs:
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
            result_bool, details = out
        else:
            # Treat None as False; otherwise coerce to bool and use string detail if not purely boolean
            if out is None:
                result_bool = False
                details = ""
            else:
                result_bool = out[0]
                details = out[1] | ""

        if result_bool == expected:
            msg = f"{PASS_TEXT[0]} [{title}]"
            if details:
                msg += f" - {details}."
            msg += f" {PASS_TEXT[1]}"
            return True, msg

        else:
            msg = f"{FAIL_TEXT[0]} [{title}]"
            if details:
                msg += f" - {details}"
            msg += f" {FAIL_TEXT[1]}"
            return False, msg

    except Exception as e:
        raise Exception(f"{e.__class__.__name__}: {e}")
"""
Geometry, Primary: "Circle and Ray no intersection" no intersections expected.
Geometry, Primary: "Circle and Ray find intersection with the tangent" one intersection expected.
Geometry, Primary: "Circle and Ray find 2 intersections" two intersections expected.
Geometry, Primary: "Circle and Ray find intersection from inside circle" one intersection expected.
Geometry, Primary: "Circle and Ray find intersection from edge of circle" one intersection expected.
Geometry, Primary: "Circle and Ray find intersection from inside in reverse orientation " one intersection expected.
Geometry: "Circle check Point normal and tangent" expect point on circle.
Geometry: "Circle and Point is not inside" expect point not on circle.
Geometry: "Circle check Point has no normal"  Correctly raised error for GetNormal with point not on circle - {e}.
Geometry: "Circle check Point has no tangent" Correctly raised error for GetTangent with point not on circle - {e}.
Geometry, Primary: "Triangle and Ray get intersections" Expected intersections.
Geometry, Primary: "Triangle and Ray get no intersections" Expected no intersection.
Geometry, Primary: "Degenerate triangle" raised exception as expected - {e}.
Geometry, Primary: "Triangle and Ray no intersection and parallel to plane" no intersection as expected.
Geometry: "Triangle and Point on edge" Expexted point on edge.
Geometry: "Triangle and Point no normal" Correctly raised error for GetNormal with point not on triangle - {e}.
Geometry: "Triangle and Point no tangent" Correctly raised error for GetTangent with point not on triangle - {e}.
Luminance, PrimaryStructures: "Luminance initialization" Color, LightRay, and Material tests successful.
Luminance: "Color Arithmetic" no exeptions expected.
Luminance: "Material Reflection" no exeptions expected.



"""

def test_circle_ray_no_intersection():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([10, 0, 0]), np.array([1, 0, 0]))
        if not circle.CheckIntersection(ray):
            return True
        else:
            intersections = circle.GetIntersection(ray)
            if intersections is None or len(intersections) == 0:
                return True
            return False
    except Exception as e:
        return False
def test_circle_ray_tangent():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([5, -10, 0]), np.array([0, 1, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            if len(intersections) == 1:
                return True
            return False
        else:
            return False
    except Exception as e:
        return False
def test_circle_ray_two_intersections():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([-10, 0, 0]), np.array([1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            if len(intersections) == 2:
                return True
            return False
        else:
            return False
    except Exception as e:
        return False
def test_circle_ray_origin_inside():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([0, 0, 0]), np.array([1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            # For a ray starting at the center, only the forward intersection is returned
            assert len(intersections) == 1
            return True
        else:
            return False
    except Exception as e:
        return False
def test_circle_ray_origin_on_edge():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([5, 0, 0]), np.array([1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            # Should be one intersection (tangent at start)
            assert len(intersections) >= 1
            return True
        else:
            return False
    except Exception as e:
        return False
def test_circle_ray_reverse_orientation ():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([0, 0, 0]), np.array([-1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            assert len(intersections) == 1
            return True
        else:
            return False
    except Exception as e:
        return False
def test_circle_point_normal_tangent():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        point_on_circle = np.array([5, 0, 0])
        if circle.CheckPoint(point_on_circle, 0.01):
            normal = circle.GetNormal(point_on_circle)
            tangent = circle.GetTangent(point_on_circle)
            assert np.allclose(normal, np.array([1, 0, 0]))  # Normal at (5, 0) should be (1, 0)
            assert np.allclose(tangent, np.array([0, 1, 0]))  # Tangent at (5, 0) should be (0, 1)
            return True
        else:
            return False
    except Exception as e:
        return False
def test_circle_point_not_on_circle():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        point_not_on_circle = np.array([6, 0, 0])
        if not circle.CheckPoint(point_not_on_circle, 0.01):
            return True
        else:
            return False
    except Exception as e:
        return False
def test_circle_tangent_error():
    errors = ""
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        point = np.array([10, 0, 0])  # Not on circle
        _ = circle.GetNormal(point)
        return False
    except Exception as e:
        errors = e
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        point = np.array([10, 0, 0])  # Not on circle
        _ = circle.GetTangent(point)
        return False
    except Exception as e:
        errors += f", {e}"
        return True, errors
def test_triangle_ray_intersection():
    triangle = Geometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([1, 1, -10]), np.array([0, 0, 1]))

    if triangle.CheckIntersection(ray):
        intersection = triangle.GetIntersection(ray)
        if intersection is not None:
            return True
        return False
    else:
        return False
def test_triangle_ray_no_intersection():
    triangle = Geometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([6, 6, -10]), np.array([0, 0, 1]))

    if not triangle.CheckIntersection(ray):
        return True
    else:
        intersection = triangle.GetIntersection(ray)
        if intersection is None:
            return True
        return False
def test_triangle_degenerate_collinear():
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    v3 = np.array([2, 2, 2])  # Collinear
    try:
        triangle = Geometry.Triangle(v1, v2, v3)
        return False
    except Exception as e:
        return True, e
def test_triangle_ray_parallel():
    triangle = Geometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([0, 0, 1]), np.array([1, 1, 0]))  # Parallel to triangle plane

    if not triangle.CheckIntersection(ray):
        return True
    else:
        return False
def test_triangle_point_on_edge():
    triangle = Geometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point_on_edge = np.array([2.5, 0, 0])

    if triangle.CheckPoint(point_on_edge, 0.01):
        return True
    else:
        return False
def test_triangle_normal_error():
    triangle = Geometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point = np.array([10, 10, 10])  # Not on triangle

    try:
        _ = triangle.GetNormal(point)
        return False
    except Exception as e:
        return True
def test_triangle_tangent_error():
    triangle = Geometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point = np.array([10, 10, 10])  # Not on triangle

    try:
        _ = triangle.GetTangent(point)
        return False
    except Exception as e:
        return True
def test_Luminance_basic():
    try:
        # Test Color class
        color1 = Luminance.ColorData(0.5, 0.2, 0.7)
        color2 = Luminance.ColorData(0.1, 0.2, 0.3, 0.5)
        assert color1.red == 0.5 and color2.alpha == 0.5

        color_sum = color1 + color2
        color_mul = color1 * 0.5
        color_eq = (color1 == Luminance.ColorData(0.5, 0.2, 0.7))
        assert isinstance(color_sum, Luminance.ColorData)
        assert isinstance(color_mul, Luminance.ColorData)
        assert color_eq is True

        # Test LightRay
        ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([1,0,0]))
        lray = Luminance.LightRay(ray.origin, ray.orientation , color1, intensity=2.0)
        assert isinstance(lray, Luminance.LightRay)

        # Test Material reflection
        mat = Luminance.Material(color1, roughness=0.2, glossiness=0.8)
        reflected = mat.RedirectLightRay(lray, np.array([0,1,0]))
        reflected_ray, refected_color = reflected.ray, reflected.base_color
        assert isinstance(reflected_ray, PrimaryStructures.Ray)
        assert isinstance(refected_color, Luminance.ColorData)

        return True
    except Exception as e:
        return False, e
def test_color_arithmetic():
    try:
        c1 = Luminance.ColorData(0.2, 0.3, 0.4)
        c2 = Luminance.ColorData(0.1, 0.1, 0.1)
        c3 = c1 + c2
        c4 = c1 - c2
        c5 = c1 * 2
        c6 = c1 / 2
        assert isinstance(c3, Luminance.ColorData)
        assert isinstance(c4, Luminance.ColorData)
        assert isinstance(c5, Luminance.ColorData)
        assert isinstance(c6, Luminance.ColorData)
    
        return True
    except Exception as e:
        return False, e
def test_material_reflect():
    try:
        color = Luminance.ColorData(0.5, 0.5, 0.5)
        mat = Luminance.Material(color, 0.1, 0.9)
        ray = PrimaryStructures.Ray(np.array([1, 2, 3]), np.array([0, 1, 0]))
        normal = np.array([0, 1, 0])
        reflected = mat.RedirectLightRay(Luminance.LightRay(ray.origin, ray.orientation , Luminance.ColorData()), normal)
        assert isinstance(reflected, PrimaryStructures.Ray)

        return True
    except Exception as e:
        return False
    
def test_reflections_basic():
    try:
        # Test angle calculations
        angle = 30.0
        refl_angle = Reflections.calculate_reflection_angle(angle)
        assert refl_angle == angle

        inc_angle = Reflections.calculate_incident_angle(10, 40)
        assert inc_angle == 30

        # Test reflect_ray
        ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([1,1,0]))
        normal = np.array([0,1,0])
        reflected = Reflections.reflect_ray(normal, ray)
        assert isinstance(reflected, PrimaryStructures.Ray)

        # Test error on shape mismatch
        try:
            Reflections.reflect_ray(np.array([0,1]), ray)
            print('Reflections: "Reflection calculation gives error" failed to raise error on shape mismatch. *Failed*')
            return False
        except Exception:
            print('Reflections: "Reflection calculation gives error" angle, reflect_ray, and error handling tests successful. *Passed*')
            return True
        
    except Exception as e:
        print(f'Reflections: "Reflection calculation gives error" failed - {e}. *Failed*')
        return False

def test_refractions_basic():
    try:
        # Test refractive index conversions
        idx = Refractions.convert_speed_to_index(2e8)
        speed = Refractions.convert_index_to_speed(1.5)
        assert abs(idx - 1.5) < 0.01
        assert abs(speed - 2e8) < 1e6

        # Test angle calculations
        angle_refr = Refractions.calculate_angle_of_refraction(30, 1.5, 1.0)
        angle_inc = Refractions.calculate_angle_of_incidence(19.47, 1.5, 1.0)
        idx2 = Refractions.calculate_refractive_index(30, angle_refr, 1.5)
        idx1 = Refractions.calculate_refractive_index_incident(30, angle_refr, 1.0)
        crit_angle = Refractions.calculate_critical_angle(1.5, 1.0)
        assert abs(angle_refr - 48.59) < 0.1
        assert abs(angle_inc - 30) < 0.1
        assert abs(idx2 - 1.0) < 0.01
        assert abs(idx1 - 1.5) < 0.01
        assert abs(crit_angle - 41.81) < 0.1

        angle = Refractions.calculate_angle_of_refraction(60, 1.5, 1.0)
        assert angle < crit_angle

        # Test refract_ray
        ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([1,0,0]))
        normal = np.array([0,1,0])
        try:
            Refractions.refract_ray(normal, ray, 1.5, 1.0)
            print('Refractions: "Refraction calculation gives error" failed to raise error for total internal reflection in refract_ray. *Failed*')
            return False
        except Exception:
            print('Refractions: "Refraction calculation gives error" index, angle, critical angle, and error handling tests successful. *Passed*')
            return True
    except Exception as e:
        print(f'Refractions: "Refraction calculation gives error" failed - {e}. *Failed*')
        return False

def test_sampler_basic():
    try:
        samp = Sampler.SamplingManager(Sampler.SampleSettings(), Sampler.RenderSettings(800, 600, 4))
        if hasattr(samp, 'sample'):
            samp.GenerateSamples()
            _ = samp.GetSamples(1, 0)
        print('Sampler: "Sampler initialization" no exeptions expected. *Passed*')
        return True
    except Exception as e:
        print(f'Sampler: failed - {e}. *Failed*')
        return False

def test_projections_basic():
    try:
        cam = Camera(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 1.77, 0.1, 1000, 1000)
        proj = Projection.Scene(cam)
        if hasattr(proj, 'project'):
            _ = proj.AddObject() # TODO: Add thing
        print('Projections: "Initialization" no exeptions expected. *Passed*')
        return True
    except Exception as e:
        print(f'Projections: failed - {e}. *Failed*')
        return False

def test_camera_basic():
    try:
        cam = Camera.CameraObject(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 1.77, 0.1, 1000, 1000)
        if hasattr(cam, 'get_view_matrix'):
            _ = cam.get_view_matrix()
        print('Camera: "Initialization" no exeptions expected. *Passed*')
        return True
    except Exception as e:
        print(f'Camera: failed - {e}. *Failed*')
        return False

def test_raycasting_basic():
    try:
        rc = Raytracing.Raytracer()
        if hasattr(rc, 'cast'):
            _ = rc.cast(np.array([0,0,0]), np.array([1,0,0]))
        print('Raycasting: "Initialization" no exeptions expected. *Passed*')
        return True
    except Exception as e:
        print('Raycasting: failed - {e}. *Failed*')
        return False

def test_circle_negative_radius():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), -5)
        print('Geometry: "Negative Circle radius gives error" Circle with negative radius should raise an error. *Failed*')
        return False
    except Exception as e:
        print(f'Geometry: "Negative Circle radius gives error" Circle with negative radius correctly raised an error - {e}. *Passed*')
        return True

def test_triangle_area_zero():
    v1 = np.array([0, 0, 0])
    v2 = np.array([0, 0, 0])
    v3 = np.array([0, 0, 0])
    try:
        triangle = Geometry.Triangle(v1, v2, v3)
        print('Geometry: "Triangle near-zero area gives error" Triangle with zero area should raise an error. *Failed*')
        return False
    except Exception as e:
        print(f'Geometry: "Triangle near-zero area gives error" Triangle with zero area correctly raised an error - {e}. *Passed*')
        return True

def test_color_clamping():
    color: Luminance.ColorData = Luminance.ColorData(0, 0, 0)
    color.rgba = np.array([1.5, -0.2, 0.5, 1.2])  # Intentionally out of bounds
    clamped = color.clamp()
    try:
        if 0.0 <= clamped.red <= 1.0 and 0.0 <= clamped.green <= 1.0 and 0.0 <= clamped.blue <= 1.0:
            print('Luminance: "Color Clamping" no exeptions epected. *Passed*')
            return True
        else:
            print('Luminance: "Color Clamping" failed to clamp the RGB channels to the ranges between 0 to 1. *Failed*')
            return False
    except Exception as e:
        print(f'Luminance: "Color Clamping" failed - {e}. *Failed*')
        return False

def test_camera_fov_limits():
    try:
        cam = Camera.CameraObject(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 0, 1.77, 0.1, 1000, 1000)
        print('Camera: "Camera with zero FOV gives error" expected an exeption. *Failed*')
        return False
    except Exception as e:
        print(f'Camera: "Camera with zero FOV gives error" succesfully raised an error - {e}. *Passed*')
        return True

def test_ray_normalization():
    ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([10,0,0]))
    norm = np.linalg.norm(ray.direction)
    try:
        if abs(norm - 1.0) < 1e-6:
            print('PrimaryStructures: "Ray direction normalization" works as expected. *Passed*')
            return True
        else:
            print('PrimaryStructures: "Ray direction normalization" failed to get the normalized version of the ray direction. *Failed*')
            return False
    except Exception as e:
        print(f'PrimaryStructures: "Ray direction normalization" failed - {e}. *Failed*')
        return False

def test_material_reflect_color_bounds():
    color = Luminance.ColorData(1.0, 1.0, 1.0)
    mat = Luminance.Material(color, 0.5, 0.5)
    reflected = mat.AffectColor(Luminance.ColorData(2.0, 2.0, 2.0))
    try:
        if all(0.0 <= c <= 1.0 for c in [reflected.red, reflected.green, reflected.blue]):
            print('Luminance: "Material reflection color bounds" no exeptions expected. *Passed*')
            return True
        else:
            print('Luminance: "Material eflection color bounds" failed to clamp color values. *Failed*')
            return False
    except Exception as e:
        print(f'Luminance: "Material eflection color bounds" failed - {e}. *Failed*')
        return False

available_tests = {
        ': "" .': test_circle_ray_no_intersection,
        ': "" .': test_circle_ray_tangent,
        ': "" .': test_circle_ray_two_intersections,
        ': "" .': test_circle_ray_origin_inside,
        ': "" .': test_circle_ray_origin_on_edge,
        ': "" .': test_circle_ray_reverse_orientation,
        ': "" .': test_circle_point_normal_tangent,
        ': "" .': test_circle_point_not_on_circle,
        ': "" .': test_circle_normal_error,
        ': "" .': test_circle_tangent_error,
        ': "" .': test_triangle_ray_intersection,
        ': "" .': test_triangle_ray_no_intersection,
        ': "" .': test_triangle_degenerate_collinear,
        ': "" .': test_triangle_ray_parallel,
        ': "" .': test_triangle_point_on_edge,
        ': "" .': test_triangle_normal_error,
        ': "" .': test_triangle_tangent_error,
        ': "" .': test_Luminance_basic,
        ': "" .': test_color_arithmetic,
        ': "" .': test_material_reflect,
        ': "" .': test_reflections_basic,
        ': "" .': test_refractions_basic,
        ': "" .': test_sampler_basic,
        ': "" .': test_projections_basic,
        ': "" .': test_camera_basic,
        ': "" .': test_raycasting_basic,
        ': "" .': test_circle_negative_radius,
        ': "" .': test_triangle_area_zero,
        ': "" .': test_color_clamping,
        ': "" .': test_camera_fov_limits,
        ': "" .': test_ray_normalization,
        ': "" .': test_material_reflect_color_bounds
}

def run_tests(ignore_passed: bool = False, ignore_fail: bool = False, ignore_error: bool = False):
    """
    Runs predefined tests, handling pass/fail/error and respecting ignore flags.
    """
    print("\n" + "="*50)
    print(f"----- Running tests (Ignore Passed: {ignore_passed}, Ignore Errors: {ignore_error}) -----")
    print("="*50)

    passed_count = 0
    total_tests = len(available_tests)

    for title, test_func in available_tests:
        result_message: str = f"{UNKNOWN_TEXT[0]} {UNKNOWN_TEXT[1]}"

        try:
            result, result_message = return_message_handler(test_func, title)

            if result:
                passed_count += 1
                if not ignore_passed:
                    print(result_message)
            else:
                if not ignore_fail:
                    print(result_message)

        except Exception as e:
            if not ignore_error:
                print(str(e))
                continue
                    
    print("\n" + "="*50)
    print(f"----- Tests completed: {passed_count}/{total_tests} passed -----")
    print("="*50)

if __name__ == "__main__":
    run_tests(True)