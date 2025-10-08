from src import Gemometry, Reflections, Refractions, Sampler, Camera, Luminance, PrimaryStructures, Projection
from src.Algorithims import Raytracing
import numpy as np

def run_tests():
    print("-----Running tests-----")
    count = 0
    tests = [
        # Geometry: Circle
        test_circle_ray_no_intersection(),
        test_circle_ray_tangent(),
        test_circle_ray_two_intersections(),
        test_circle_ray_origin_inside(),
        test_circle_ray_origin_on_edge(),
        test_circle_ray_reverse_direction(),
        test_circle_point_normal_tangent(),
        test_circle_point_not_on_circle(),
        test_circle_normal_error(),
        test_circle_tangent_error(),

        # Geometry: Triangle
        test_triangle_ray_intersection(),
        test_triangle_ray_no_intersection(),
        test_triangle_degenerate_collinear(),
        test_triangle_ray_parallel(),
        test_triangle_point_on_edge(),
        test_triangle_normal_error(),
        test_triangle_tangent_error(),

        # Luminance/Color/Material
        test_Luminance_basic(),
        test_color_arithmetic(),
        test_material_reflect(),

        # Reflections/Refractions
        test_reflections_basic(),
        test_refractions_basic(),

        # Other modules
        test_sampler_basic(),
        test_projections_basic(),
        test_camera_basic(),
        test_raycasting_basic(),

        # Additional tests
        test_circle_negative_radius(),
        test_triangle_area_zero(),
        test_color_clamping(),
        test_camera_fov_limits(),
        test_ray_normalization(),
        test_material_reflect_color_bounds()
    ]

    for test in tests:
        if test:
            count += 1
    
    print(f"-----Tests completed: {count}/{len(tests)} passed-----")
    

def test_circle_ray_no_intersection():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([10, 0, 0]), np.array([1, 0, 0]))
        if not circle.CheckIntersection(ray):
            print('Gemometry, Primary: "Circle and Ray no intersection" Expected no intersections. *Passed*')
            return True
        else:
            intersections = circle.GetIntersection(ray)
            if intersections is None or len(intersections) == 0:
                print('Gemometry, Primary: "Circle and Ray no intersection" Expected no intersections. *Passed*')
                return True
            print('Gemometry, Primary: "Circle and Ray no intersection" Found an intersection. *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry, Primary: "Circle and Ray no intersection" General error: {e} *Failed*')
        return False

def test_circle_ray_tangent():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([5, -10, 0]), np.array([0, 1, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            if len(intersections) == 1:
                print('Gemometry, Primary: "Circle and Ray find intersection with the tangent" Expect 1 intersection (tangent). *Passed*')
                return True
            print('Gemometry, Primary: "Circle and Ray find intersection with the tangent" Expect 1 intersection (tangent). *Failed*')
            return False
        else:
            print('Gemometry, Primary: "Circle and Ray find intersection with the tangent" Expect 1 intersection (tangent). *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry, Primary: "Circle and Ray tangent" General error: {e} *Failed*')
        return False

def test_circle_ray_two_intersections():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([-10, 0, 0]), np.array([1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            if len(intersections) == 2:
                print('Gemometry, Primary: "Circle and Ray find 2 intersection" Expect 2 intersections. *Passed*')
                return True
            print('Gemometry, Primary: "Circle and Ray find 2 intersection" Expect 2 intersections. *Failed*')
            return False
        else:
            print('Gemometry, Primary: "Circle and Ray find 2 intersection" Expect 2 intersections. *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry, Primary: "Circle and Ray two intersections" General error: {e} *Failed*')
        return False

def test_circle_ray_origin_inside():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([0, 0, 0]), np.array([1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            # For a ray starting at the center, only the forward intersection is returned
            assert len(intersections) == 1
            print('Gemometry, Primary: "Circle and Ray find intersection from inside circle" one intersection as expected. *Passed*')
            return True
        else:
            print('Gemometry, Primary: "Circle and Ray find intersection from inside circle" No intersection detected for ray from inside circle. *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry, Primary: "Circle and Ray origin inside" General error: {e} *Failed*')
        return False

def test_circle_ray_origin_on_edge():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([5, 0, 0]), np.array([1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            # Should be one intersection (tangent at start)
            assert len(intersections) >= 1
            print('Gemometry, Primary: "Circle and Ray find intersection from edge of circle" intersection detected as expected. *Passed*')
            return True
        else:
            print('Gemometry, Primary: "Circle and Ray find intersection from edge of circle" No intersection detected for ray from edge of circle. *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry, Primary: "Circle and Ray origin on edge" General error: {e} *Failed*')
        return False

def test_circle_ray_reverse_direction():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        ray = PrimaryStructures.Ray(np.array([0, 0, 0]), np.array([-1, 0, 0]))
        if circle.CheckIntersection(ray):
            intersections = circle.GetIntersection(ray)
            assert len(intersections) == 1
            print('Gemometry, Primary: "Circle and Ray find intersection from inside in reverse direction" one intersection as expected. *Passed*')
            return True
        else:
            print('Gemometry, Primary: "Circle and Ray find intersection from inside in reverse direction" No intersection detected for reverse direction ray from center. *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry, Primary: "Circle and Ray reverse direction" General error: {e} *Failed*')
        return False

def test_circle_point_normal_tangent():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        point_on_circle = np.array([5, 0, 0])
        if circle.CheckPoint(point_on_circle, 0.01):
            normal = circle.GetNormal(point_on_circle)
            tangent = circle.GetTangent(point_on_circle)
            assert np.allclose(normal, np.array([1, 0, 0]))
            assert np.allclose(tangent, np.array([0, 1, 0]))
            print('Gemometry: "Circle check Point normal and tangent" Expect point on circle. *Passed*')
            return True
        else:
            print('Gemometry: "Circle check Point normal and tangent" . *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry: "Circle check Point normal and tangent" General error: {e} *Failed*')
        return False

def test_circle_point_not_on_circle():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        point_not_on_circle = np.array([6, 0, 0])
        if not circle.CheckPoint(point_not_on_circle, 0.01):
            print('Gemometry: "Circle and Point is not inside" Expect point not on circle. *Passed*')
            return True
        else:
            print('Gemometry: "Circle and Point is not inside" Expect point not on circle. *Failed*')
            return False
    except Exception as e:
        print(f'Gemometry: "Circle and Point not on circle" General error: {e} *Failed*')
        return False

def test_circle_normal_error():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        point = np.array([10, 0, 0])  # Not on circle
        _ = circle.GetNormal(point)
        print('Geometry: "Circle check Point has no normal" Expected error for GetNormal with point not on circle, but none raised. *Failed*')
        return False
    except Exception as e:
        print(f'Geometry: "Circle check Point has no normal"  Correctly raised error for GetNormal with point not on circle - {e}. *Passed*')
        return True
    
def test_circle_tangent_error():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
        point = np.array([10, 0, 0])  # Not on circle
        _ = circle.GetTangent(point)
        print('Geometry: "Circle check Point has no tangent" Expected error for GetTangent with point not on circle, but none raised. *Failed*')
        return False
    except Exception as e:
        print(f'Geometry: "Circle check Point has no tangent" Correctly raised error for GetTangent with point not on circle - {e}. *Passed*')
        return True

def test_triangle_ray_intersection():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([1, 1, -10]), np.array([0, 0, 1]))

    if triangle.CheckIntersection(ray):
        intersection = triangle.GetIntersection(ray)
        if intersection is not None:
            print('Gemometry, Primary: "Triangle and Ray get intersections" Expected intersections. *Passed*')
            return True
        print('Gemometry, Primary: "Triangle and Ray get intersections" No intersections gotten. *Failed*')
        return False
    else:
        print('Gemometry, Primary:  "Triangle and Ray get intersections" No intersections found. *Failed*')
        return False

def test_triangle_ray_no_intersection():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([6, 6, -10]), np.array([0, 0, 1]))

    if not triangle.CheckIntersection(ray):
        print('Gemometry, Primary: "Triangle and Ray get no intersections" Expected no intersection. *Passed*')
        return True
    else:
        intersection = triangle.GetIntersection(ray)
        if intersection is None:
            print('Gemometry, Primary: "Triangle and Ray get no intersections" Expected no intersection. *Passed*')
            return True
        print('Gemometry, Primary: "Triangle and Ray get no intersections" Found intersections. *Failed*')
        return False

def test_triangle_degenerate_collinear():
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    v3 = np.array([2, 2, 2])  # Collinear
    triangle = Gemometry.Triangle(v1, v2, v3)
    ray = PrimaryStructures.Ray(np.array([0, 0, -1]), np.array([0, 0, 1]))

    try:
        result = triangle.CheckIntersection(ray)
        print(f'Gemometry, Primary: "Degenerate triangle" no exeptions raised - {result}. *Failed*')
        return False
    except Exception as e:
        print(f'Gemometry, Primary: "Degenerate triangle" raised exception as expected - {e}. *Passed*')
        return True

def test_triangle_ray_parallel():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([0, 0, 1]), np.array([1, 1, 0]))  # Parallel to triangle plane

    if not triangle.CheckIntersection(ray):
        print('Gemometry, Primary: "Triangle and Ray no intersection and parallel to plane" no intersection as expected. *Passed*')
        return True
    else:
        print('Gemometry, Primary: "Triangle and Ray no intersection and parallel to plane" unexpected intersection for parallel ray. *Failed*')
        return False

def test_triangle_point_on_edge():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point_on_edge = np.array([2.5, 0, 0])

    if triangle.CheckPoint(point_on_edge, 0.01):
        print('Geometry: "Triangle and Point on edge" Expexted point on edge. *Passed*')
        return True
    else:
        print('Geometry: "Triangle and Point on edge" Point is not on edge. *Failed*')
        return False

def test_triangle_normal_error():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point = np.array([10, 10, 10])  # Not on triangle

    try:
        _ = triangle.GetNormal(point)
        print('Geometry: "Triangle and Point no normal" Expected error for GetNormal with point not on triangle, but none raised. *Failed*')
        return False
    except Exception as e:
        print(f'Geometry: "Triangle and Point no normal" Correctly raised error for GetNormal with point not on triangle - {e}. *Passed*')
        return True

def test_triangle_tangent_error():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point = np.array([10, 10, 10])  # Not on triangle

    try:
        _ = triangle.GetTangent(point)
        print('Geometry: "Triangle and Point no tangent" Expected error for GetTangent with point not on triangle, but none raised. *Failed*')
        return False
    except Exception as e:
        print(f'Geometry: "Triangle and Point no tangent" Correctly raised error for GetTangent with point not on triangle - {e}. *Passed*')
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
        lray = Luminance.LightRay(np.array([0,0,0]), np.array([1,0,0]), color1, intensity=2.0)
        assert isinstance(lray, Luminance.LightRay)

        # Test Material reflection
        mat = Luminance.Material(color1, roughness=0.2, glossiness=0.8)
        reflected = mat.RedirectLightRay(lray, np.array([0,1,0]))
        reflected_ray, refected_color = reflected.ray, reflected.base_color
        assert isinstance(reflected_ray, PrimaryStructures.Ray)
        assert isinstance(refected_color, Luminance.ColorData)

        print('Luminance, PrimaryStructures: Color, LightRay, and Material tests successful. *Passed*')
        return True
    except Exception as e:
        print(f'Luminance, PrimaryStructures: Failed - {e}. *Failed*')
        return False

def test_color_arithmetic():
    try:
        c1 = Luminance.Color(0.2, 0.3, 0.4)
        c2 = Luminance.Color(0.1, 0.1, 0.1)
        c3 = c1 + c2
        c4 = c1 - c2
        c5 = c1 * 2
        c6 = c1 / 2
        assert isinstance(c3, Luminance.Color)
        assert isinstance(c4, Luminance.Color)
        assert isinstance(c5, Luminance.Color)
        assert isinstance(c6, Luminance.Color)
    
        print('Luminance: Color arithmetic operations successful. *Passed*')
        return True
    except Exception as e:
        print(f'Luminance: Color arithmetic failed - {e}. *Failed*')
        return False

def test_material_reflect():
    try:
        color = Luminance.Color(0.5, 0.5, 0.5)
        mat = Luminance.SimpleMaterial(color, 0.1, 0.9)
        ray = PrimaryStructures.Ray(np.array([1, 2, 3]), np.array([0, 1, 0]))
        normal = np.array([0, 1, 0])
        reflected = mat.ReflectRay(ray, normal, np.array([1, 2, 3]))
        assert isinstance(reflected, PrimaryStructures.Ray)

        print('Luminance: Material reflect operation successful. *Passed*')
        return True
    except Exception as e:
        print(f'Luminance: Material reflect failed - {e}. *Failed*')
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
            print('Reflections: Failed to raise error on shape mismatch. *Failed*')
            return False
        except Exception:
            print('Reflections: angle, reflect_ray, and error handling tests successful. *Passed*')
            return True
        
    except Exception as e:
        print(f'Reflections: Failed - {e}. *Failed*')
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

        # Test error for total internal reflection
        try:
            Refractions.calculate_angle_of_refraction(60, 1.5, 1.0)
            print('Refractions: Failed to raise error for total internal reflection. *Failed*')
            return False
        except Exception:
            pass

        # Test refract_ray
        ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([1,0,0]))
        normal = np.array([0,1,0])
        try:
            Refractions.refract_ray(normal, ray, 1.5, 1.0)
            print('Refractions: Failed to raise error for total internal reflection in refract_ray. *Failed*')
            return False
        except Exception:
            print('Refractions: index, angle, critical angle, and error handling tests successful. *Passed*')
            return True
    except Exception as e:
        print(f'Refractions: Failed - {e}. *Failed*')
        return False

def test_sampler_basic():
    try:
        sampler = Sampler.Sampler(Sampler.SampleSettings(), Sampler.RenderSettings(800, 600, 4))
        if hasattr(sampler, 'sample'):
            _ = sampler.sample(np.array([0.5, 0.5]))
        print('Sampler: Instantiation and method call successful. *Passed*')
        return True
    except Exception as e:
        print(f'Sampler: Failed - {e}. *Failed*')
        return False

def test_projections_basic():
    try:
        proj = Projection.Projection(Camera(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 1.77, 0.1, 1000, 1000))
        if hasattr(proj, 'project'):
            _ = proj.project(np.array([1,2,3]))
        print('Projections: Instantiation and method call successful. *Passed*')
        return True
    except Exception as e:
        print(f'Projections: Failed - {e}. *Failed*')
        return False

def test_camera_basic():
    try:
        cam = Camera.Camera(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 1.77, 0.1, 1000, 1000)
        if hasattr(cam, 'get_view_matrix'):
            _ = cam.get_view_matrix()
        print('Camera: Instantiation and method call successful. *Passed*')
        return True
    except Exception as e:
        print(f'Camera: Failed - {e}. *Failed*')
        return False

def test_raycasting_basic():
    try:
        rc = Raytracing.Raycaster()
        if hasattr(rc, 'cast'):
            _ = rc.cast(np.array([0,0,0]), np.array([1,0,0]))
        print('Raycasting: Instantiation and method call successful. *Passed*')
        return True
    except Exception as e:
        print('Raycasting: Failed - {e}. *Failed*')
        return False

def test_circle_negative_radius():
    try:
        circle = Gemometry.Circle(np.array([0, 0, 0]), -5)
        print('Geometry: Circle with negative radius should raise an error. *Failed*')
        return False
    except Exception:
        print('Geometry: Circle with negative radius correctly raised an error. *Passed*')
        return True

def test_triangle_area_zero():
    v1 = np.array([0, 0, 0])
    v2 = np.array([0, 0, 0])
    v3 = np.array([0, 0, 0])
    try:
        triangle = Gemometry.Triangle(v1, v2, v3)
        print("Geometry: Triangle with zero area should raise an error. *Failed*")
        return False
    except Exception:
        print("Geometry: Triangle with zero area correctly raised an error. *Passed*")
        return True

def test_color_clamping():
    color: Luminance.ColorData = Luminance.ColorData(0, 0, 0)
    color.rgba = np.array([1.5, -0.2, 0.5, 1.2])  # Intentionally out of bounds
    clamped = color.clamp()
    if 0.0 <= clamped.red <= 1.0 and 0.0 <= clamped.green <= 1.0 and 0.0 <= clamped.blue <= 1.0:
        print("Luminance: Color clamping works as expected. *Passed*")
        return True
    else:
        print("Luminance: Color clamping failed. *Failed*")
        return False

def test_camera_fov_limits():
    try:
        cam = Camera.CameraObject(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 0, 1.77, 0.1, 1000, 1000)
        print("Camera: Camera with zero FOV should raise an error. *Failed*")
        return False
    except Exception:
        print("Camera: Camera with zero FOV correctly raised an error. *Passed*")
        return True

def test_ray_normalization():
    ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([10,0,0]))
    norm = np.linalg.norm(ray.direction)
    if abs(norm - 1.0) < 1e-6:
        print("PrimaryStructures: Ray direction normalization works as expected. *Passed*")
        return True
    else:
        print("PrimaryStructures: Ray direction normalization failed. *Failed*")
        return False

def test_material_reflect_color_bounds():
    color = Luminance.ColorData(1.0, 1.0, 1.0)
    mat = Luminance.Material(color, 0.5, 0.5)
    reflected = mat.AffectColor(Luminance.ColorData(2.0, 2.0, 2.0))
    if all(0.0 <= c <= 1.0 for c in [reflected.red, reflected.green, reflected.blue]):
        print("Luminance: Material reflect color bounds are correct. *Passed*")
        return True
    else:
        print("Luminance: Material reflect color bounds failed. *Failed*")
        return False

if __name__ == "__main__":
    run_tests()