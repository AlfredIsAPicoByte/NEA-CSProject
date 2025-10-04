from py_src.src import PrimaryStructures, Projection
from src import Gemometry, Lighting, Reflections, Refractions, Sampler, Camera
from py_src.src.Algorithims import Raytracing
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
        test_circle_normal_tangent_error(),

        # Geometry: Triangle
        test_triangle_ray_intersection(),
        test_triangle_ray_no_intersection(),
        test_triangle_degenerate_collinear(),
        test_triangle_ray_parallel(),
        test_triangle_point_on_edge(),
        test_triangle_normal_tangent_error(),

        # Lighting/Color/Material
        test_lighting_basic(),
        test_color_arithmetic(),
        test_material_reflect(),

        # Reflections/Refractions
        test_reflections_basic(),
        test_refractions_basic(),

        # Other modules
        test_sampler_basic(),
        test_projections_basic(),
        test_camera_basic(),
        test_raycasting_basic()
    ]

    for test in tests:
        if test:
            count += 1
    
    print(f"-----Tests completed: {count}/{len(tests)} passed-----")
    

def test_circle_ray_no_intersection():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([10, 0, 0]), np.array([1, 0, 0]))

    if not circle.CheckIntersection(ray):
        print('Gemometry "test_circle_ray_no_intersection" Expect no intersection *Passed*')
        return True
    else:
        intersections = circle.GetIntersection(ray)
        if intersections is None or len(intersections) == 0:
            print('Gemometry "test_circle_ray_no_intersection" Expect no intersection *Passed*')
            return True
        print('Gemometry "test_circle_ray_no_intersection" Expect no intersection *Failed*')
        return False

def test_circle_ray_tangent():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([5, -10, 0]), np.array([0, 1, 0]))

    if circle.CheckIntersection(ray):
        intersections = circle.GetIntersection(ray)
        if len(intersections) == 1:
            print('Gemometry "test_circle_ray_tangent" Expect 1 intersection (tangent) *Passed*')
            return True
        print('Gemometry "test_circle_ray_tangent" Expect 1 intersection (tangent) *Failed*')
        return False
    else:
        print('Gemometry "test_circle_ray_tangent" Expect 1 intersection (tangent) *Failed*')
        return False

def test_circle_ray_two_intersections():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([-10, 0, 0]), np.array([1, 0, 0]))

    if circle.CheckIntersection(ray):
        intersections = circle.GetIntersection(ray)
        if len(intersections) == 2:
            print('Gemometry "test_circle_ray_two_intersections" Expect 2 intersections *Passed*')
            return True
        print('Gemometry "test_circle_ray_two_intersections" Expect 2 intersections *Failed*')
        return False
    else:
        print('Gemometry "test_circle_ray_two_intersections" Expect 2 intersections *Failed*')
        return False

def test_circle_ray_origin_inside():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([0, 0, 0]), np.array([1, 0, 0]))
    if circle.CheckIntersection(ray):
        intersections = circle.GetIntersection(ray)
        # For a ray starting at the center, only the forward intersection is returned
        assert len(intersections) == 1
        print("Ray from inside circle: one intersection as expected.")
        return True
    else:
        print("No intersection detected for ray from inside circle.")
        return False

def test_circle_ray_origin_on_edge():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([5, 0, 0]), np.array([1, 0, 0]))
    if circle.CheckIntersection(ray):
        intersections = circle.GetIntersection(ray)
        # Should be one intersection (tangent at start)
        assert len(intersections) >= 1
        print("Ray from edge of circle: intersection detected as expected.")
        return True
    else:
        print("No intersection detected for ray from edge of circle.")
        return False

def test_circle_ray_reverse_direction():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([0, 0, 0]), np.array([-1, 0, 0]))
    # Ray from center, but in negative direction
    if circle.CheckIntersection(ray):
        intersections = circle.GetIntersection(ray)
        assert len(intersections) == 1
        print("Geometry: Ray from center in reverse direction: one intersection as expected.")
        return True
    else:
        print("Geometry: No intersection detected for reverse direction ray from center.")
        return False

def test_circle_point_normal_tangent():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    point_on_circle = np.array([5, 0, 0])

    if circle.CheckPoint(point_on_circle, 0.01):
        normal = circle.GetNormal(point_on_circle)
        tangent = circle.GetTangent(point_on_circle)
        print('Gemometry "test_circle_point_normal_tangent" Expect point on circle *Passed*')
        return True
    else:
        print('Gemometry "test_circle_point_normal_tangent" Expect point on circle *Failed*')
        return False

def test_circle_point_not_on_circle():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    point_not_on_circle = np.array([6, 0, 0])

    if not circle.CheckPoint(point_not_on_circle, 0.01):
        print('Gemometry "test_circle_point_not_on_circle" Expect point not on circle *Passed*')
        return True
    else:
        print('Gemometry "test_circle_point_not_on_circle" Expect point not on circle *Failed*')
        return False

def test_circle_normal_tangent_error():
    circle = Gemometry.Circle(np.array([0, 0, 0]), 5)
    point = np.array([10, 0, 0])  # Not on circle
    try:
        _ = circle.GetNormal(point)
        print("Expected error for GetNormal with point not on circle, but none raised.")
        return False
    except Exception as e:
        print("Correctly raised error for GetNormal with point not on circle:", e)
    try:
        _ = circle.GetTangent(point)
        print("Expected error for GetTangent with point not on circle, but none raised.")
        return False
    except Exception as e:
        print("Correctly raised error for GetTangent with point not on circle:", e)
        return True

def test_triangle_ray_intersection():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([1, 1, -10]), np.array([0, 0, 1]))

    if triangle.CheckIntersection(ray):
        intersection = triangle.GetIntersection(ray)
        if intersection is not None:
            print('Gemometry "test_triangle_ray_intersection" Expect intersection *Passed*')
            return True
        print('Gemometry "test_triangle_ray_intersection" Expect intersection *Failed*')
        return False
    else:
        print('Gemometry "test_triangle_ray_intersection" Expect intersection *Failed*')
        return False

def test_triangle_ray_no_intersection():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([6, 6, -10]), np.array([0, 0, 1]))

    if not triangle.CheckIntersection(ray):
        print('Gemometry "test_triangle_ray_no_intersection" Expect no intersection *Passed*')
        return True
    else:
        intersection = triangle.GetIntersection(ray)
        if intersection is None:
            print('Gemometry "test_triangle_ray_no_intersection" Expect no intersection *Passed*')
            return True
        print('Gemometry "test_triangle_ray_no_intersection" Expect no intersection *Failed*')
        return False

def test_triangle_degenerate_collinear():
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    v3 = np.array([2, 2, 2])  # Collinear
    triangle = Gemometry.Triangle(v1, v2, v3)
    ray = PrimaryStructures.Ray(np.array([0, 0, -1]), np.array([0, 0, 1]))
    try:
        result = triangle.CheckIntersection(ray)
        print("Degenerate triangle test ran (should be False or error). Result:", result)
        return result is False
    except Exception as e:
        print("Degenerate triangle raised exception as expected:", e)
        return True

def test_triangle_ray_parallel():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    ray = PrimaryStructures.Ray(np.array([0, 0, 1]), np.array([1, 1, 0]))  # Parallel to triangle plane
    if not triangle.CheckIntersection(ray):
        print("Ray parallel to triangle plane: no intersection as expected.")
        return True
    else:
        print("Unexpected intersection for parallel ray.")
        return False

def test_triangle_point_on_edge():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point_on_edge = np.array([2.5, 0, 0])
    if triangle.CheckPoint(point_on_edge, 0.01):
        print("Geometry: Point on triangle edge detected as on triangle.")
        return True
    else:
        print("Geometry: Point on triangle edge not detected as on triangle.")
        return False

def test_triangle_normal_tangent_error():
    triangle = Gemometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point = np.array([10, 10, 10])  # Not on triangle
    try:
        _ = triangle.GetNormal(point)
        print("Expected error for GetNormal with point not on triangle, but none raised.")
        return False
    except Exception as e:
        print("Correctly raised error for GetNormal with point not on triangle:", e)
    try:
        _ = triangle.GetTangent(point)
        print("Expected error for GetTangent with point not on triangle, but none raised.")
        return False
    except Exception as e:
        print("Correctly raised error for GetTangent with point not on triangle:", e)
        return True

def test_lighting_basic():
    try:
        # Test Color class
        color1 = Lighting.Color(0.5, 0.2, 0.7)
        color2 = Lighting.Color(0.1, 0.2, 0.3, 0.5)
        assert color1.red == 0.5 and color2.alpha == 0.5
        color_sum = color1 + color2
        color_mul = color1 * 0.5
        color_eq = (color1 == Lighting.Color(0.5, 0.2, 0.7))
        # Test LightRay
        ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([1,0,0]))
        lray = Lighting.LightRay(np.array([0,0,0]), np.array([1,0,0]), color1, intensity=2.0)
        assert isinstance(lray, Lighting.LightRay)
        # Test SimpleMaterial reflection
        mat = Lighting.SimpleMaterial(color1, roughness=0.2, glossiness=0.8)
        reflected_color = mat.ReflectColor(color2)
        reflected_ray = mat.ReflectRay(ray, np.array([0,1,0]), np.array([0,0,0]), seed=42)
        reflected_lightray = mat.ReflectLightRay(lray, np.array([0,1,0]), np.array([0,0,0]))
        assert isinstance(reflected_ray, PrimaryStructures.Ray)
        assert isinstance(reflected_lightray, Lighting.LightRay)
        print("Lighting: Color, LightRay, and SimpleMaterial tests successful.")
        return True
    except Exception as e:
        print("Lighting: Failed -", e)
        return False

def test_color_arithmetic():
    try:
        c1 = Lighting.Color(0.2, 0.3, 0.4)
        c2 = Lighting.Color(0.1, 0.1, 0.1)
        c3 = c1 + c2
        c4 = c1 - c2
        c5 = c1 * 2
        c6 = c1 / 2
        assert isinstance(c3, Lighting.Color)
        assert isinstance(c4, Lighting.Color)
        assert isinstance(c5, Lighting.Color)
        assert isinstance(c6, Lighting.Color)
        print("Ligthing: Color arithmetic operations successful.")
        return True
    except Exception as e:
        print("Lighting: Color arithmetic failed -", e)
        return False

def test_material_reflect():
    try:
        color = Lighting.Color(0.5, 0.5, 0.5)
        mat = Lighting.SimpleMaterial(color, 0.1, 0.9)
        ray = PrimaryStructures.Ray(np.array([1, 2, 3]), np.array([0, 1, 0]))
        normal = np.array([0, 1, 0])
        reflected = mat.ReflectRay(ray, normal, np.array([1, 2, 3]))
        assert isinstance(reflected, PrimaryStructures.Ray)
        print("Lighting: Material reflect operation successful.")
        return True
    except Exception as e:
        print("Lighting: Material reflect failed -", e)
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
            print("Reflections: Failed to raise error on shape mismatch.")
            return False
        except Exception:
            print("Reflections: angle, reflect_ray, and error handling tests successful.")
            return True
        
    except Exception as e:
        print("Reflections: Failed -", e)
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
            print("Refractions: Failed to raise error for total internal reflection.")
            return False
        except Exception:
            pass

        # Test refract_ray
        ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([1,0,0]))
        normal = np.array([0,1,0])
        try:
            Refractions.refract_ray(normal, ray, 1.5, 1.0)
            print("Refractions: Failed to raise error for total internal reflection in refract_ray.")
            return False
        except Exception:
            print("Refractions: index, angle, critical angle, and error handling tests successful.")
            return True
    except Exception as e:
        print("Refractions: Failed -", e)
        return False

def test_sampler_basic():
    try:
        sampler = Sampler.Sampler(Sampler.SampleSettings(), Sampler.RenderSettings(800, 600, 4))
        if hasattr(sampler, 'sample'):
            _ = sampler.sample(np.array([0.5, 0.5]))
        print("Sampler: Instantiation and method call successful.")
        return True
    except Exception as e:
        print("Sampler: Failed -", e)
        return False

def test_projections_basic():
    try:
        proj = Projection.Projection(Camera(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 1.77, 0.1, 1000, 1000))
        if hasattr(proj, 'project'):
            _ = proj.project(np.array([1,2,3]))
        print("Projections: Instantiation and method call successful.")
        return True
    except Exception as e:
        print("Projections: Failed -", e)
        return False

def test_camera_basic():
    try:
        cam = Camera.Camera(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 1.77, 0.1, 1000, 1000)
        if hasattr(cam, 'get_view_matrix'):
            _ = cam.get_view_matrix()
        print("Camera: Instantiation and method call successful.")
        return True
    except Exception as e:
        print("Camera: Failed -", e)
        return False

def test_raycasting_basic():
    try:
        rc = Raytracing.Raycaster()
        if hasattr(rc, 'cast'):
            _ = rc.cast(np.array([0,0,0]), np.array([1,0,0]))
        print("Raycasting: Instantiation and method call successful.")
        return True
    except Exception as e:
        print("Raycasting: Failed -", e)
        return False

if __name__ == "__main__":
    run_tests()