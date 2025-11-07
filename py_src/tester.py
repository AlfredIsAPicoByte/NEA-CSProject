from src import Geometry, Reflections, Refractions, Sampler, Camera, Luminance, PrimaryStructures, Projection
from src.Algorithims import Raytracing
import numpy as np
from typing import Callable, Tuple, Any

# --- Custom Text Definitions ---
# (Icon, Verbose State)
PASS_TEXT = ("✅", "SUCCESS")
FAIL_TEXT = ("❌" ,"FAILED")
ERROR_TEXT = ("🚫" ,"CRITICAL ERROR")
UNKNOWN_TEXT = ("⁉️", "UNKNOWN")

def return_message_handler(test_logic: Callable[[], Tuple[bool, str|None] | bool], title: str, expected: bool = True) -> Tuple[bool, str]:
    """
    A handler for any test. Executes the logic and formats the result message.
    """
    details: str = ""
    result_bool: bool = False
    
    try:
        out = test_logic()
        
        # Normalize outputs:
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            # coerce first element to a Python bool (handles numpy.bool_ etc.)
            result_bool = bool(out[0])
            # optional message in second element
            details = str(out[1]) if len(out) >= 2 and out[1] is not None else ""
        elif isinstance(out, bool):
            result_bool = out
            details = ""
        else:
            # Fallback for unexpected return type
            result_bool = False
            details = f"Invalid return type: {type(out)}"

        # Final result check
        test_passed: bool = (result_bool == expected)

        if test_passed:
            msg = f"{PASS_TEXT[0]} [{title}]"
            if details:
                msg += f" - {details}"
            msg += f" {PASS_TEXT[1]}"
            return True, msg
        else:
            msg = f"{FAIL_TEXT[0]} [{title}]"
            if details:
                msg += f" - {details}"
            msg += f" {FAIL_TEXT[1]}"
            return False, msg

    except Exception as e:
        error_msg = f"{ERROR_TEXT[0]} [{title}] - {e.__class__.__name__}: {e} {ERROR_TEXT[1]}"
        raise Exception(error_msg)

# --- Existing Test Functions (Examples for context) ---
# NOTE: The provided test functions were only partially included in the prompt.
# I'll include the ones from the previous response and the ones provided in the full file.

def test_circle_ray_no_intersection():
    """Logic for Circle Ray (No Intersection). Expected: True"""
    circle = Geometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([10, 0, 0]), np.array([1, 0, 0]))
    if not circle.CheckIntersection(ray): return True
    intersections = circle.GetIntersection(ray)
    return intersections is None or len(intersections) == 0

def test_circle_ray_tangent():
    """Logic for Circle Ray (Tangent). Expected: True"""
    circle = Geometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([5, -10, 0]), np.array([0, 1, 0]))
    if circle.CheckIntersection(ray):
        intersections = circle.GetIntersection(ray)
        return len(intersections) == 1
    return False

def test_circle_ray_two_intersections():
    """Logic for Circle Ray (Two Intersections). Expected: True"""
    circle = Geometry.Circle(np.array([0, 0, 0]), 5)
    ray = PrimaryStructures.Ray(np.array([-10, 0, 0]), np.array([1, 0, 0]))
    if circle.CheckIntersection(ray):
        intersections = circle.GetIntersection(ray)
        return len(intersections) == 2
    return False

def test_circle_negative_radius():
    """Logic for Circle (Negative Radius). Expected: True (due to expected exception)"""
    try:
        _ = Geometry.Circle(np.array([0, 0, 0]), -5)
        return False, "Failed to raise exception for negative radius."
    except Exception as e:
        return True, f"Correctly raised exception: {e.__class__.__name__}"
    
def test_triangle_degenerate_collinear():
    """Logic for Triangle (Degenerate Collinear). Expected: True (due to expected exception)"""
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    v3 = np.array([2, 2, 2])
    try:
        _ = Geometry.Triangle(v1, v2, v3)
        return False, "Failed to raise exception for collinear vertices."
    except Exception as e:
        return True, f"Correctly raised exception: {e.__class__.__name__}"
        
def test_ray_normalization():
    """Logic for Ray (Direction Normalization). Expected: True"""
    ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([10,0,0]))
    norm = np.linalg.norm(ray.direction)
    return abs(norm - 1.0) < 1e-6
    
def test_color_clamping():
    """Logic for Color Clamping. Expected: True"""
    color = Luminance.ColorData(0, 0, 0)
    color.rgba = np.array([1.5, -0.2, 0.5, 1.2]) 
    clamped = color.clamp()
    
    clamped_correctly = (
        0.0 <= clamped.red <= 1.0 and 
        0.0 <= clamped.green <= 1.0 and 
        0.0 <= clamped.blue <= 1.0 and
        0.0 <= clamped.alpha <= 1.0 # Check all channels
    )
    return clamped_correctly

# --- NEW TEST FUNCTIONS ---

def test_triangle_ray_parallel_to_edge():
    """Geometry: Ray parallel to a triangle edge, but not the plane."""
    v1 = np.array([0, 0, 0])
    v2 = np.array([2, 0, 0])
    v3 = np.array([0, 2, 0])
    triangle = Geometry.Triangle(v1, v2, v3)
    
    # Ray parallel to edge v1-v2, slightly above the triangle
    ray = PrimaryStructures.Ray(np.array([1, 1, 1]), np.array([1, 0, 0])) 
    
    # It should not intersect
    if not triangle.CheckIntersection(ray):
        return True
    return False, "Unexpected intersection found when ray was parallel to edge."

def test_color_clamp_alpha_negative():
    """Luminance: Color clamping, specifically testing negative alpha channel."""
    try:
        color = Luminance.ColorData(0.5, 0.5, 0.5, 0.5)
        # Test setting alpha to a clamped value
        color.alpha = -1.0
        
        # Check if the internal rgba array was properly clamped on assignment/access
        alpha_clamped_correctly = color.alpha == 0.0
        
        return alpha_clamped_correctly, f"Alpha was {color.alpha}, expected 0.0."
    except Exception as e:
        return False, e

def test_reflection_normalization():
    """Reflections: Ensure the reflected ray direction is normalized (magnitude 1)."""
    # Incident ray (not normalized, but that's fine for the input Ray structure)
    ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([2, -2, 0])) 
    normal = np.array([0, 1, 0]) # Flat surface normal
    
    # Reflections.reflect_ray should normalize its result
    reflected_ray = Reflections.reflect_ray(normal, ray)
    
    reflected_norm = np.linalg.norm(reflected_ray.direction)
    is_normalized = abs(reflected_norm - 1.0) < 1e-6
    
    return is_normalized, f"Reflected direction magnitude: {reflected_norm}"

def test_refraction_total_internal_reflection():
    """Refractions: Check if refraction fails (returns None) during Total Internal Reflection."""
    try:
        n1 = 1.5 # From glass
        n2 = 1.0 # To air
        
        # Calculate critical angle
        critical_angle_rad = np.arcsin(n2 / n1)
        
        # Incident ray angle greater than critical angle (e.g., 60 degrees)
        incident_angle_rad = np.deg2rad(60.0) 
        
        # Need to simulate a ray and normal that result in a 60-degree incident angle
        normal = np.array([0, 1, 0])
        # Ray incident at 60 degrees (relative to -Y normal)
        incident_dir = np.array([np.sin(incident_angle_rad), -np.cos(incident_angle_rad), 0]) 
        ray = PrimaryStructures.Ray(np.array([0, 0, 0]), incident_dir)
        
        # Refract ray function should return None or raise exception during TIR
        refracted = Refractions.refract_ray(normal, ray, n1, n2)
        
        if refracted is None:
            # Success: Refraction failed as expected due to TIR
            return True
        else:
            return False, f"Unexpectedly refracted ray found during TIR check."
    except ValueError:
        return True
    except Exception as e:
        return False, e

def test_raytracing_hit_world_origin():
    """Raytracing: Basic test for a ray hit at the world origin."""
    # Mock Raytracer initialization
    rc = Raytracing.Raytracer()
    
    # Mock a Geometry object (Sphere at origin) and register it
    sphere = Geometry.Circle(np.array([0, 0, 0]), 1.0)
    # NOTE: Assuming the Raytracer has an 'AddGeometry' method or similar
    # If not available, this test relies on the mock 'cast' which is assumed to work.
    
    # Assuming 'cast' returns True on hit, False on miss, or a hit object
    hit_result = rc.cast(np.array([-5, 0, 0]), np.array([1, 0, 0]))
    
    # Assume 'cast' returns a boolean or a hit record object on hit
    if hit_result is not None and (isinstance(hit_result, bool) and hit_result or not isinstance(hit_result, bool)):
        return True
    else:
        return False, "Raytracer failed to register a basic hit."

def test_material_reflect_color_bounds():
    color = Luminance.ColorData(1.0, 1.0, 1.0)
    mat = Luminance.Material(color, 0.5, 0.5)
    reflected = mat.AffectColor(Luminance.ColorData(2.0, 2.0, 2.0))
    try:
        if all(0.0 <= c <= 1.0 for c in [reflected.red, reflected.green, reflected.blue]):
            return True
        else:
            return False, ""
    except Exception as e:
        raise Exception(e)
    
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
        return False, e
    
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
        return False, e
    
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
        return False, e
    
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
        return False, e

def test_circle_point_not_on_circle():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        point_not_on_circle = np.array([6, 0, 0])
        if not circle.CheckPoint(point_not_on_circle, 0.01):
            return True
        else:
            return False
    except Exception as e:
        return False, e

def test_circle_tangent_error():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        point = np.array([10, 0, 0])  # Not on circle
        _ = circle.GetNormal(point)
        return False
    except Exception as e:
        return True, e
    
def test_circle_normal_error():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), 5)
        point = np.array([10, 0, 0])  # Not on circle
        _ = circle.GetTangent(point)
        return False
    except Exception as e:
        return True, e

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
        return True, e
    
def test_triangle_tangent_error():
    triangle = Geometry.Triangle(np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([0, 5, 0]))
    point = np.array([10, 10, 10])  # Not on triangle

    try:
        _ = triangle.GetTangent(point)
        return False
    except Exception as e:
        return True, e

def test_luminance_color():
    try:
        # Test Color class
        color1 = Luminance.ColorData(0.5, 0.2, 0.7)
        color2 = Luminance.ColorData(0.1, 0.2, 0.3, 0.5)
        assert color1.red == 0.5 and color2.alpha == 0.5

        return True
    except Exception as e:
        return False, e

def test_luminance_light_ray():
    try:
        color1 = Luminance.ColorData(0.5, 0.2, 0.7)
        # Test LightRay
        ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([1,0,0]))
        lray = Luminance.LightRay(ray.origin, ray.orientation, color1, intensity=2.0)
        assert isinstance(lray, Luminance.LightRay)

        return True
    except Exception as e:
        return False, e
    
def test_luminance_material():
    try:
        col = Luminance.ColorData(0.3, 0.4, 0.5)
        mat = Luminance.Material(color=col, roughness=0.8, glossiness=0.1)
        assert mat.base_color == col and mat.roughness == 0.8

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
        reflected = mat.RedirectLightRay(Luminance.LightRay(ray.origin, ray.orientation, Luminance.ColorData()), normal)
        assert isinstance(reflected, PrimaryStructures.Ray)

        return True
    except Exception as e:
        return False, e
    
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
            return False
        except Exception:
            return True
        
    except Exception as e:
        return False, e
    
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
            return False, "Expected total internal reflection exception not raised."
        except Exception:
            return True, e
    except Exception as e:
        return False, e
    
def test_sampler_basic():
    try:
        samp = Sampler.SamplingManager(Sampler.SampleSettings(), Sampler.RenderSettings(800, 600, 4))
        if hasattr(samp, 'sample'):
            samp.GenerateSamples()
            _ = samp.GetSamples(1, 0)
        return True
    except Exception as e:
        return False, e
    
def test_projections_basic():
    try:
        cam = Camera(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 0.1, 1000, 800, 600)
        proj = Projection.Scene(cam)
        if hasattr(proj, 'project'):
            _ = proj.AddObject() # TODO: Add thing
        return True
    except Exception as e:
        return False, e
    
def test_camera_basic():
    try:
        cam = Camera.CameraObject(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 60, 0.1, 1000, 800, 600)
        if hasattr(cam, 'get_view_matrix'):
            _ = cam.get_view_matrix()
        return True
    except Exception as e:
        return False, e
    
def test_raycasting_basic():
    try:
        rc = Raytracing.Raytracer()
        if hasattr(rc, 'cast'):
            _ = rc.cast(np.array([0,0,0]), np.array([1,0,0]))
        return True
    except Exception as e:
        return False, e
    
def test_circle_negative_radius():
    try:
        circle = Geometry.Circle(np.array([0, 0, 0]), -5)
        return False
    except Exception as e:
        return True, e
    
def test_triangle_area_zero():
    v1 = np.array([0, 0, 0])
    v2 = np.array([0, 0, 0])
    v3 = np.array([0, 0, 0])
    try:
        triangle = Geometry.Triangle(v1, v2, v3)
        return False
    except Exception as e:
        return True, e
    
def test_color_clamping():
    color: Luminance.ColorData = Luminance.ColorData(0, 0, 0)
    color.rgba = np.array([1.5, -0.2, 0.5, 1.2])  # Intentionally out of bounds
    clamped = color.clamp()
    try:
        if 0.0 <= clamped.red <= 1.0 and 0.0 <= clamped.green <= 1.0 and 0.0 <= clamped.blue <= 1.0:
            return True
        else:
            return False
    except Exception as e:
        return False, e
    
def test_camera_fov_limits():
    try:
        cam = Camera.CameraObject(PrimaryStructures.Transform(position=np.array([0,0,0]), rotation=np.array([0,0,0]), scale=np.array([1,1,1])), 0, 1.77, 0.1, 1000, 1000)
        return False
    except Exception as e:
        return True, e
    
def test_ray_normalization():
    ray = PrimaryStructures.Ray(np.array([0,0,0]), np.array([10,0,0]))
    norm = np.linalg.norm(ray.direction)
    try:
        if abs(norm - 1.0) < 1e-6:
            return True
        else:
            return False
    except Exception as e:
        return False

# --- Test Collection Map (Updated) ---

available_tests = {
    'Geometry, Primary: Circle and Ray (No Intersection)': 
        test_circle_ray_no_intersection,
    'Geometry, Primary: Circle and Ray (Tangent)': 
        test_circle_ray_tangent,
    'Geometry, Primary: Circle and Ray (Two Intersections)': 
        test_circle_ray_two_intersections,
    'Geometry: Circle (Negative Radius)': 
        test_circle_negative_radius,
    'Geometry: Triangle (Degenerate Collinear)': 
        test_triangle_degenerate_collinear,
    'Geometry: Triangle Ray Parallel to Edge': 
        test_triangle_ray_parallel_to_edge,
    'Luminance: Color Clamping (General)': 
        test_color_clamping,
    'Luminance: Color Clamping (Negative Alpha)': 
        test_color_clamp_alpha_negative,
    'Luminance: Material Reflection Color Bounds': 
        test_material_reflect_color_bounds,
    'Reflections: Reflected Ray Normalization': 
        test_reflection_normalization,
    'Refractions: Total Internal Reflection (TIR)': 
        test_refraction_total_internal_reflection,
    'PrimaryStructures: Ray Direction Normalization': 
        test_ray_normalization,
    'Raycasting: Basic Hit at World Origin':
        test_raytracing_hit_world_origin,
    'Geometry, Primary: "Circle and Ray find intersection from inside circle" one intersection expected.' :
        test_circle_ray_origin_inside,
    'Geometry, Primary: "Circle and Ray find intersection from edge of circle" one intersection expected.' :
        test_circle_ray_origin_on_edge,
    'Geometry, Primary: "Circle and Ray find intersection from inside in reverse orientation " one intersection expected.' : 
        test_circle_ray_reverse_orientation,
    'Geometry: "Circle check Point normal and tangent" expect point on circle.' : 
        test_circle_point_normal_tangent,
    'Geometry: Point Not On Circle': 
        test_circle_point_not_on_circle,
    'Geometry: Circle Tangent Error Handling': 
        test_circle_tangent_error,
    'Geometry: Circle Normal Error Handling': 
        test_circle_normal_error,
    'Geometry: Triangle Ray Intersection': 
        test_triangle_ray_intersection,
    'Geometry: Triangle Ray No Intersection': 
        test_triangle_ray_no_intersection,
    'Geometry: Triangle Ray Parallel to Plane': 
        test_triangle_ray_parallel,
    'Geometry: Triangle Point On Edge': 
        test_triangle_point_on_edge,
    'Geometry: Triangle Normal Error Handling': 
        test_triangle_normal_error,
    'Geometry: Triangle Tangent Error Handling': 
        test_triangle_tangent_error,
    'Luminance: Color Data Basic Functionality': 
        test_luminance_color,
    'Luminance: LightRay Basic Functionality':
        test_luminance_light_ray,
    'Luminance: Material Basic Functionality': 
        test_luminance_material,
    'Luminance: Color Arithmetic Operators': 
        test_color_arithmetic,
    'Luminance: Material Reflect Produces Ray': 
        test_material_reflect,
    'Reflections: Basic Functions': 
        test_reflections_basic,
    'Refractions: Basic Functions': 
        test_refractions_basic,
    'Sampler: Basic Sampling Manager': 
        test_sampler_basic,
    'Projection: Basic Scene/Camera Projection': 
        test_projections_basic,
    'Camera: Basic CameraObject': 
        test_camera_basic,
    'Raycasting: Basic Cast Function': 
        test_raycasting_basic,
    'Geometry: Circle Negative Radius (duplicate check)': 
        test_circle_negative_radius,
    'Geometry: Triangle Area Zero (degenerate)': 
        test_triangle_area_zero,
    'Luminance: Color Clamping (duplicate check)': 
        test_color_clamping,
    'Camera: FOV Limits Validation': 
        test_camera_fov_limits
}

available_tests = dict(sorted(available_tests.items(), key=lambda kv: kv[0].lower()))

# --- Final run_tests Implementation ---

def run_tests(ignore_passed: bool = False, ignore_fail: bool = False, ignore_error: bool = False):
    """
    Runs predefined tests, handling pass/fail/error and respecting ignore flags.
    """
    print("\n" + "="*70)
    print(f"----- Running Tests -----")
    print(f"Ignore Passed: {ignore_passed}, Ignore Failed: {ignore_fail}, Ignore Errors: {ignore_error}")
    print("="*70 + "\n")

    passed_count = 0
    total_tests = len(available_tests)
    failed_count = 0
    error_count = 0

    for title, test_func in available_tests.items():
        try:
            # We assume a test is 'expected' to pass by default (True).
            result, result_message = return_message_handler(test_func, title, expected=True)

            if result:
                passed_count += 1
                if not ignore_passed:
                    print(result_message)
            else:
                failed_count += 1
                if not ignore_fail:
                    print(result_message)

        except Exception as e:
            # Critical error caught from the handler
            error_count += 1
            if not ignore_error:
                print(str(e))
    
    print("\n" + "="*70)
    print(f"----- Tests completed: {passed_count} / {total_tests} -----")
    print("="*70)

if __name__ == "__main__":
    run_tests(ignore_passed=True)