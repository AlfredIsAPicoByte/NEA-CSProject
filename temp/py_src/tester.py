from src import *
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

# --- Available Tests Registry ---

def test_ray() -> bool:
    """
    Test basic Ray structure functionality.
    """
    try:
        from src.PrimaryStructures import Ray

        origin = np.zeros(3)
        direction = np.array([1, 0, 0])
        ray = Ray(origin, direction)

        # Use numpy-aware comparisons to avoid ambiguous truth-value errors
        if not np.allclose(ray.origin, origin):
            return False, f"Ray origin mismatch, {ray.origin} != {origin}"
        if not np.allclose(ray.direction, direction):
            return False, f"Ray direction mismatch, {ray.direction} != {direction}"

        point_at_5 = ray.point_at(5)
        expected_point = np.array([5, 0, 0])
        if not np.allclose(point_at_5, expected_point):
            return False, f"Ray point_at_parameter incorrect: got {point_at_5}, expected {expected_point}"
        
        return True
    except Exception as e:
        return False, str(e)

def test_shapes() -> bool:
    shapes = {
        "Circle": [[0, 0], 1],
        "Triangle": [[0, 0], [1, 0], [0, 1]],
        "Polygon": [[0, 0], [1, 0], [1,1], [0,1]],
        "Sphere": [[0, 0, 0], 1],
        "Cube": [[0 , 0, 0], 1],
    }

    from src.Geometry import ShapeFactory

    for shape_name, shape_args in shapes.items():
        try:
            factory_class = getattr(__import__("src.Geometry", fromlist=[shape_name + "Factory"]), shape_name + "Factory")
            factory: ShapeFactory = factory_class()
            shape = factory.create(*shape_args)
            if shape is None:
                return False, f"{shape_name} creation returned None"
        except Exception as e:
            return False, f"Error creating {shape_name}: {e}"
    
    # All shapes created successfully
    return True

def test_transform() -> bool:
    from src.PrimaryStructures import Transform

    position = np.array([1, 2, 3])
    rotation = np.array([0, 0, 0])
    scale = np.array([1, 1, 1])

    transform = Transform(position, rotation, scale)

    if not np.allclose(transform.position, position):
        return False, f"Transform position mismatch, {transform.position} != {position}"
    if not np.allclose(transform.rotation, rotation):
        return False, f"Transform rotation mismatch, {transform.rotation} != {rotation}"
    if not np.allclose(transform.scale, scale):
        return False, f"Transform scale mismatch, {transform.scale} != {scale}"
    
    transform.translate(np.array([1, 0, 0]))
    expected_position = np.array([2, 2, 3])
    if not np.allclose(transform.position, expected_position):
        return False, f"Transform translation mismatch, {transform.position} != {expected_position}"
    
    transform.rotate(np.pi/2, np.array([0, 1, 0]))
    excepted_rotation = rotation + np.array([0, np.pi/2, 0])
    if not np.allclose(transform.rotation, excepted_rotation):
        return False, f"Transform rotation mismatch, {transform.rotation} != {excepted_rotation}"
    
    transform.enlarge(np.array([2, 2, 2]))
    expected_scale = np.array([2, 2, 2])
    if not np.allclose(transform.scale, expected_scale):
        return False, f"Transform scale mismatch, {transform.scale} != {expected_scale}"
    
    return True

def test_ratios() -> bool:
    from src.PrimaryStructures import Ratio

    ratio = Ratio(16, 9)
    if ratio.width != 16 or ratio.height != 9:
        return False, f"Ratio dimensions mismatch, {ratio.width}x{ratio.height} != 16x9"
    if abs(ratio.value - (16 / 9)) > 1e-6:
        return False, f"Ratio value mismatch, {ratio.value} != {16 / 9}"
    return True

def test_ray_check_points() -> bool:
    from src.PrimaryStructures import Ray

    origin = np.zeros(3)
    direction = np.array([0, 1, 0])
    ray = Ray(origin, direction)

    test_points = {
        0: np.array([0, 0, 0]),
        1: np.array([0, 1, 0]),
        5: np.array([0, 5, 0]),
        -3: np.array([0, -3, 0]),
        100: np.array([0, 50, 0]),
        -12: np.array([0, -6, 0]),
        2.5: np.array([0, 2.5, 0]),
        15: np.array([0, 0, 15]),
    }
    expected_results = [
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        False
    ]

    for (t, expected_point), expected_result in zip(test_points.items(), expected_results):
        point = ray.point_at(t)
        matches = np.allclose(point, expected_point)
        if expected_result != matches:
            return False, f"Ray point_at({t}) incorrect: got {point}, expected {expected_point} (expected_result={expected_result}, got {matches})"

    return True

def test_ray_transormation() -> bool:
    from src.PrimaryStructures import Ray

    origin = np.zeros(3)
    direction = np.array([0, 1, 0])
    ray = Ray(origin, direction)

    ray.rotate(np.pi / 2, np.array([0, 0, 1]))  # Rotate 90 degrees around Z-axis
    expected_direction = np.array([-1, 0, 0])
    if not np.allclose(ray.direction, expected_direction):
        return False, f"Ray rotation incorrect: got {ray.direction}, expected {expected_direction}"
    
    ray.translate(np.array([1, 0, 0]))  # Translate by (1, 0, 0)
    expected_origin = np.array([1, 0, 0])
    if not np.allclose(ray.origin, expected_origin):
        return False, f"Ray translation incorrect: got {ray.origin}, expected {expected_origin}"
    
    return True

def test_vobject_creation() -> bool:
    from src.Geometry import VObject,SphereFactory
    from src.PrimaryStructures import Transform

    transform = Transform(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 1]))
    sphere_factory = SphereFactory()
    sphere_shape = sphere_factory.create(np.array([0, 0, 0]), 1)

    vobject = VObject(sphere_shape, transform)

    if vobject.shape != sphere_shape:
        return False, f"VObject shape mismatch, {vobject.shape} != {sphere_shape}"
    if vobject.transform != transform:
        return False, f"VObject transform mismatch, {vobject.transform} != {transform}"

    return True

def test_color_operations() -> bool:
    from src.Luminance import Color

    color1 = Color(0.2, 0.4, 0.6)
    color2 = Color(0.1, 0.2, 0.3)

    added_color = color1 + color2
    expected_add = Color(0.3, 0.6, 0.9)
    if not np.allclose([added_color.red, added_color.green, added_color.blue], [expected_add.red, expected_add.green, expected_add.blue]):
        return False, f"Color addition incorrect: got {added_color}, expected {expected_add}"

    scaled_color = color1 * 2
    expected_scale = Color(0.4, 0.8, 1.2)
    if not np.allclose([scaled_color.red, scaled_color.green, scaled_color.blue], [expected_scale.red, expected_scale.green, expected_scale.blue]):
        return False, f"Color scaling incorrect: got {scaled_color}, expected {expected_scale}"

    return True

def test_camera() -> bool:
    from src.Camera import VCamera
    from src.PrimaryStructures import Transform, Ratio
    import numpy as np

    transform = Transform(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 1]))
    ratio = Ratio(16, 9)
    screen_scale = 90 # making the resolution 1440x810
    fov = 90

    camera = VCamera(transform, fov, 0.1, 1000, screen_scale * ratio.width, screen_scale * ratio.height)

    if camera.transform != transform:
        return False, f"Camera transform mismatch, {camera.transform} != {transform}"
    if camera.aspect != ratio:
        return False, f"Camera aspect ratio mismatch, {camera.aspect} != {ratio}"
    if camera.fov != fov:
        return False, f"Camera FOV mismatch, {camera.fov} != {fov}"
    
    new_screen_scale = 110
    camera.resize_aspect(Ratio(4, 3), new_screen_scale) # making the resolution 440x330
    expected_ratio = Ratio(4, 3)
    if camera.aspect != expected_ratio:
        return False, f"Camera resize mismatch, {camera.aspect} != {expected_ratio}"

    return True

def test_light_ray() -> bool:
    from src.Luminance import Color, LightRay
    from src.PrimaryStructures import Ray
    import numpy as np

    origin = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    ray = Ray(origin, direction)
    intensity = 1.0

    color = Color(1.0, 1.0, 1.0)
    light_ray = LightRay.from_ray(ray, color, intensity)

    if light_ray.intensity != intensity:
        return False, f"LightRay intensity mismatch, {light_ray.intensity} != {intensity}"
    if light_ray.base_color != color:
        return False, f"LightRay color mismatch, {light_ray.color} != {color}"

    return True

def test_light_source() -> bool:
    from src.Luminance import LightSource

    position = np.array([10, 10, 10])
    intensity = 1.0
    light = LightSource(position, intensity)

    if not np.allclose(light.position, position):
        return False, f"LightSource position mismatch, {light.position} != {position}"
    if light.intensity != intensity:
        return False, f"LightSource intensity mismatch, {light.intensity} != {intensity}"

    return True

def test_ray_shape_intersection() -> bool:
    from src.Geometry import SphereFactory, CircleFactory
    from src.PrimaryStructures import Ray
    import numpy as np

    sphere_factory = SphereFactory()
    sphere = sphere_factory.create(np.array([0, 0, 0]), 1)

    ray_inside = Ray(np.array([0, 0, 0]), np.array([1, 0, 0]))
    ray_outside = Ray(np.array([2, 2, 2]), np.array([1, 0, 0]))

    if not sphere.CheckRayIntersection(ray_inside):
        return False, "Ray inside sphere should intersect but did not."
    if sphere.CheckRayIntersection(ray_outside):
        return False, "Ray outside sphere should not intersect but did."
    
    circle_factory = CircleFactory()
    circle = circle_factory.create(np.array([0, 0]), 1)

    ray_through = Ray(np.array([0, -1]), np.array([0, 1]))
    ray_miss = Ray(np.array([2, -1]), np.array([0, 1]))
    if not circle.CheckRayIntersection(ray_through):
        return False, "Ray through circle should intersect but did not."
    if circle.CheckRayIntersection(ray_miss):
        return False, "Ray missing circle should not intersect but did"

    return True

available_tests: dict[str, Callable[[], Tuple[bool, str|None] | bool]] = {
    "Ray Structure Test": test_ray,
    "Shape Creation Test": test_shapes,
    "Transform Structure Test": test_transform,
    "Ratio Structure Test": test_ratios,
    "Ray Check Points Test": test_ray_check_points,
    "Ray Transformation Test": test_ray_transormation,
    "VObject Creation Test": test_vobject_creation,
    "Color Operations Test": test_color_operations,
    "Camera Object Test": test_camera,
    "LightRay Structure Test": test_light_ray,
    "LightSource Structure Test": test_light_source,
    "Ray-Shape Intersection Test": test_ray_shape_intersection,
}

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