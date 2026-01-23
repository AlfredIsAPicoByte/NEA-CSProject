import numpy as np
from src.Data.Transform import Transform
from src.Geometry.BVH import build_bvh_tree
from src.Geometry.Primitive import Primitive
from src.Geometry.Core import Sphere
from src.Rendering.RayTracing.Intersections import BVHIntersection
from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from src.Data.Scene import Scene


def test_bvh_respects_world_transforms():
    # Create two spheres separated along X axis
    p1 = Primitive(name="A", transform=Transform(position=np.array([0.0, 0.0, 0.0])), shape=Sphere())
    p2 = Primitive(name="B", transform=Transform(position=np.array([5.0, 0.0, 0.0])), shape=Sphere())

    p1.update_matrices()
    p2.update_matrices()

    bvh = BVHIntersection()
    root = build_bvh_tree([p1, p2])

    # Root box should span from near 0 to near 5 (with padding)
    min_x = root.box.min_point[0]
    max_x = root.box.max_point[0]

    assert min_x < 1.0, f"Expected min_x < 1, got {min_x}"
    assert max_x > 4.0, f"Expected max_x > 4, got {max_x}"


def test_bvh_large_distant_object_no_hit():
    """Test that a large object far beyond max_distance is not hit."""
    # Create a large sphere far away
    large_sphere = Primitive(
        name="LargeBackground", 
        transform=Transform(position=np.array([0.0, 0.0, 1500.0]), scale=np.array([100.0, 100.0, 100.0])), 
        shape=Sphere()
    )
    large_sphere.update_matrices()

    scene = Scene()
    scene.add_object(large_sphere)

    # Ray from origin towards the sphere
    ray = TracingRay(origin=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 1.0]))

    bvh = BVHIntersection(max_distance=1000.0)  # Default max_distance
    hit = bvh.find_hit(scene, ray)

    assert not hit.hit, f"Expected no hit for distant large object, but got hit at distance {hit.distance}"


def test_bvh_large_distant_object_with_hit():
    """Test that the same large distant object is hit when max_distance is increased."""
    # Create a large sphere far away
    large_sphere = Primitive(
        name="LargeBackground", 
        transform=Transform(position=np.array([0.0, 0.0, 1500.0]), scale=np.array([100.0, 100.0, 100.0])), 
        shape=Sphere()
    )
    large_sphere.update_matrices()

    scene = Scene()
    scene.add_object(large_sphere)

    # Ray from origin towards the sphere
    ray = TracingRay(origin=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 1.0]))

    bvh = BVHIntersection(max_distance=2000.0)  # Increased max_distance
    hit = bvh.find_hit(scene, ray)

    assert hit.hit, f"Expected hit for distant large object with increased max_distance, but got no hit"
    assert hit.distance > 1400.0, f"Expected hit distance > 1400, got {hit.distance}"


def test_bvh_large_scaled_object():
    """Test BVH with a large scaled object close by."""
    # Create a large scaled sphere close by
    large_sphere = Primitive(
        name="LargeClose", 
        transform=Transform(position=np.array([0.0, 0.0, 10.0]), scale=np.array([50.0, 50.0, 50.0])), 
        shape=Sphere()
    )
    large_sphere.update_matrices()

    scene = Scene()
    scene.add_object(large_sphere)

    # Ray from origin towards the sphere
    ray = TracingRay(origin=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 1.0]))

    bvh = BVHIntersection(max_distance=1000.0)
    hit = bvh.find_hit(scene, ray)

    assert hit.hit, f"Expected hit for large scaled close object, but got no hit"


from src.Data.Camera import Camera, CameraType


def test_bvh_camera_rays_hit_distant_objects():
    """Test that camera-generated rays hit distant large objects."""
    # Camera at origin, looking down +z
    cam_transform = Transform(position=np.array([0.0, 0.0, 0.0]), rotation=np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=90.0, near=0.1, far=10000.0,
        resolution_width=10, resolution_height=10,
        camera_type=CameraType.PERSPECTIVE
    )

    # Large object far away
    large_sphere = Primitive(
        name="DistantLarge", 
        transform=Transform(position=np.array([0.0, 0.0, 1000.0]), scale=np.array([50.0, 50.0, 50.0])), 
        shape=Sphere()
    )
    large_sphere.update_matrices()

    scene = Scene(camera=cam)
    scene.add_object(large_sphere)

    # Generate a ray through the center of the image
    cam_ray = cam.generate_ray(0.5, 0.5)  # Center normalized
    ray = TracingRay(origin=cam_ray.origin, orientation=cam_ray.orientation)

    bvh = BVHIntersection(max_distance=2000.0)
    hit = bvh.find_hit(scene, ray)

    assert hit.hit, f"Expected hit for distant large object with camera ray, but got no hit. Ray origin: {ray.origin}, direction: {ray.orientation}"


def test_bvh_no_hit_beyond_max_distance():
    """Test that objects beyond max_distance are not hit."""
    # Object at distance 1500
    distant_obj = Primitive(
        name="VeryDistant", 
        transform=Transform(position=np.array([0.0, 0.0, 1500.0])), 
        shape=Sphere()
    )
    distant_obj.update_matrices()

    scene = Scene()
    scene.add_object(distant_obj)

    ray = TracingRay(origin=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 1.0]))

    bvh = BVHIntersection(max_distance=1000.0)
    hit = bvh.find_hit(scene, ray)

    # Since max_distance=1000, and object at 1500, should not hit
    # But as we saw, for small objects, min(*safe_transform.scale)=1, max_dist_local=1000
    # For sphere radius 1, at 1500, local distance 1500, so should not hit
    assert not hit.hit, f"Expected no hit for object beyond max_distance, but got hit at {hit.distance}"
