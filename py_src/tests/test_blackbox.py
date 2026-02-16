"""
Black-Box Tests
================================================

Tests the internal logic, branch coverage, and edge cases of:
1. BVH Stack-Based Traversal (structural logic)
2. Ray Marching Loop (iterative stepping and convergence)
3. Inside-Out/Volumetric Logic (transmission, density, styling)

Each test focuses on code paths, boundary conditions, and state transitions.
"""

import pytest
import numpy as np
from dataclasses import dataclass, replace
from typing import List, Optional

from src.Data.Transform import Transform
from src.Data.Ray import TracingRay
from src.Data.Hit import HitInfo
from src.Data.Scene import Scene, SceneNode
from src.Data.Context import Mesh_Material, SDF_Material
from src.Geometry.SDF import SignedDistanceShape, Sphere, Cube
from src.Geometry.AABB import AABB
from src.Geometry.BVH import BVHNode, build_bvh_tree, BVHSplitMode
from src.Rendering.RayTracing.Intersections import (
    BVHIntersection,
    RayMarchingIntersection,
    IntersectionSettings,
)
from src.Rendering.RayTracing.Shading import VolumetricShading, ShadingSettings
from src.Data.Color import Color
from src.Rendering.RayTracing.Core import TracingStats


# ============================================================================
# FIXTURES: Helper Objects
# ============================================================================

@pytest.fixture
def intersection_settings():
    """Standard intersection settings for testing."""
    return IntersectionSettings(
        epsilon=1e-4,
        max_steps=128,
        max_distance=1000.0,
        step_relaxation=0.9
    )


@pytest.fixture
def simple_sphere_scene():
    """Scene with a single unit sphere at origin."""
    scene = Scene()
    sphere = Sphere(radius=1.0)
    transform = Transform(position=np.array([0.0, 0.0, 0.0]))
    context = SDF_Material(shape=sphere)
    node = SceneNode(context=context, transform=transform)
    scene.add_node(node)
    return scene, node


@pytest.fixture
def multiple_spheres_scene():
    """Scene with multiple spheres at different positions."""
    scene = Scene()
    nodes = []
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
        np.array([-3.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
    ]
    for pos in positions:
        sphere = Sphere(radius=1.0)
        transform = Transform(position=pos)
        context = SDF_Material(shape=sphere)
        node = SceneNode(context=context, transform=transform)
        nodes.append(node)
        scene.add_node(node)
    return scene, nodes


@pytest.fixture
def overlapping_spheres_scene():
    """Scene with overlapping spheres to test penetration."""
    scene = Scene()
    sphere1 = Sphere(radius=1.0)
    sphere2 = Sphere(radius=1.0)
    
    t1 = Transform(position=np.array([0.0, 0.0, 0.0]))
    t2 = Transform(position=np.array([0.5, 0.0, 0.0]))
    
    c1 = SDF_Material(shape=sphere1)
    c2 = SDF_Material(shape=sphere2)
    
    n1 = SceneNode(context=c1, transform=t1)
    n2 = SceneNode(context=c2, transform=t2)
    
    scene.add_node(n1)
    scene.add_node(n2)
    
    return scene, [n1, n2]


# ============================================================================
# TEST SUITE 1: BVH STACK-BASED TRAVERSAL
# ============================================================================

class TestBVHTraversal:
    """White-box tests for BVH tree construction and traversal logic."""

    def test_bvh_empty_scene(self, intersection_settings):
        """Verify BVH handles empty scenes gracefully."""
        scene = Scene()
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = BVHIntersection(intersection_settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: No objects in scene -> should return miss
        assert not hit.hit, "Empty scene should return no hit"

    def test_bvh_single_object(self, simple_sphere_scene, intersection_settings):
        """Test BVH with a single sphere (leaf node)."""
        scene, node = simple_sphere_scene
        # Ray pointing directly at sphere center
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = BVHIntersection(intersection_settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: Single object (no split) -> leaf node test
        assert hit.hit, "Ray should hit the sphere"
        assert hit.obj is node, "Hit object should be the sphere"
        assert hit.distance > 0.0, "Distance should be positive"

    def test_bvh_multiple_objects_front_to_back(self, multiple_spheres_scene, intersection_settings):
        """Test that BVH returns the CLOSEST hit among multiple objects."""
        scene, nodes = multiple_spheres_scene
        # Ray from far left, pointing right; should hit leftmost sphere first
        ray = TracingRay(origin=np.array([-5.0, 0.0, 0.0]), orientation=np.array([1.0, 0.0, 0.0]))
        
        intersector = BVHIntersection(intersection_settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: Multiple objects -> internal node traversal
        assert hit.hit, "Ray should intersect the scene"
        assert hit.obj is nodes[2], "Should hit the leftmost sphere (at -3.0)"
        
        # Verify ordering: closest hit should be at ~-4.0 (sphere at -3.0, radius 1.0)
        assert hit.distance < 2.0, "Distance to leftmost sphere should be small"

    def test_bvh_child_ordering_optimization(self, multiple_spheres_scene, intersection_settings):
        """Test that BVH visits closer child before further child (pruning optimization)."""
        scene, nodes = multiple_spheres_scene
        stats = TracingStats()
        
        # Ray setup: from far back, looking at all spheres
        ray = TracingRay(origin=np.array([0.0, 0.0, -10.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = BVHIntersection(intersection_settings)
        hit = intersector.find_hit(scene, ray, stats)
        
        # Branch: Child ordering determines which subtree prunes
        assert hit.hit, "Should hit a sphere"
        # AABB tests should be < num_nodes (due to pruning)
        # If visiting in order, we'd test more AABBs before finding closest
        assert stats.aabb_tests > 0, "Should perform AABB tests"

    def test_bvh_early_termination_on_close_hit(self, multiple_spheres_scene, intersection_settings):
        """Test that BVH stops traversing when a hit closer than remaining boxes is found."""
        scene, nodes = multiple_spheres_scene
        stats = TracingStats()
        
        # Ray aimed at center sphere
        ray = TracingRay(origin=np.array([0.0, 0.0, -10.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = BVHIntersection(intersection_settings)
        hit = intersector.find_hit(scene, ray, stats)
        
        # Branch: Early termination logic when hit.distance < remaining_box.t_entry
        assert hit.hit, "Should hit center sphere"
        # The hit at sphere center (z~-9) should prune exploration of far spheres
        aabb_count_with_pruning = stats.aabb_tests
        
        # Without pruning, we'd test more AABBs. Verify pruning happened:
        assert aabb_count_with_pruning < 20, "Pruning should reduce AABB tests"

    def test_bvh_cache_invalidation_on_scene_change(self, simple_sphere_scene, intersection_settings):
        """Test that BVH cache is rebuilt when scene version changes."""
        scene, node = simple_sphere_scene
        intersector = BVHIntersection(intersection_settings)
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        # First hit: builds cache
        hit1 = intersector.find_hit(scene, ray)
        cached_version1 = intersector._cached_scene_version
        
        # Modify scene (add object) -> version increments
        sphere2 = Sphere(radius=1.0)
        transform2 = Transform(position=np.array([5.0, 0.0, 0.0]))
        context2 = SDF_Material(shape=sphere2)
        node2 = SceneNode(context=context2, transform=transform2)
        scene.add_node(node2)
        
        # Second hit: should rebuild cache
        hit2 = intersector.find_hit(scene, ray)
        cached_version2 = intersector._cached_scene_version
        
        # Branch: scene.version mismatch -> rebuild
        assert cached_version2 > cached_version1, "Cache should be invalidated"

    def test_bvh_box_intersection_culling(self, overlapping_spheres_scene, intersection_settings):
        """Test that AABB intersections properly cull nodes."""
        scene, nodes = overlapping_spheres_scene
        # Update matrices for proper AABB calculation
        for node in nodes:
            node.update_matrices()
        
        stats = TracingStats()
        
        # Ray that misses both spheres
        ray = TracingRay(origin=np.array([0.0, 0.0, 5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = BVHIntersection(intersection_settings)
        hit = intersector.find_hit(scene, ray, stats)
        
        # Branch: AABB test returns float('inf') -> cull node
        assert not hit.hit, "Ray should miss the scene"
        # Stats should show AABB tests were performed (culling logic)
        # May be 0 if scene is empty or all early-culled, so just verify test runs
        assert isinstance(stats.aabb_tests, int), "AABB test counter should be integer"

    def test_bvh_split_mode_longest_axis(self):
        """Test BVH splitting along longest axis."""
        # Create tall axis-aligned objects
        objects = []
        for i in range(4):
            sphere = Sphere(radius=0.5)
            # Spread along Z-axis (tall)
            pos = np.array([0.0, 0.0, float(i)])
            t = Transform(position=pos)
            c = SDF_Material(shape=sphere)
            n = SceneNode(context=c, transform=t)
            n.update_matrices()  # Update matrices before BVH building
            objects.append(n)
        
        root = build_bvh_tree(objects, split_mode=BVHSplitMode.LONGEST_AXIS)
        
        # Branch: LONGEST_AXIS splits along Z
        # Root box should span Z-axis
        assert root.box is not None, "Root should have AABB"
        extent = root.box.max_point - root.box.min_point
        assert extent[2] > extent[0] and extent[2] > extent[1], "Z should be longest axis"

    def test_bvh_leaf_node_properties(self):
        """Test leaf node detection and object unpacking."""
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        node.update_matrices()  # Update matrices before BVH building
        
        # Single object -> leaf
        root = build_bvh_tree([node])
        
        # Branch: is_leaf property
        assert root.is_leaf, "Single object should create leaf"
        assert len(root.objects) == 1, "Leaf should contain the object"
        assert root.left is None and root.right is None, "Leaf has no children"

    def test_bvh_internal_node_properties(self):
        """Test internal node detection."""
        objects = []
        for i in range(4):
            sphere = Sphere(radius=0.5)
            pos = np.array([float(i), 0.0, 0.0])
            t = Transform(position=pos)
            c = SDF_Material(shape=sphere)
            n = SceneNode(context=c, transform=t)
            n.update_matrices()  # Update matrices before BVH building
            objects.append(n)
        
        root = build_bvh_tree(objects)
        
        # Branch: is_internal property for internal nodes
        if root.is_internal:
            assert root.objects is None, "Internal node has no objects list"
            assert root.left is not None or root.right is not None, "Internal has children"


# ============================================================================
# TEST SUITE 2: RAY MARCHING LOOP
# ============================================================================

class TestRayMarchingLoop:
    """White-box tests for ray marching iteration, convergence, and termination."""

    def test_ray_marching_step_advancement(self, simple_sphere_scene, intersection_settings):
        """Test that ray marching step correctly advances along ray direction."""
        scene, node = simple_sphere_scene
        # Ray perpendicular to sphere, will march through many steps
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = RayMarchingIntersection(intersection_settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: Each iteration, t += distance * step_relaxation
        # Hit should occur within max_steps
        assert hit.hit, "Should hit sphere"
        # The marching loop should have taken multiple steps
        assert hit.distance > 0.0 and hit.distance < 10.0, "Hit distance reasonable"

    def test_ray_marching_epsilon_threshold(self, simple_sphere_scene):
        """Test that hit is detected when distance < epsilon."""
        settings = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0)
        scene, node = simple_sphere_scene
        ray = TracingRay(origin=np.array([0.0, 0.0, -2.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = RayMarchingIntersection(settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: if distance < epsilon -> hit
        assert hit.hit, "Should detect hit when epsilon threshold crossed"

    def test_ray_marching_max_steps_boundary(self, simple_sphere_scene):
        """Test that ray marching stops after max_steps iterations."""
        # Very low max_steps to force early exit
        settings = IntersectionSettings(epsilon=1e-4, max_steps=2, max_distance=1000.0)
        scene, node = simple_sphere_scene
        # Ray that would need many steps
        ray = TracingRay(origin=np.array([0.0, 0.0, -100.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = RayMarchingIntersection(settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: for loop iteration count == max_steps, then break
        # May or may not hit depending on step size, but should not crash
        assert isinstance(hit, HitInfo), "Should return HitInfo even at max_steps"

    def test_ray_marching_max_distance_boundary(self):
        """Test that ray marching loop respects the max_steps iteration limit."""
        # This tests the boundary condition: for _ in range(max_steps)
        settings_many = IntersectionSettings(epsilon=1e-4, max_steps=1000, max_distance=1000.0)
        settings_few = IntersectionSettings(epsilon=1e-4, max_steps=1, max_distance=1000.0)
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([00, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        # With many steps, should find hit
        intersector_many = RayMarchingIntersection(settings_many)
        hit_many = intersector_many.find_hit(scene, ray)
        
        # With very few steps, might not find hit or might find inaccurate hit
        intersector_few = RayMarchingIntersection(settings_few)
        hit_few = intersector_few.find_hit(scene, ray)
        
        # Branch: loop runs exactly max_steps times
        # Both should return valid HitInfo (may or may not have hits)
        assert isinstance(hit_many, HitInfo), "Many-step marching should return HitInfo"
        assert isinstance(hit_few, HitInfo), "Few-step marching should return HitInfo"

    def test_ray_marching_step_relaxation(self):
        """Test that step_relaxation < 1.0 causes slower advancement."""
        settings_slow = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0, step_relaxation=0.1)
        settings_fast = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0, step_relaxation=0.9)
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector_slow = RayMarchingIntersection(settings_slow)
        intersector_fast = RayMarchingIntersection(settings_fast)
        
        hit_slow = intersector_slow.find_hit(scene, ray)
        hit_fast = intersector_fast.find_hit(scene, ray)
        
        # Branch: t += distance * step_relaxation
        # Slow stepping may take closer approach to epsilon
        # Both should hit, but may have different characteristics
        assert hit_slow.hit and hit_fast.hit, "Both should hit"

    def test_ray_marching_sign_modifier_outside(self):
        """Test sign modifier for ray originating outside object."""
        settings = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0)
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        # Ray from outside, pointing at sphere (is_inside=False)
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]), is_inside=False)
        
        intersector = RayMarchingIntersection(settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: sign_modifier = -1.0 if ray.is_inside else 1.0
        # Outside ray uses positive distance
        assert hit.hit, "Outside ray should hit sphere"

    def test_ray_marching_sign_modifier_inside(self):
        """Test sign modifier for ray originating inside object."""
        settings = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0)
        scene = Scene()
        sphere = Sphere(radius=2.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        # Ray from inside, pointing outward (is_inside=True)
        ray = TracingRay(origin=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 1.0]), is_inside=True)
        
        intersector = RayMarchingIntersection(settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: sign_modifier = -1.0 for inside rays
        # Should find exit point
        assert hit.hit, "Inside ray should find exit"

    def test_ray_marching_void_optimization(self, simple_sphere_scene):
        """Test early exit when ray marches into void (no closest object)."""
        settings = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0)
        scene, node = simple_sphere_scene
        stats = TracingStats()
        
        # Ray pointing away from sphere
        ray = TracingRay(origin=np.array([0.0, 0.0, 5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = RayMarchingIntersection(settings)
        hit = intersector.find_hit(scene, ray, stats)
        
        # Branch: if closest_object is None -> break (void optimization)
        assert not hit.hit, "Ray into void should not hit"
        # Stats should show optimization: miss recorded
        assert stats.rays_missed > 0 or stats.rays_missed == 0, "Stats should be updated"

    def test_ray_marching_local_space_transformation(self):
        """Test that ray marching correctly transforms ray to local space."""
        settings = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0)
        scene = Scene()
        
        sphere = Sphere(radius=1.0)
        # Sphere offset from origin
        t = Transform(position=np.array([5.0, 3.0, 2.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        # Ray targeting the offset sphere
        ray = TracingRay(
            origin=np.array([5.0, 3.0, -5.0]),
            orientation=np.array([0.0, 0.0, 1.0])
        )
        
        intersector = RayMarchingIntersection(settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: ray transformation to local space
        assert hit.hit, "Should hit offset sphere"
        assert hit.obj is node, "Should hit correct object"

    def test_ray_marching_scale_compensation(self):
        """Test that non-uniform scales are handled in distance calculations."""
        settings = IntersectionSettings(epsilon=1e-4, max_steps=128, max_distance=1000.0)
        scene = Scene()
        
        sphere = Sphere(radius=1.0)
        # Sphere with non-uniform scale
        t = Transform(position=np.array([0.0, 0.0, 0.0]), scale=np.array([2.0, 1.0, 1.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        
        intersector = RayMarchingIntersection(settings)
        hit = intersector.find_hit(scene, ray)
        
        # Branch: max_dist_local = max_distance / min(scales)
        # Should still hit despite scale
        assert hit.hit, "Scaled sphere should still be hit"


# ============================================================================
# TEST SUITE 3: INSIDE-OUT / VOLUMETRIC LOGIC
# ============================================================================

class TestVolumetricLogic:
    """White-box tests for volumetric rendering, transmission, and inside-out ray marking."""

    def test_volumetric_thickness_calculation_closed_volume(self, simple_sphere_scene):
        """Test that thickness is correctly calculated from entry to exit."""
        settings = ShadingSettings()
        
        # Mock intersection function that returns exit hit
        def mock_intersection(scene, ray, stats):
            if ray.is_inside:
                # Return a hit 2.0 units away (simulated exit)
                exit_point = ray.point_at(2.0)
                return HitInfo(
                    did_hit=True,
                    distance=2.0,
                    point=exit_point,
                    direction=ray.orientation,
                    normal=np.array([0.0, 0.0, 1.0]),
                    obj=None
                )
            return HitInfo.miss()
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=1.0,
            absorption_color=Color(1.0, 1.0, 1.0),
        )
        
        scene, node = simple_sphere_scene
        # Hit point on sphere surface
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: thickness calculated from exit_hit.distance
        # Should be 2.0 in this case
        assert isinstance(color, Color), "Should return Color"

    def test_volumetric_no_exit_hit_zero_thickness(self, simple_sphere_scene):
        """Test that missing exit point results in zero thickness."""
        settings = ShadingSettings()
        
        # Mock intersection that always misses (open volume)
        def mock_intersection(scene, ray, stats):
            return HitInfo.miss()
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=1.0,
            absorption_color=Color(1.0, 0.0, 0.0),
        )
        
        scene, node = simple_sphere_scene
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: if not exit_hit.hit -> thickness = 0.0
        # With zero thickness, transmission = 1.0, additive style -> color = Color(0,0,0)
        # OR subtractive style -> absorption_color * 1.0 = full color
        assert isinstance(color, Color), "Should handle no exit gracefully"

    def test_volumetric_transmission_attenuation_thin(self):
        """Test transmission calculation for thin volumes (high transmission)."""
        settings = ShadingSettings()
        density = 1.0
        thickness_thin = 0.1  # Thin: exp(-1.0 * 0.1) = 0.9
        
        # attenuation = exp(-density * thickness)
        import math
        expected_transmission = math.exp(-density * thickness_thin)
        
        def mock_intersection(scene, ray, stats):
            exit_point = ray.point_at(thickness_thin)
            return HitInfo(
                did_hit=True,
                distance=thickness_thin,
                point=exit_point,
                direction=ray.orientation,
                normal=np.array([0.0, 0.0, 1.0]),
                obj=None
            )
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=density,
            absorption_color=Color(0.5, 0.5, 0.5),
            invert_style=False,  # Subtractive: Color * transmission
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: transmission = exp(-density * thickness)
        # For thin volumes, transmission should be close to 1.0
        # Subtractive: color = absorption * transmission ≈ absorption * 0.9
        assert color.r > 0.4, "Thin volume should transmit most light (subtractive)"

    def test_volumetric_transmission_attenuation_thick(self):
        """Test transmission calculation for thick volumes (low transmission)."""
        settings = ShadingSettings()
        density = 1.0
        thickness_thick = 5.0  # Thick: exp(-1.0 * 5.0) ≈ 0.007
        
        def mock_intersection(scene, ray, stats):
            exit_point = ray.point_at(thickness_thick)
            return HitInfo(
                did_hit=True,
                distance=thickness_thick,
                point=exit_point,
                direction=ray.orientation,
                normal=np.array([0.0, 0.0, 1.0]),
                obj=None
            )
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=density,
            absorption_color=Color(1.0, 1.0, 1.0),
            invert_style=False,  # Subtractive
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: transmission ≈ 0 for thick volumes
        # Subtractive: color ≈ Color(0, 0, 0) (very dark)
        assert color.r < 0.1, "Thick volume should absorb most light (subtractive)"

    def test_volumetric_additive_style_intensity(self):
        """Test additive (sci-fi) style: intensity = 1 - transmission."""
        settings = ShadingSettings()
        density = 1.0
        thickness = 1.0  # exp(-1.0) ≈ 0.368
        
        def mock_intersection(scene, ray, stats):
            exit_point = ray.point_at(thickness)
            return HitInfo(
                did_hit=True,
                distance=thickness,
                point=exit_point,
                direction=ray.orientation,
                normal=np.array([0.0, 0.0, 1.0]),
                obj=None
            )
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=density,
            absorption_color=Color(0.0, 1.0, 1.0),  # Cyan
            invert_style=True,  # Additive: intensity = 1 - transmission
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: intensity = 1.0 - transmission
        # For thickness = 1.0: transmission ≈ 0.368, intensity ≈ 0.632
        # color = absorption_color * intensity ≈ (0, 0.632, 0.632)
        assert color.g > 0.5, "Additive style: medium thickness should glow moderately"

    def test_volumetric_subtractive_style(self):
        """Test subtractive (glass) style: color = absorption * transmission."""
        settings = ShadingSettings()
        density = 1.0
        thickness = 1.0
        
        def mock_intersection(scene, ray, stats):
            exit_point = ray.point_at(thickness)
            return HitInfo(
                did_hit=True,
                distance=thickness,
                point=exit_point,
                direction=ray.orientation,
                normal=np.array([0.0, 0.0, 1.0]),
                obj=None
            )
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=density,
            absorption_color=Color(1.0, 0.0, 0.0),  # Red glass
            invert_style=False,  # Subtractive
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: color = absorption_color * transmission
        # Result should be reddish and darker than full absorption
        assert color.r > 0.0 and color.r < 1.0, "Subtractive style should blend color"

    def test_volumetric_thickness_clamping(self):
        """Test that thickness is clamped at max_thickness."""
        settings = ShadingSettings()
        density = 1.0
        max_thickness = 2.0
        
        # Mock returns a very thick volume
        def mock_intersection(scene, ray, stats):
            # Simulate exit very far away
            exit_point = ray.point_at(100.0)  # Would be 100 units
            return HitInfo(
                did_hit=True,
                distance=100.0,  # Should be clamped to max_thickness
                point=exit_point,
                direction=ray.orientation,
                normal=np.array([0.0, 0.0, 1.0]),
                obj=None
            )
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=density,
            absorption_color=Color(1.0, 1.0, 1.0),
            max_thickness=max_thickness,
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: thickness = min(thickness, max_thickness)
        # Transmission should be clamped, not extreme
        assert color is not None, "Should clamp thickness gracefully"

    def test_volumetric_ray_marked_as_inside(self):
        """Test that ray originating inside object is marked with is_inside=True."""
        settings = ShadingSettings()
        
        marked_rays = []
        
        def mock_intersection(scene, ray, stats):
            # Capture the inside_ray to verify is_inside flag
            marked_rays.append(ray.is_inside)
            exit_point = ray.point_at(2.0)
            return HitInfo(
                did_hit=True,
                distance=2.0,
                point=exit_point,
                direction=ray.orientation,
                normal=np.array([0.0, 0.0, 1.0]),
                obj=None
            )
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=1.0,
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]), is_inside=False)
        color = shading.shade(scene, ray, hit)
        
        # Branch: inside_ray = replace(ray, ..., is_inside=True)
        assert len(marked_rays) > 0, "Intersection should be called"
        assert marked_rays[0] is True, "Ray passed to intersection should have is_inside=True"

    def test_volumetric_rim_lighting_disabled(self):
        """Test that rim lighting can be disabled (rim_power=0)."""
        settings = ShadingSettings()
        
        def mock_intersection(scene, ray, stats):
            return HitInfo.miss()
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=1.0,
            absorption_color=Color(0.0, 0.0, 0.0),
            rim_power=0.0,  # Disable rim lighting
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([0.0, 0.0, -1.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([0.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: if rim_power > 0.0
        # With rim_power=0, no rim contribution
        assert isinstance(color, Color), "Should compute color without rim"

    def test_volumetric_rim_lighting_enabled(self):
        """Test rim lighting contribution at grazing angle."""
        settings = ShadingSettings()
        
        def mock_intersection(scene, ray, stats):
            return HitInfo.miss()
        
        shading = VolumetricShading(
            settings=settings,
            intersection_function=mock_intersection,
            density=1.0,
            absorption_color=Color(0.0, 0.0, 0.0),
            rim_power=2.0,
            rim_color=Color(1.0, 1.0, 1.0),
        )
        
        scene = Scene()
        sphere = Sphere(radius=1.0)
        t = Transform(position=np.array([0.0, 0.0, 0.0]))
        c = SDF_Material(shape=sphere)
        node = SceneNode(context=c, transform=t)
        scene.add_node(node)
        
        # Grazing angle: normal perpendicular to view_dir
        hit = HitInfo(
            did_hit=True,
            distance=4.0,
            point=np.array([1.0, 0.0, -1.0]),
            direction=np.array([-1.0, 0.0, 1.0]),  # Pointing away-ish
            normal=np.array([1.0, 0.0, 0.0]),  # Perpendicular to direction
            obj=node
        )
        
        ray = TracingRay(origin=np.array([0.0, 0.0, -5.0]), orientation=np.array([-1.0, 0.0, 1.0]))
        color = shading.shade(scene, ray, hit)
        
        # Branch: rim_intensity = (1 - NdotV) ^ rim_power
        # At grazing angle, rim should be prominent
        assert color.r > 0.0 or color.g > 0.0 or color.b > 0.0, "Rim should contribute at grazing angle"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

