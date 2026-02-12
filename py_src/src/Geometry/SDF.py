import numpy as np
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

from src.Data.Transform import Transform
from src.Data.Ray import Ray
from .Operations import *
from .AABB import convert_bounds_to_corners, convert_bounds_to_corners_2d

class SignedDistanceFunction(ABC):
    """
    Abstract base class for Signed Distance Functions (SDFs).
    SDFs provide a way to represent 2D and 3D shapes implicitly.
    All shapes are considered to be centered at the origin in their local space.
    """

    @abstractmethod
    def get_distance(self, point: np.ndarray) -> float:
        """
        Compute the signed distance from the given point to the surface defined by the SDF.
        
        :param point: A 3D point as a numpy array.
        :return: The signed distance to the surface.
        """
        pass

class SignedDistanceGradient(ABC):
    """
    Abstract base class for Signed Distance Gradients (SDGs).
    SDGs provide the gradient of the signed distance function at a point.
    """

    @abstractmethod
    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the signed distance function at the given point.
        
        :param point: A 3D point as a numpy array.
        :return: The gradient vector as a numpy array.
        """
        pass

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Compute the normalized gradient vector at the given point on the surface.
        
        :param point: A 3D point on the surface as a numpy array.
        :return: The normal vector as a numpy array.
        """
        gradient = self.get_gradient(point)
        norm = np.linalg.norm(gradient)
        if norm > 0:
            return gradient / norm
        return np.array([0, 1, 0])

    def get_tangent_bitangent(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the tangent and bitangent vectors at the given point on the surface.
        
        :param point: A 3D point on the surface as a numpy array.
        :return: A tuple (tangent, bitangent) as numpy arrays.
        """
        normal = self.get_normal(point)
        # Create an arbitrary vector that is not parallel to the normal
        if abs(normal[0]) < abs(normal[1]):
            tangent = np.cross(normal, np.array([1, 0, 0]))
        else:
            tangent = np.cross(normal, np.array([0, 1, 0]))
        
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        
        return tangent, bitangent
    
class CorrespondingBoxCorners(ABC):
    """
    Abstract base class to define the corner points for Signed Distance Functions to improve performance.
    This is useful for quickly computing bounding boxes and spatial partitioning.
    
    2D shapes: return (4, 3)
    3D shapes: return (8, 3)

    Note: The corners should be defined in local space. And not all shapes will have a simple box.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        """
        Compute the points that encapsulates an object
        """
        raise NotImplementedError("Local corners not implemented for this shape.")
    
class SignedDistanceShape(SignedDistanceFunction, SignedDistanceGradient):
    """
    Abstract base class for shapes defined by Signed Distance Functions paired with their Gradients.

    This base contains a compatibility helper `get_ray_intersections()` which older tests
    expect to exist (it wraps the newer `ray_intersect` API).
    """
    
    @abstractmethod
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        """
        Compute the intersection of a ray with the shape.
        
        :param ray: The ray to test for intersection.
        :param max_t: The maximum distance to consider for intersection.
        :return: A list of distances along the ray where intersections occur. An empty list if no intersection.
        """
        pass

    def get_ray_intersections(self, ray: Ray, max_t: float = 1e30):
        """Compatibility wrapper expected by older code/tests.

        Returns None when there is no hit, or the list of hit distances when there are.
        """
        hits = self.ray_intersect(ray, max_t)
        if not hits:
            return None
        return hits

    @abstractmethod
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        """
        Compute the UV coordinates for texture mapping at the given point on the surface.
        
        :param point: A 3D point on the surface as a numpy array.
        :return: A tuple (u, v) representing the UV coordinates.
        """
        pass
    
    @property
    def is_closed(self) -> bool:
        """
        Indicates whether the shape is closed (encloses a volume).
        A closed shape has no holes or gaps.
        
        :return: True if the shape is closed, False otherwise.
        """
        return True
    
    @property
    def is_manifold(self) -> bool:
        """
        Indicates whether the shape is manifold.
        A manifold shape has a well-defined surface without holes or singularities.
        
        :return: True if the shape is manifold, False otherwise.
        """
        return True
    
    @property
    def is_convex(self) -> bool:
        """
        Indicates whether the shape is convex.
        A convex shape has no indentations; a line segment between any two points in the shape lies entirely within the shape.
        
        :return: True if the shape is convex, False otherwise.
        """
        return False
    
    def get_convex_hull(self) -> Optional[List[np.ndarray]]:
        """
        Compute the convex hull of the shape if applicable.
        
        :return: A list of points representing the convex hull, or None if not applicable.
        """
        return None
    
    @property
    def dimension(self) -> int:
        """
        Returns the dimensionality of the shape (2D or 3D).
        
        :return: 2 for 2D shapes, 3 for 3D shapes.
        """
        return 3

class SignedDistanceShape2D(SignedDistanceShape):
    """
    Abstract base class for 2D shapes defined by Signed Distance Functions and their gradients.
    Combines both SDF and SDG functionalities for 2D space.
    """
    @property
    def dimension(self) -> int:
        return 2

    @property
    def perimeter(self) -> float:
        """
        Compute the perimeter of the shape (for 2D shapes).
        
        :return: The perimeter as a float.
        """
        raise NotImplementedError("Perimeter computation not implemented for this shape.")
    
    @property
    def area(self) -> float:
        """
        Compute the surface area of the shape.
        
        :return: The surface area as a float.
        """
        raise NotImplementedError("Area computation not implemented for this shape.")

class SignedDistanceShape3D(SignedDistanceShape):
    """
    Abstract base class for 3D shapes defined by Signed Distance Functions and their gradients.
    Combines both SDF and SDG functionalities for 3D space.
    """
    @property
    def dimension(self) -> int:
        return 3

    @property
    def volume(self) -> float:
        """
        Compute the volume enclosed by the shape.
        
        :return: The volume as a float.
        """
        raise NotImplementedError("Volume computation not implemented for this shape.")
    
    @property
    def surface_area(self) -> float:
        """
        Compute the surface area of the shape.
        
        :return: The surface area as a float.
        """
        raise NotImplementedError("Surface area computation not implemented for this shape.")


class RoundedShape(SignedDistanceShape):
    def __init__(self, shape: SignedDistanceShape, radius: float = 0.1):
        self.shape = shape
        self.radius = radius
        
    def get_distance(self, point: np.ndarray) -> float:
        return self.shape.get_distance(point) - self.radius
        
    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        return self.shape.get_gradient(point)
        
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        return self.shape.get_uv(point)
        
    def get_local_corners(self, padding: float=1e-2):
        # Rounding expands the object by radius
        return self.shape.get_local_corners(padding + self.radius)

class OnionShape(SignedDistanceShape):
    pass

class ShapeExtrusion(SignedDistanceShape3D, CorrespondingBoxCorners):
    """
    A 3D shape created by extruding a 2D signed distance shape along the local Z-axis.
    The 2D shape is assumed to lie on the XY plane.
    """
    def __init__(self, shape_2d: SignedDistanceShape2D, height: float = 1.0):
        if not isinstance(shape_2d, CorrespondingBoxCorners):
            raise NotImplementedError("The shape does not inherit the bounding box logic")

        self.shape_2d = shape_2d

        self.half_height = height / 2.0

    def get_distance(self, point: np.ndarray) -> float:
        # Project 3D point to 2D plane (XY)
        # We map 3D (x, y, z) -> 2D (x, y)
        p_2d = point[:2]
        
        # Get distance to the 2D profile
        d_2d = self.shape_2d.get_distance(p_2d)
        
        # Calculate distance logic for extrusion along Z-AXIS
        # w.x = distance to 2D shape side (in XY plane)
        # w.y = distance to top/bottom cap (abs(z) - h/2)
        w = np.array([d_2d, abs(point[2]) - self.half_height])
        
        # SDF logic: interior distance (negative) + exterior distance (positive length)
        return min(max(w[0], w[1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0))

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical gradient (Finite Difference)
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        dz = self.get_distance(point + np.array([0, 0, h])) - self.get_distance(point - np.array([0, 0, h]))
        grad = np.array([dx, dy, dz])
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Sphere Tracing
        t = 0.0
        for _ in range(128):
            p = ray.point_at(t)
            d = self.get_distance(p)
            if d < 1e-4: # Hit threshold
                return [t]
            t += d
            if t > max_t:
                return []
        return []

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # U: Derived from the 2D profile (XY)
        # V: Derived from the height (Z)
        p_2d = point[:2]
        u, _ = self.shape_2d.get_uv(p_2d) 
        
        # Map height (Z) [-h/2, h/2] to [0, 1]
        v = (point[2] + self.half_height) / (self.half_height * 2)
        return u, v

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        local_2d = self.shape_2d.get_local_corners(padding)
        min_xy = local_2d[:, :2].min(axis=0)
        max_xy = local_2d[:, :2].max(axis=0)

        h = self.half_height + padding
        return convert_bounds_to_corners(
            np.array([min_xy[0], min_xy[1], -h]),
            np.array([max_xy[0], max_xy[1],  h])
        )

    @property
    def volume(self) -> float:
        return self.shape_2d.area * self.half_height * 2

    @property
    def surface_area(self) -> float:
        # 2 Caps + Side walls
        return (2 * self.shape_2d.area) + (self.shape_2d.perimeter * self.half_height * 2)

    @property
    def is_convex(self) -> bool:
        return self.shape_2d.is_convex

    def get_convex_hull(self) -> Optional[List[np.ndarray]]:
        hull_2d = self.shape_2d.get_convex_hull()
        if not hull_2d:
            return None
        
        points = []
        # Create 3D hull by capping the 2D hull at front (+Z) and back (-Z)
        for p in hull_2d:
            # p is [x, y, 0] from the 2D shape
            points.append(np.array([p[0], p[1],  self.half_height])) # Front cap
            points.append(np.array([p[0], p[1], -self.half_height])) # Back cap
        return points

class ShapeRevolution(SignedDistanceShape3D, CorrespondingBoxCorners):
    """
    A 3D shape created by revolving a 2D signed distance shape around the Y-axis.
    The 2D shape is assumed to be defined in the XY plane, where X represents the radius.
    """
    def __init__(self, shape_2d: SignedDistanceShape2D):
        if not isinstance(shape_2d, CorrespondingBoxCorners):
            raise NotImplementedError("The shape does not inherit the bounding box logic")

        self.shape_2d = shape_2d

    def get_distance(self, point: np.ndarray) -> float:
        # Convert 3D point (x, y, z) to Cylindrical coordinates (r, y)
        # We assume revolution around Y-axis, so Radius is distance in XZ plane
        r = np.linalg.norm(point[[0, 2]])
        p_2d = np.array([r, point[1]])
        
        return self.shape_2d.get_distance(p_2d)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical gradient (Finite Difference)
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        dz = self.get_distance(point + np.array([0, 0, h])) - self.get_distance(point - np.array([0, 0, h]))
        grad = np.array([dx, dy, dz])
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 0 else np.array([0.0, 1.0, 0.0])

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Sphere Tracing
        t = 0.0
        for _ in range(128):
            p = ray.point_at(t)
            d = self.get_distance(p)
            if d < 1e-4:
                return [t]
            t += d
            if t > max_t:
                return []
        return []

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # U: Rotational angle around Y axis
        u = (np.arctan2(point[2], point[0]) + np.pi) / (2 * np.pi)
        
        # V: Derived from the 2D profile's vertical mapping
        r = np.linalg.norm(point[[0, 2]])
        p_2d = np.array([r, point[1]])
        _, v_profile = self.shape_2d.get_uv(p_2d)
        
        return u, v_profile

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        local_2d = self.shape_2d.get_local_corners(padding)
        min_xy = local_2d[:, :2].min(axis=0)
        max_xy = local_2d[:, :2].max(axis=0)

        max_radius = max(abs(min_xy[0]), abs(max_xy[0])) + padding
        min_y = min_xy[1] - padding
        max_y = max_xy[1] + padding

        return convert_bounds_to_corners(
            np.array([-max_radius, min_y, -max_radius]),
            np.array([ max_radius, max_y,  max_radius])
        )   

    @property
    def volume(self) -> float:
        # Pappus's Theorem (V = 2 * pi * r_centroid * Area) would be ideal here,
        # but requires knowing the centroid of the 2D shape.
        return 0.0 

    @property
    def surface_area(self) -> float:
        return 0.0

    @property
    def is_convex(self) -> bool:
        return False

    def get_convex_hull(self) -> Optional[List[np.ndarray]]:
        return None

class ShapeCombination(SignedDistanceShape, CorrespondingBoxCorners):
    """
    Base class for binary operations between two SDF shapes (A and B).
    Handles transformation logic centrally so subclasses don't have to.
    """
    def __init__(self, shape_a: SignedDistanceShape, transform_a: Transform, 
                 shape_b: SignedDistanceShape, transform_b: Transform):
        if not (isinstance(shape_a, CorrespondingBoxCorners) or isinstance(shape_b, CorrespondingBoxCorners)):
            raise TypeError("Either shape does not inherit the bounding box logic")
        
        if shape_a.dimension != shape_b.dimension:
            raise TypeError("The combined shapes must be in the same dimensions")
        
        self.shape_a = shape_a
        self.transform_a = transform_a
        self.shape_b = shape_b
        self.transform_b = transform_b

    def get_distance(self, point: np.ndarray) -> float:
        """
        1. Transforms point into Local Space of Shape A and B.
        2. Gets distance.
        3. Scales distance back to World Space (crucial for correct SDFs).
        4. Combines results.
        """
        # Transform Point: World -> Local A
        local_p_a = self.transform_a.local_transform_point(point)
        dist_a = self.shape_a.get_distance(local_p_a)
        
        # Scale Correction: SDF distances must be scaled back to world units.
        # We use the minimum scale component to be conservative (avoids overstepping).
        scale_a = self.transform_a.scale
        if hasattr(scale_a, '__iter__'): 
            dist_a *= min(scale_a)
        else:
            dist_a *= scale_a

        # Transform Point: World -> Local B
        local_p_b = self.transform_b.local_transform_point(point)
        dist_b = self.shape_b.get_distance(local_p_b)
        
        scale_b = self.transform_b.scale
        if hasattr(scale_b, '__iter__'):
            dist_b *= min(scale_b)
        else:
            dist_b *= scale_b
        
        return self._combine_distances(float(dist_a), float(dist_b))

    def _combine_distances(self, dist_a: float, dist_b: float) -> float:
        raise NotImplementedError("Subclasses must implement _combine_distances")

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Transform points to local space to check distance
        p_a = self.transform_a.local_transform_point(point)
        d_a = self.shape_a.get_distance(p_a) * (min(self.transform_a.scale) if hasattr(self.transform_a.scale, '__iter__') else self.transform_a.scale)

        p_b = self.transform_b.local_transform_point(point)
        d_b = self.shape_b.get_distance(p_b) * (min(self.transform_b.scale) if hasattr(self.transform_b.scale, '__iter__') else self.transform_b.scale)

        if abs(d_a) < abs(d_b):
            # Gradient A in Local Space
            local_grad = self.shape_a.get_gradient(p_a)
            # Rotate Gradient to World Space (inverse transpose of rotation, or just rotation for orthogonal matrices)
            return self.transform_a.world_transform_direction(local_grad)
        else:
            # Gradient B in Local Space
            local_grad = self.shape_b.get_gradient(p_b)
            return self.transform_b.world_transform_direction(local_grad)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # Similar logic to gradient: find closest shape, map UV in its local space
        p_a = self.transform_a.local_transform_point(point)
        d_a = self.shape_a.get_distance(p_a)

        p_b = self.transform_b.local_transform_point(point)
        d_b = self.shape_b.get_distance(p_b)

        if abs(d_a) < abs(d_b):
            return self.shape_a.get_uv(p_a)
        else:
            return self.shape_b.get_uv(p_b)

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        """
        Returns the bounding box of the COMBINATION.
        This requires transforming the children's corners into this object's space.
        """
        # 1. Get local corners of children
        corners_a = self.shape_a.get_local_corners(padding)
        corners_b = self.shape_b.get_local_corners(padding)

        # 2. Transform children corners to Current Space (Local -> World)
        corners_a_transformed = np.array([self.transform_a.world_transform_point(p) for p in corners_a])
        corners_b_transformed = np.array([self.transform_b.world_transform_point(p) for p in corners_b])

        return self._combine_bounds(corners_a_transformed, corners_b_transformed)

    def _combine_bounds(self, corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
        """Default behavior: Union of bounds (for ShapeUnion and general case)"""
        all_pts = np.vstack([corners_a, corners_b])
        min_pt = all_pts.min(axis=0)
        max_pt = all_pts.max(axis=0)
        
        if self.shape_a.dimension == 2:
            return convert_bounds_to_corners_2d(min_pt, max_pt)
        return convert_bounds_to_corners(min_pt, max_pt)

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Standard sphere tracing
        t = 0.0
        MAX_STEPS = 128
        EPSILON = 1e-4

        for _ in range(MAX_STEPS):
            p = ray.origin + ray.direction * t
            dist = self.get_distance(p)
            
            if dist < EPSILON:
                return [t]
            
            t += dist
            if t > max_t:
                return []
        
        return []

class ShapeUnion(ShapeCombination):
    def _combine_distances(self, dist_a: float, dist_b: float) -> float:
        return op_union(dist_a, dist_b)

class ShapeIntersection(ShapeCombination):
    def _combine_distances(self, dist_a: float, dist_b: float) -> float:
        return op_intersect(dist_a, dist_b)

    def _combine_bounds(self, corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
        """
        Intersection Bounds Optimization.
        The resulting volume is contained within the intersection of the two AABBs.
        """
        min_a, max_a = corners_a.min(axis=0), corners_a.max(axis=0)
        min_b, max_b = corners_b.min(axis=0), corners_b.max(axis=0)

        # Intersect the two AABBs
        min_pt = np.maximum(min_a, min_b)
        max_pt = np.minimum(max_a, max_b)

        # If intersection is empty (min > max in any axis), return empty
        if np.any(min_pt > max_pt):
            return np.zeros((0, 3), dtype=np.float32)

        if self.shape_a.dimension == 2:
            return convert_bounds_to_corners_2d(min_pt, max_pt)
        return convert_bounds_to_corners(min_pt, max_pt)

class ShapeSubtraction(ShapeCombination):
    def _combine_distances(self, dist_a: float, dist_b: float) -> float:
        # Note: Order matters. Here we assume A - B
        return op_subtract(dist_a, dist_b)

    def _combine_bounds(self, corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
        """
        Subtraction Bounds.
        The result of (A - B) is always contained within A.
        We can ignore B's bounds for the bounding box.
        """
        min_pt = corners_a.min(axis=0)
        max_pt = corners_a.max(axis=0)
        
        if self.shape_a.dimension == 2:
            return convert_bounds_to_corners_2d(min_pt, max_pt)
        return convert_bounds_to_corners(min_pt, max_pt)

class ShapeSmoothUnion(ShapeCombination):
    def __init__(self, shape_a: SignedDistanceShape, transform_a: Transform, 
                 shape_b: SignedDistanceShape, transform_b: Transform, 
                 k: float = 0.5):
        super().__init__(shape_a, transform_a, shape_b, transform_b)
        self.k = k

    def _combine_distances(self, dist_a: float, dist_b: float) -> float:
        return op_smooth_union(dist_a, dist_b, self.k)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Override gradient for SmoothUnion.
        Analytic 'closest shape' gradient is wrong in the blending region.
        We use finite difference.
        """
        epsilon = 1e-4
        d = self.get_distance(point)
        dx = self.get_distance(point + np.array([epsilon, 0, 0])) - d
        dy = self.get_distance(point + np.array([0, epsilon, 0])) - d
        dz = self.get_distance(point + np.array([0, 0, epsilon])) - d
        
        grad = np.array([dx, dy, dz])
        norm = np.linalg.norm(grad)

        return grad / norm if norm > 0 else np.zeros(3)
    
class ShapeXor(ShapeCombination):
    def _combine_distances(self, dist_a: float, dist_b: float) -> float:
        return op_xor(dist_a, dist_b)

class Circle(SignedDistanceShape2D, CorrespondingBoxCorners):
    """
    A simple 2D circle shape defined by a signed distance function.
    Centered at the origin with a given radius.
    """

    def __init__(self, radius: float = 0.5):
        self.radius = radius

    def get_distance(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(point[:2]) - self.radius)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(point[:3])
        if dist == 0:
            return np.array([0.0, 1.0, 0])
        return point[:3] / dist

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Ray is in-plane, use 2D intersection
        origin_2d = ray.origin[:2]
        direction_2d = ray.direction[:2]

        # Ray-circle intersection in 2D
        origin_2d = ray.origin[:2]

        # Ensure direction is normalized
        direction_2d = ray.direction[:2] 
        dir_len = np.linalg.norm(direction_2d)
        if dir_len == 0: return []

        direction_2d /= dir_len
        a = 1.0 # Since direction is normalized
        b = 2 * np.dot(origin_2d, direction_2d)
        c = np.dot(origin_2d, origin_2d) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return []
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        hits = []
        if 0 < t1 < max_t: hits.append(t1)
        if 0 < t2 < max_t: hits.append(t2)
        
        return sorted(hits)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        angle = np.arctan2(point[1], point[0])
        u = (angle + np.pi) / (2 * np.pi)
        v = 0.5  # Circle has no height variation
        return u, v

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.radius + padding
        return convert_bounds_to_corners_2d(np.array([-r, -r]), np.array([r, r]))

    @property
    def perimeter(self) -> float:
        return 2 * np.pi * self.radius
    
    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2

    @property
    def is_convex(self) -> bool:
        return True

    def get_convex_hull(self, resolution: int = 16) -> List[np.ndarray]:
        # Approximate the convex hull with a polygon (e.g., 16 points)
        points = []

        for i in range(resolution):
            angle = (i / resolution) * 2 * np.pi
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            points.append(np.array([x, y, 0]))
        return points

class Rectangle(SignedDistanceShape2D, CorrespondingBoxCorners):
    """
    A simple 2D square shape defined by a signed distance function.
    Centered at the origin with a given half-size.
    """
    def __init__(self, size: np.ndarray = np.array([1.0, 1.0])):
        self.half_size = size / 2

    def get_distance(self, point: np.ndarray) -> float:
        dx = abs(point[0]) - self.half_size[0]
        dy = abs(point[1]) - self.half_size[1]
        outside_dist = np.maximum(dx, dy)
        inside_dist = np.minimum(np.maximum(dx, dy), 0)
        return outside_dist + inside_dist

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Gradient approximation for rectangles
        grad = np.zeros(3)
        if abs(point[0]) > self.half_size[0]:
            grad[0] = np.sign(point[0])
        if abs(point[1]) > self.half_size[1]:
            grad[1] = np.sign(point[1])
        norm = np.linalg.norm(grad)
        if norm > 0:
            return grad / norm
        return np.array([0.0, 1.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        origin_2d = ray.origin[:2]
        direction_2d = ray.direction[:2]
        
        # Handle small direction components to avoid div by zero
        inv_dir = np.zeros_like(direction_2d)
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / direction_2d
        
        tmin = (-self.half_size - origin_2d) * inv_dir
        tmax = (self.half_size - origin_2d) * inv_dir

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        if t_exit >= t_enter and t_enter < max_t:
            hits = []
            if t_enter > 0: hits.append(t_enter)
            if t_exit > 0: hits.append(t_exit)
            return sorted(list(set(hits))) # remove duplicates if corner hit
        
        return []

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        u = (point[0] + self.half_size[0]) / (2 * self.half_size[0])
        v = (point[1] + self.half_size[1]) / (2 * self.half_size[1])
        return u, v
    
    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.half_size + padding
        return convert_bounds_to_corners_2d(-r, r)
    
    @property
    def perimeter(self) -> float:
        return 4 * self.half_size[0] + 4 * self.half_size[1]

    @property
    def area(self) -> float:
        return 4 * self.half_size[0] * self.half_size[1]
    
    @property
    def is_convex(self) -> bool:
        return True
    
    def get_convex_hull(self) -> List[np.ndarray]:
        points = [
            np.array([-self.half_size[0], -self.half_size[1], 0]),
            np.array([ self.half_size[0], -self.half_size[1], 0]),
            np.array([ self.half_size[0],  self.half_size[1], 0]),
            np.array([-self.half_size[0],  self.half_size[1], 0])
        ]
        return points

class Square(Rectangle):
    """
    A simple 2D square shape defined by a signed distance function.
    Centered at the origin with a given half-size.
    """
    def __init__(self, size: float = 1.0):
        self.half_size = np.array([size / 2, size / 2])

class Triangle(SignedDistanceShape2D, CorrespondingBoxCorners):
    """
    A simple 2D triangle shape defined by a signed distance function.
    Defined by three vertices in 3D space (z=0)."""
    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        # Precompute edges and normal for winding order
        self.e0 = v1 - v0
        self.e1 = v2 - v1
        self.e2 = v0 - v2

    def get_distance(self, point: np.ndarray) -> float:
        # Inigo Quilez 2D Triangle SDF
        p = point[:2]
        p0, p1, p2 = self.v0[:2], self.v1[:2], self.v2[:2]
        
        e0 = p1 - p0
        e1 = p2 - p1
        e2 = p0 - p2
        
        v0 = p - p0
        v1 = p - p1
        v2 = p - p2
        
        pq0 = v0 - e0 * np.clip(np.dot(v0, e0) / np.dot(e0, e0), 0.0, 1.0)
        pq1 = v1 - e1 * np.clip(np.dot(v1, e1) / np.dot(e1, e1), 0.0, 1.0)
        pq2 = v2 - e2 * np.clip(np.dot(v2, e2) / np.dot(e2, e2), 0.0, 1.0)
        
        s = np.sign(e0[0]*e2[1] - e0[1]*e2[0])
        d = np.minimum(np.minimum(np.dot(pq0, pq0), np.dot(pq1, pq1)), np.dot(pq2, pq2))
        
        # Winding number / sign test
        cond0 = (p[1] >= p0[1]) != (p[1] >= p1[1]) and (p[0] <= (e0[0]) * (p[1] - p0[1]) / (e0[1]) + p0[0])
        cond1 = (p[1] >= p1[1]) != (p[1] >= p2[1]) and (p[0] <= (e1[0]) * (p[1] - p1[1]) / (e1[1]) + p1[0])
        cond2 = (p[1] >= p2[1]) != (p[1] >= p0[1]) and (p[0] <= (e2[0]) * (p[1] - p2[1]) / (e2[1]) + p2[0])
        
        # Check if inside
        if cond0 == cond1 == cond2: # Note: this is a simplified winding check, robust implementation varies
             return -np.sqrt(d)
        return np.sqrt(d)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Central difference approximation is safer for Triangle due to sharp corners
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        return np.array([dx, dy, 0]) / (2*h)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # Barycentric coordinates for UV mapping
        p = point[:2]
        v0, v1, v2 = self.v0[:2], self.v1[:2], self.v2[:2]
        
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if denom == 0:
            return 0.0, 0.0  # Degenerate triangle
        
        a = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
        b = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
        c = 1.0 - a - b
        
        # Map barycentric to UV space (simple linear mapping)
        u = a * 0.0 + b * 1.0 + c * 0.5  # Example UV mapping
        v = a * 0.0 + b * 0.0 + c * 1.0
        return u, v
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Möller–Trumbore intersection algorithm
        epsilon = 1e-8
        
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            return []  # Ray is parallel to triangle

        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return []

        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)

        if v < 0.0 or u + v > 1.0:
            return []

        t = f * np.dot(edge2, q)

        if epsilon < t < max_t:
            return [t]
            
        return []

    def get_local_corners(self, padding: float = 1e-4) -> np.ndarray:
        min_xy = np.minimum.reduce([self.v0[:2], self.v1[:2], self.v2[:2]]) - padding
        max_xy = np.maximum.reduce([self.v0[:2], self.v1[:2], self.v2[:2]]) + padding
        return convert_bounds_to_corners_2d(min_xy, max_xy)
    
    @property
    def area(self) -> float:
        return float(0.5 * np.linalg.norm(np.cross(self.v1-self.v0, self.v2-self.v0)))
    
    @property
    def perimeter(self) -> float:
        return float(np.linalg.norm(self.v1-self.v0) + 
                np.linalg.norm(self.v2-self.v1) + 
                np.linalg.norm(self.v0-self.v2))
    
class Ellipse(SignedDistanceShape2D, CorrespondingBoxCorners):
    """
    A simple 2D ellipse shape defined by a signed distance function.
    Centered at the origin with given radii along x and y axes.
    """
    def __init__(self, radius_x: float = 0.5, radius_y: float = 0.3):
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.radii = np.array([radius_x, radius_y])

    def get_distance(self, point: np.ndarray) -> float:
        # Robust iterative root finding for distance to ellipse
        p = np.abs(point[:2])
        ab = self.radii
        
        # Check for degenerate cases
        if p[0] > p[1]: 
            p = p[[1, 0]]
            ab = ab[[1, 0]]
            
        l = ab[1] * ab[1] - ab[0] * ab[0]
        m = ab[0] * p[0] / l
        n = ab[1] * p[1] / l
        t = np.clip((m + n - 1.0) / 2.0, 0.0, 1.0)
        
        for _ in range(3):
            xi = t * t + 1.0
            yi = t - 1.0
            f = xi * xi * m + yi * yi * n - xi * yi
            g = 4.0 * t * xi * m + 2.0 * yi * n - (xi + 3.0 * t * yi)
            dt = f / g if abs(g) > 1e-6 else 0.0
            t = np.clip(t - dt, 0.0, 1.0)
            
        closest = np.array([ab[0] * p[0] / (t * t + ab[0]), ab[1] * p[1] / (t + ab[1])])
        return float(np.linalg.norm(p - closest) * np.sign(np.linalg.norm(p / ab) - 1.0))

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Central difference approximation
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        res = np.array([dx, dy, 0])
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else np.array([0.0, 0.0, 0.0])

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        ox = ray.origin[0] / self.radius_x
        oy = ray.origin[1] / self.radius_y
        dx = ray.direction[0] / self.radius_x
        dy = ray.direction[1] / self.radius_y
        
        a = dx**2 + dy**2
        b = 2 * (ox * dx + oy * dy)
        c = ox**2 + oy**2 - 1.0
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return []
            
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        hits = []
        if 0 < t1 < max_t: hits.append(t1)
        if 0 < t2 < max_t: hits.append(t2)
        
        return sorted(hits)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # UV mapping based on angle
        u = (np.arctan2(point[1] * self.radius_x, point[0] * self.radius_y) + np.pi) / (2 * np.pi)
        v = 0.5
        return u, v
    
    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.radii + padding
        return convert_bounds_to_corners_2d(-r, r)

    @property
    def perimeter(self) -> float:
        # Ramanujan approximation 2
        a, b = self.radius_x, self.radius_y
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

    @property
    def area(self) -> float:
        return np.pi * self.radius_x * self.radius_y
    
class Plane(SignedDistanceShape3D, CorrespondingBoxCorners):
    def __init__(self, normal: np.ndarray = np.array([0, 1, 0]), d: float = 0.0):
        # Plane equation: dot(p, n) + d = 0
        self.normal = normal / np.linalg.norm(normal)
        self.d = d

    def get_distance(self, point: np.ndarray) -> float:
        return np.dot(point, self.normal) + self.d

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        denom = np.dot(self.normal, ray.direction)
        
        if abs(denom) < 1e-6:
            return []
        
        t = -(np.dot(self.normal, ray.origin) + self.d) / denom
        if 0 < t < max_t:
            return [t]
        
        return []

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # Planar projection based on dominant axis
        n = np.abs(self.normal)
        if n[0] > n[1] and n[0] > n[2]:
            return point[1], point[2]
        elif n[1] > n[0] and n[1] > n[2]:
            return point[0], point[2]
        else:
            return point[0], point[1]
        
    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        R = 1e6  # Scene-scale bound
        return convert_bounds_to_corners(
            np.array([-R, -R, -R]),
            np.array([ R,  R,  R])
        )

class Moon(SignedDistanceShape2D):
    def __init__(self, radius_a: float = 0.5, radius_b: float = 0.3, offset: float = 0.2):
        self.radius_a = radius_a
        self.radius_b = radius_b
        self.offset = offset

    def get_distance(self, point: np.ndarray) -> float:
        # Simplified moon shape (two circles)
        d1 = np.linalg.norm(point - np.array([self.offset, 0])) - self.radius_a
        d2 = np.linalg.norm(point - np.array([-self.offset, 0])) - self.radius_b
        return max(float(d1), float(d2))

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        res = np.array([dx, dy, 0])
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Standard sphere tracing
        t = 0.0
        MAX_STEPS = 128
        EPSILON = 1e-4

        for _ in range(MAX_STEPS):
            p = ray.origin + ray.direction * t
            dist = self.get_distance(p)
            
            if dist < EPSILON:
                return [t]
            
            t += dist
            if t > max_t:
                return []
        
        return []
        
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        return 0, 0

class Cross(SignedDistanceShape2D):
    def __init__(self, size: float = 0.5, thickness: float = 0.2):
        self.size = size
        self.thickness = thickness
    
    def get_distance(self, point: np.ndarray) -> float:
        # Cross shape SDF
        dx = abs(point[0])
        dy = abs(point[1])
        qx = dx - self.thickness
        qy = dy - self.size
        return min(max(qx, qy), 0.0) + np.linalg.norm(np.maximum(np.array([qx, qy]), 0.0))

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        res = np.array([dx, dy, 0])
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Standard sphere tracing
        t = 0.0
        MAX_STEPS = 128
        EPSILON = 1e-4

        for _ in range(MAX_STEPS):
            p = ray.origin + ray.direction * t
            dist = self.get_distance(p)
            
            if dist < EPSILON:
                return [t]
            
            t += dist
            if t > max_t:
                return []
        
        return []
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        return 0, 0

class Gear(SignedDistanceShape2D):
    def __init__(self, radius: float = 0.5, teeth: int = 8, tooth_depth: float = 0.1):
        self.radius = radius
        self.teeth = teeth
        self.tooth_depth = tooth_depth
    
    def get_distance(self, point: np.ndarray) -> float:
        # Gear shape SDF
        angle = np.arctan2(point[1], point[0])
        r = np.linalg.norm(point)
        tooth_angle = (angle + np.pi) / (2 * np.pi) * self.teeth
        tooth_mod = abs(tooth_angle - round(tooth_angle))
        gear_radius = self.radius + self.tooth_depth * (1 - tooth_mod * 2)
        return r - gear_radius

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        res = np.array([dx, dy, 0])
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Standard sphere tracing
        t = 0.0
        MAX_STEPS = 128
        EPSILON = 1e-4

        for _ in range(MAX_STEPS):
            p = ray.origin + ray.direction * t
            dist = self.get_distance(p)
            
            if dist < EPSILON:
                return [t]
            
            t += dist
            if t > max_t:
                return []
        
        return []
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        return 0, 0

class Star(SignedDistanceShape2D):
    def __init__(self, points: int = 5, inner_radius: float = 0.2, outer_radius: float = 0.5):
        self.points = points
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def get_distance(self, point: np.ndarray) -> float:
        # Star shape SDF
        angle = np.arctan2(point[1], point[0])
        r = np.linalg.norm(point)
        sector_angle = 2 * np.pi / self.points
        sector_index = int((angle + np.pi) / sector_angle) % self.points
        inner_sector_angle = sector_index * sector_angle
        outer_sector_angle = (sector_index + 1) * sector_angle
        # Calculate distance to nearest edge of the star's point
        d1 = r - self.outer_radius
        d2 = r - self.inner_radius
        return min(float(d1), float(d2))

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        res = np.array([dx, dy, 0])
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Standard sphere tracing
        t = 0.0
        MAX_STEPS = 128
        EPSILON = 1e-4

        for _ in range(MAX_STEPS):
            p = ray.origin + ray.direction * t
            dist = self.get_distance(p)
            
            if dist < EPSILON:
                return [t]
            
            t += dist
            if t > max_t:
                return []
        
        return []
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        return 0, 0

class RegularPolygon(SignedDistanceShape2D):
    def __init__(self, sides: int = 6, radius: float = 0.5):
        self.sides = sides
        self.radius = radius

    def get_distance(self, point: np.ndarray) -> float:
        angle = np.arctan2(point[1], point[0])
        r = np.linalg.norm(point)
        sector_angle = 2 * np.pi / self.sides
        sector_index = int((angle + np.pi) / sector_angle) % self.sides
        inner_sector_angle = sector_index * sector_angle
        outer_sector_angle = (sector_index + 1) * sector_angle
        # Calculate distance to nearest edge of the polygon's side
        d1 = r - self.radius
        d2 = r - self.radius * np.cos(np.pi / self.sides)
        return min(d1, d2)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        res = np.array([dx, dy, 0])
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Standard sphere tracing
        t = 0.0
        MAX_STEPS = 128
        EPSILON = 1e-4

        for _ in range(MAX_STEPS):
            p = ray.origin + ray.direction * t
            dist = self.get_distance(p)
            
            if dist < EPSILON:
                return [t]
            
            t += dist
            if t > max_t:
                return []
        
        return []
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        return 0, 0

class Polygon(SignedDistanceShape2D):
    pass

class Heart(SignedDistanceShape2D):
    def __init__(self, size: float = 0.5):
        self.size = size

    def get_distance(self, point: np.ndarray) -> float:
        x = point[0] / self.size
        y = point[1] / self.size
        a = x**2 + y**2 - 1
        return (a**3 - x**2 * y**3) * self.size

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation
        h = 1e-4
        dx = self.get_distance(point + np.array([h, 0, 0])) - self.get_distance(point - np.array([h, 0, 0]))
        dy = self.get_distance(point + np.array([0, h, 0])) - self.get_distance(point - np.array([0, h, 0]))
        res = np.array([dx, dy, 0])
        norm = np.linalg.norm(res)
        return res / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        if abs(ray.direction[2]) >= 1e-6:
            # Ray is not in-plane, no intersection
            return []
        
        # Standard sphere tracing
        t = 0.0
        MAX_STEPS = 128
        EPSILON = 1e-4

        for _ in range(MAX_STEPS):
            p = ray.origin + ray.direction * t
            dist = self.get_distance(p)
            
            if dist < EPSILON:
                return [t]
            
            t += dist
            if t > max_t:
                return []
        
        return []
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        return 0, 0

class Sphere(SignedDistanceShape3D, CorrespondingBoxCorners):
    def __init__(self, radius: float = 0.5):
        self.radius = radius

    def get_distance(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(point) - self.radius)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(point)
        if d > 0:
            return point / d
        return np.array([0, 1, 0])

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        oc = ray.origin
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return []
            
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        hits = []
        if 0 < t1 < max_t: hits.append(t1)
        if 0 < t2 < max_t: hits.append(t2)
        return sorted(hits)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # Spherical coordinates
        p = point / self.radius
        u = 0.5 + (np.arctan2(p[2], p[0])) / (2 * np.pi)
        v = 0.5 - (np.arcsin(p[1])) / np.pi
        return u, v
    
    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.radius + padding
        return convert_bounds_to_corners(np.array([-r, -r, -r]), np.array([r, r, r]))

    @property
    def volume(self) -> float:
        return (4/3) * np.pi * self.radius**3

    @property
    def surface_area(self) -> float:
        return 4 * np.pi * self.radius**2

class Cube(SignedDistanceShape3D, CorrespondingBoxCorners):
    def __init__(self, size: float = 1.0):
        self.half_size = size / 2

    def get_distance(self, point: np.ndarray) -> float:
        q = np.abs(point) - self.half_size
        return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation for robustness near edges
        h = 1e-4
        x = self.get_distance(point + np.array([h,0,0])) - self.get_distance(point - np.array([h,0,0]))
        y = self.get_distance(point + np.array([0,h,0])) - self.get_distance(point - np.array([0,h,0]))
        z = self.get_distance(point + np.array([0,0,h])) - self.get_distance(point - np.array([0,0,h]))
        return np.array([x,y,z]) / (2*h)

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        inv_dir = np.zeros_like(ray.direction)
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / ray.direction
            
        t_min_vec = (-self.half_size - ray.origin) * inv_dir
        t_max_vec = (self.half_size - ray.origin) * inv_dir
        
        t1 = np.minimum(t_min_vec, t_max_vec)
        t2 = np.maximum(t_min_vec, t_max_vec)
        
        t_enter = np.max(t1)
        t_exit = np.min(t2)
        
        if t_exit >= t_enter and t_enter < max_t:
            hits = []
            if t_enter > 0: hits.append(t_enter)
            if t_exit > 0: hits.append(t_exit)
            return sorted(list(set(hits)))
            
        return []

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # Box mapping
        p = point
        abs_p = np.abs(p)
        if abs_p[0] >= abs_p[1] and abs_p[0] >= abs_p[2]:
            return (p[2]/self.half_size + 1)/2, (p[1]/self.half_size + 1)/2
        elif abs_p[1] >= abs_p[0] and abs_p[1] >= abs_p[2]:
             return (p[0]/self.half_size + 1)/2, (p[2]/self.half_size + 1)/2
        else:
             return (p[0]/self.half_size + 1)/2, (p[1]/self.half_size + 1)/2

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.half_size + padding
        return convert_bounds_to_corners(np.array([-r, -r, -r]), np.array([r, r, r]))

    @property
    def volume(self) -> float:
        return (self.half_size * 2) ** 3

class Cylinder(SignedDistanceShape3D, CorrespondingBoxCorners):
    def __init__(self, radius: float = 0.5, height: float = 1.0):
        self.radius = radius
        self.height = height

    def get_distance(self, point: np.ndarray) -> float:
        d = np.abs(np.linalg.norm(point[[0, 2]]) - self.radius)
        # Finite cylinder logic: max(horizontal_dist, vertical_dist)
        d_vec = np.array([np.linalg.norm(point[[0, 2]]) - self.radius, abs(point[1]) - self.height/2])
        return min(max(d_vec[0], d_vec[1]), 0.0) + np.linalg.norm(np.maximum(d_vec, 0.0))

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Finite difference
        h = 1e-4
        x = self.get_distance(point + np.array([h,0,0])) - self.get_distance(point - np.array([h,0,0]))
        y = self.get_distance(point + np.array([0,h,0])) - self.get_distance(point - np.array([0,h,0]))
        z = self.get_distance(point + np.array([0,0,h])) - self.get_distance(point - np.array([0,0,h]))
        return np.array([x,y,z]) / (2*h)

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Analytic Finite Cylinder Intersection
        hits = []
        ro = ray.origin
        rd = ray.direction
        
        # 1. Intersect Infinite Cylinder (x^2 + z^2 = r^2)
        # ignore y component for quadratic
        oc_xz = ro[[0, 2]]
        rd_xz = rd[[0, 2]]
        
        a = np.dot(rd_xz, rd_xz)
        if a > 0:
            b = 2.0 * np.dot(oc_xz, rd_xz)
            c = np.dot(oc_xz, oc_xz) - self.radius**2
            disc = b*b - 4*a*c
            
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / (2*a)
                t2 = (-b + sqrt_disc) / (2*a)
                
                # Check bounds for t1 (height check)
                y1 = ro[1] + t1 * rd[1]
                if abs(y1) <= self.height / 2:
                    if 0 < t1 < max_t: hits.append(t1)

                # Check bounds for t2
                y2 = ro[1] + t2 * rd[1]
                if abs(y2) <= self.height / 2:
                    if 0 < t2 < max_t: hits.append(t2)

        # 2. Intersect Caps (Planes at y = +/- h/2)
        if abs(rd[1]) > 1e-6:
            for sign in [-1, 1]:
                cap_y = sign * self.height / 2
                t_cap = (cap_y - ro[1]) / rd[1]
                if 0 < t_cap < max_t:
                    p_cap = ro + t_cap * rd
                    # Check if inside circle
                    if p_cap[0]**2 + p_cap[2]**2 <= self.radius**2:
                        hits.append(t_cap)

        return sorted(hits)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        u = (np.arctan2(point[2], point[0]) + np.pi) / (2*np.pi)
        v = (point[1] + self.height/2) / self.height
        return u, v

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.radius + padding
        h = self.height * 0.5 + padding
        return convert_bounds_to_corners(
            np.array([-r, -h, -r]),
            np.array([ r,  h,  r])
        )

    @property
    def volume(self) -> float:
        return np.pi * self.radius**2 * self.height

    @property
    def surface_area(self) -> float:
        # 2 circles + side area
        return 2 * np.pi * self.radius * (self.radius + self.height)

class Pyramid(SignedDistanceShape3D, CorrespondingBoxCorners):
    """
    A simple 3D pyramid shape defined by a signed distance function.
    Centered at the origin with a square base and a given height.
    """
    def __init__(self, base_size: float = 1.0, height: float = 1.0):
        self.base_half_size = base_size / 2
        self.height = height

    def get_distance(self, point: np.ndarray) -> float:
        # SDF for Pyramid with base center at (0,-h/2,0) and apex at (0,h/2,0)
        # Shift point so base is at y=0 for calculation, then shift back
        # Actually, simpler to treat apex as origin for calculation or use IQ approach
        
        # Standard IQ Pyramid: Base half-width 'h', Height '1' (normalized).
        # Adjusting variables to match our dimensions.
        p = point.copy()
        
        # Shift so apex is at y = height, base is at y = 0
        p[1] += self.height / 2
        
        # Symmetry
        p = np.array([abs(p[0]), p[1], abs(p[2])])
        if p[0] > p[2]: p[0], p[2] = p[2], p[0]
        p[0] -= self.base_half_size
        p[2] -= self.base_half_size
        
        q = np.array([p[0], p[1] - self.height, p[2]])
        
        # Normal of the side plane
        # m^2 = h^2 + b^2
        m2 = self.height**2 + self.base_half_size**2
        
        # Project point
        k = np.dot(q, np.array([self.height, self.base_half_size, 0])) / m2
        closest_on_slope = q - k * np.array([self.height, self.base_half_size, 0])
        
        dist_slope = np.linalg.norm(closest_on_slope) * np.sign(k)
        
        # Max of slope distance and base distance
        # Approximate check for bounding
        dist_base = -p[1] # inside base
        
        # Simplified distance bound (exact Euclidean is complex near corners)
        # Using a bounding composite
        dx = np.abs(point[0]) - self.base_half_size
        dz = np.abs(point[2]) - self.base_half_size
        dy = point[1] + self.height/2
        
        # If outside base box
        if dx > 0 or dz > 0:
            return float(min(np.linalg.norm(point - np.array([0, self.height/2, 0])),
                       np.linalg.norm(np.array([max(dx,0), dy, max(dz,0)]))))

        # Approximate signed distance for interior
        return max(dist_slope, abs(point[1]) - self.height/2)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        h = 1e-4
        x = self.get_distance(point + np.array([h,0,0])) - self.get_distance(point - np.array([h,0,0]))
        y = self.get_distance(point + np.array([0,h,0])) - self.get_distance(point - np.array([0,h,0]))
        z = self.get_distance(point + np.array([0,0,h])) - self.get_distance(point - np.array([0,0,h]))
        return np.array([x,y,z]) / (2*h)

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Decompose pyramid into 5 planar shapes: 1 square base, 4 triangular sides
        hits = []
        bs = self.base_half_size
        h = self.height
        
        # Vertices
        apex = np.array([0, h/2, 0])
        p0 = np.array([-bs, -h/2, -bs])
        p1 = np.array([ bs, -h/2, -bs])
        p2 = np.array([ bs, -h/2,  bs])
        p3 = np.array([-bs, -h/2,  bs])

        # Helper to intersect a single triangle
        def intersect_tri(v0, v1, v2):
            edge1 = v1 - v0
            edge2 = v2 - v0
            h_vec = np.cross(ray.direction, edge2)
            det = np.dot(edge1, h_vec)
            if -1e-8 < det < 1e-8: return
            inv_det = 1.0 / det
            s = ray.origin - v0
            u = inv_det * np.dot(s, h_vec)
            if u < 0.0 or u > 1.0: return
            q = np.cross(s, edge1)
            v = inv_det * np.dot(ray.direction, q)
            if v < 0.0 or u + v > 1.0: return
            t = inv_det * np.dot(edge2, q)
            if 1e-4 < t < max_t:
                hits.append(t)

        # 4 Sides
        intersect_tri(apex, p0, p1) # Front
        intersect_tri(apex, p1, p2) # Right
        intersect_tri(apex, p2, p3) # Back
        intersect_tri(apex, p3, p0) # Left

        # Base (Square treated as two triangles or simple plane check)
        # Plane y = -h/2
        if abs(ray.direction[1]) > 1e-8:
            t_base = (-h/2 - ray.origin[1]) / ray.direction[1]
            if 1e-4 < t_base < max_t:
                p_base = ray.origin + t_base * ray.direction
                if -bs <= p_base[0] <= bs and -bs <= p_base[2] <= bs:
                    hits.append(t_base)

        return sorted(hits)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # Planar map from top
        u = (point[0] / (2 * self.base_half_size)) + 0.5
        v = (point[2] / (2 * self.base_half_size)) + 0.5
        return u, v

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.base_half_size + padding
        h = self.height * 0.5 + padding
        return convert_bounds_to_corners(
            np.array([-r, -h, -r]),
            np.array([ r,  h,  r])
        )

    @property
    def volume(self) -> float:
        base_area = (self.base_half_size * 2) ** 2
        return (1/3) * base_area * self.height

    @property
    def surface_area(self) -> float:
        base_width = self.base_half_size * 2
        base_area = base_width ** 2
        slant_height = np.sqrt(self.base_half_size**2 + self.height**2)
        lateral_area = 2 * base_width * slant_height # 4 * (0.5 * b * s)
        return base_area + lateral_area
    
class Cone(SignedDistanceShape3D, CorrespondingBoxCorners):
    """
    A simple 3D cone shape defined by a signed distance function.
    Centered at the origin with a given base radius and height.
    """
    def __init__(self, base_radius: float = 0.5, height: float = 1.0):
        self.base_radius = base_radius
        self.height = height

    def get_distance(self, point: np.ndarray) -> float:
        # Cone SDF (Inigo Quilez)
        # q = vec2(length(p.xz), p.y)
        q = np.array([np.linalg.norm(point[[0, 2]]), point[1]])
        
        # Tip is at (0, h/2), Base is at (0, -h/2)
        # Shift y so tip is at y=0 for calculation simplified
        h = self.height
        r = self.base_radius
        
        # Transform q so tip is at (0, h) relative to base at 0
        # For simplicity, we calculate vectors relative to tip
        # Tip: (0, h/2), Base: (r, -h/2) in 2D profile
        
        tip = np.array([0.0, h/2])
        base_corner = np.array([r, -h/2])
        
        # Vector from tip to base corner
        k = base_corner - tip
        w = q - tip
        
        # Clamp projection
        t = np.dot(w, k) / np.dot(k, k)
        t = np.clip(t, 0.0, 1.0)
        
        closest = tip + t * k
        d_side = np.linalg.norm(q - closest)
        
        # Sign test (is point inside?)
        # Cross product in 2D is just determinant check
        cross = w[0] * k[1] - w[1] * k[0]
        if cross > 0: d_side = -d_side # Inside cone slope
        
        # Cap at bottom (y = -h/2)
        d_base = -h/2 - q[1]
        
        # Combined intersection
        # Simplified: max(d_side, d_base) isn't strictly Euclidean but works for raymarching.
        # Exact Euclidean requires checking regions. 
        
        # Region check:
        if q[1] > -h/2 and cross < 0: return float(d_side) # Outside slope
        if q[1] < -h/2 and np.linalg.norm(q - base_corner) < np.linalg.norm(q - np.array([0, -h/2])):
             # Outside corner
             return float(np.linalg.norm(q - base_corner))
             
        # Interior/Exterior bound
        # Using simple max approximation for robustness
        return max(d_side, abs(q[1] + h/2) if q[1] < -h/2 else -1.0)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        h = 1e-4
        x = self.get_distance(point + np.array([h,0,0])) - self.get_distance(point - np.array([h,0,0]))
        y = self.get_distance(point + np.array([0,h,0])) - self.get_distance(point - np.array([0,h,0]))
        z = self.get_distance(point + np.array([0,0,h])) - self.get_distance(point - np.array([0,0,h]))
        return np.array([x,y,z]) / (2*h)

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Analytic Cone Intersection (Apex at 0, h/2, 0; Base at 0, -h/2, 0)
        # Cone defined by x^2 + z^2 = (radius * (h/2 - y) / h)^2
        hits = []
        ro = ray.origin
        rd = ray.direction
        
        rh = self.base_radius / self.height
        y_off = self.height / 2
        
        # Coefficients for quadratic A*t^2 + B*t + C = 0
        # Expanded from (ox+tx)^2 + (oz+tz)^2 = rh^2 * (y_off - (oy+ty))^2
        
        # Simplify vars
        ro_x, ro_y, ro_z = ro
        rd_x, rd_y, rd_z = rd
        
        f = rh * rh
        # Precompute common terms
        k = y_off - ro_y
        
        A = rd_x**2 + rd_z**2 - f * rd_y**2
        B = 2 * (ro_x*rd_x + ro_z*rd_z + f * rd_y * k)
        C = ro_x**2 + ro_z**2 - f * k**2
        
        # Solve Quadratic
        disc = B*B - 4*A*C
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            # Be careful with A being near zero (ray parallel to cone slope)
            if abs(A) > 1e-6:
                ts = [(-B - sqrt_disc) / (2*A), (-B + sqrt_disc) / (2*A)]
                for t in ts:
                    if 0 < t < max_t:
                        y = ro_y + t * rd_y
                        # Check height bounds (-h/2 to h/2)
                        if -self.height/2 <= y <= self.height/2:
                            hits.append(t)

        # Intersect Base Cap (y = -h/2)
        if abs(rd_y) > 1e-6:
            t_cap = (-self.height/2 - ro_y) / rd_y
            if 0 < t_cap < max_t:
                p_cap = ro + t_cap * rd
                if p_cap[0]**2 + p_cap[2]**2 <= self.base_radius**2:
                    hits.append(t_cap)
                    
        return sorted(hits)
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        u = (np.arctan2(point[2], point[0]) + np.pi) / (2 * np.pi)
        v = (point[1] + self.height/2) / self.height
        return u, v

    def get_local_corners(self, padding: float = 1e-2) -> np.ndarray:
        r = self.base_radius + padding
        h = self.height * 0.5 + padding
        return convert_bounds_to_corners(
            np.array([-r, -h, -r]),
            np.array([ r,  h,  r])
        )

    @property
    def volume(self) -> float:
        return (1/3) * np.pi * self.base_radius**2 * self.height

    @property
    def surface_area(self) -> float:
        slant_height = np.sqrt(self.base_radius**2 + self.height**2)
        return np.pi * self.base_radius * (self.base_radius + slant_height)

class Torus(SignedDistanceShape3D):
    def __init__(self, major_radius: float = 0.5, minor_radius: float = 0.2):
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def get_distance(self, point: np.ndarray) -> float:
        q = np.array([np.linalg.norm(point[[0, 2]]) - self.major_radius, point[1]])
        return float(np.linalg.norm(q) - self.minor_radius)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Analytic gradient
        # Project point onto the major ring
        xz_dist = np.linalg.norm(point[[0, 2]])
        if xz_dist == 0: return np.array([0, 1, 0])
        
        # Point on the major ring (center of the tube cross-section)
        p_ring = np.array([point[0], 0, point[2]]) * (self.major_radius / xz_dist)
        
        vec = point - p_ring
        dist = np.linalg.norm(vec)
        if dist == 0: return np.array([0, 1, 0])
        return vec / dist

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Analytic torus intersection requires solving a quartic equation.
        # Sphere tracing is often more robust numerically for Torii.
        # We perform a sphere trace, but ensure it returns a List[float].
        
        t = 0.0
        for _ in range(128):
            p = ray.point_at(t)
            # Distance field of torus
            q = np.array([np.linalg.norm(p[[0, 2]]) - self.major_radius, p[1]])
            d = float(np.linalg.norm(q) - self.minor_radius)
            
            if d < 1e-4:
                return [t]
            t += d
            if t > max_t:
                return []
        
        return []

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        u = (np.arctan2(point[2], point[0]) + np.pi) / (2*np.pi)
        
        # Calculate angle around the tube
        xz_len = np.linalg.norm(point[[0, 2]])
        p_ring_dist = xz_len - self.major_radius
        v = (np.arctan2(point[1], p_ring_dist) + np.pi) / (2*np.pi)
        return u, v
    
    @property
    def volume(self) -> float:
        return (np.pi * self.minor_radius**2) * (2 * np.pi * self.major_radius)

class Capsule(SignedDistanceShape3D):
    def __init__(self, radius: float = 0.2, height: float = 1.0):
        self.radius = radius
        self.height = height 
        self.a = np.array([0, -height/2, 0])
        self.b = np.array([0, height/2, 0])

    def get_distance(self, point: np.ndarray) -> float:
        pa = point - self.a
        ba = self.b - self.a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
        return float(np.linalg.norm(pa - ba * h) - self.radius)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Standard finite difference for simplicity
        h = 1e-4
        dx = self.get_distance(point + np.array([h,0,0])) - self.get_distance(point - np.array([h,0,0]))
        dy = self.get_distance(point + np.array([0,h,0])) - self.get_distance(point - np.array([0,h,0]))
        dz = self.get_distance(point + np.array([0,0,h])) - self.get_distance(point - np.array([0,0,h]))
        return np.array([dx, dy, dz]) / (2*h)

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Capsule = Finite Cylinder (radius r) + 2 Spheres (radius r) at ends
        hits = []
        ro = ray.origin
        rd = ray.direction
        
        # 1. Cylinder Part (between -h/2 and h/2 on Y)
        oc_xz = ro[[0, 2]]
        rd_xz = rd[[0, 2]]
        a = np.dot(rd_xz, rd_xz)
        
        if a > 0:
            b = 2.0 * np.dot(oc_xz, rd_xz)
            c = np.dot(oc_xz, oc_xz) - self.radius**2
            disc = b*b - 4*a*c
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t_cyl = [(-b - sqrt_disc) / (2*a), (-b + sqrt_disc) / (2*a)]
                for t in t_cyl:
                    if 0 < t < max_t:
                        y = ro[1] + t * rd[1]
                        # Strictly strictly inside the cylinder height
                        if abs(y) <= self.height / 2:
                            hits.append(t)

        # 2. Sphere Caps at (0, -h/2, 0) and (0, h/2, 0)
        for sign in [-1, 1]:
            center = np.array([0, sign * self.height / 2, 0])
            oc = ro - center
            
            # Sphere Intersection
            as_ = np.dot(rd, rd)
            bs = 2.0 * np.dot(oc, rd)
            cs = np.dot(oc, oc) - self.radius**2
            discs = bs*bs - 4*as_*cs
            
            if discs >= 0:
                sqrt_discs = np.sqrt(discs)
                t_s = [(-bs - sqrt_discs) / (2*as_), (-bs + sqrt_discs) / (2*as_)]
                for t in t_s:
                    if 0 < t < max_t:
                        # Check if this hit is on the "outer" hemisphere
                        # (y should be > h/2 for top cap, < -h/2 for bottom cap)
                        y = ro[1] + t * rd[1]
                        if (sign == 1 and y >= self.height/2) or (sign == -1 and y <= -self.height/2):
                            hits.append(t)
                            
        return sorted(hits)

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        u = (np.arctan2(point[2], point[0]) + np.pi) / (2*np.pi)
        v = (point[1] + self.height/2 + self.radius) / (self.height + 2*self.radius)
        return u, v

    @property
    def volume(self) -> float:
        cyl_vol = np.pi * self.radius**2 * self.height
        sphere_vol = (4/3) * np.pi * self.radius**3
        return cyl_vol + sphere_vol

    @property
    def surface_area(self) -> float:
        cyl_area = 2 * np.pi * self.radius * self.height
        sphere_area = 4 * np.pi * self.radius**2
        return cyl_area + sphere_area

class EllipticSphere(SignedDistanceShape3D, CorrespondingBoxCorners):
    def __init__(self, radius_x: float = 0.5, radius_y: float = 0.3, radius_z: float = 0.4):
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.radius_z = radius_z
        self.radii = np.array([radius_x, radius_y, radius_z])

    def get_distance(self, point: np.ndarray) -> float:
        # Approximate SDF for Ellipsoid
        p = point / self.radii
        return float(np.linalg.norm(p) - 1.0) * min(self.radii)

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Numerical approximation
        h = 1e-4
        x = self.get_distance(point + np.array([h,0,0])) - self.get_distance(point - np.array([h,0,0]))
        y = self.get_distance(point + np.array([0,h,0])) - self.get_distance(point - np.array([0,h,0]))
        z = self.get_distance(point + np.array([0,0,h])) - self.get_distance(point - np.array([0,0,h]))
        return np.array([x,y,z]) / (2*h)

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Ray-Ellipsoid intersection via quadratic solution
        ro = ray.origin / self.radii
        rd = ray.direction / self.radii
        
        a = np.dot(rd, rd)
        b = 2.0 * np.dot(ro, rd)
        c = np.dot(ro, ro) - 1.0
        disc = b*b - 4*a*c
        
        if disc < 0:
            return []
        
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        return []