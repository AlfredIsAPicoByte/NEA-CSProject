from abc import ABC, abstractmethod
from pyclbr import Class
import numpy as np
from typing import TYPE_CHECKING, Optional, Union, List, Tuple

from src.Data.Ray import Ray
from .AABB import AABB

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
    
class CorrespondingBoundingBox(ABC):
    """
    Abstract base class to define an AABB for Signed Distance Functions to improve performance.
    Only applied to simple shapes.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        """
        Compute the axis-aligned bounding box (AABB) of the shape after applying a transformation matrix.
        
        :param transformation_matrix: A 4x4 transformation matrix as a numpy array.
        :param padding: A small padding value to expand the AABB.
        :return: An AABB instance representing the transformed bounding box.
        """
        raise NotImplementedError("AABB transformation not implemented for this shape.")

class SignedDistanceShape(SignedDistanceFunction, SignedDistanceGradient):
    """
    Abstract base class for shapes defined by Signed Distance Functions paired with their Gradients.
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

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        """
        Default AABB computation for SDF shapes.
        """
        r = 0.5 + padding

        local_bounds = np.array([
            [-r, -r, 0], [r, -r, 0],
            [-r, r, 0],  [r, r, 0]
        ])
        world_bounds = AABB.transform_local_bounds(transformation_matrix, local_bounds)
        
        min_point = np.min(world_bounds, axis=0)
        max_point = np.max(world_bounds, axis=0)
        return AABB(min_point, max_point)

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

class SignedDistanceShape3DExtrusion(SignedDistanceShape3D):
    """
    A 3D shape created by extruding a 2D signed distance shape along an axis.
    """
    def __init__(self, shape_2d: SignedDistanceShape2D, height: float = 1.0, axis: np.ndarray = np.array([0, 1, 0])):
        self.shape_2d = shape_2d
        self.height = height
        self.axis = axis / np.linalg.norm(axis)
    
    def get_distance(self, point: np.ndarray) -> float:
        raise NotImplementedError()
    
    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        raise NotImplementedError()
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError()
    
    @property
    def volume(self) -> float:
        raise NotImplementedError()
    
    @property
    def surface_area(self) -> float:
        raise NotImplementedError()
    
    @property
    def is_closed(self) -> bool:
        return self.shape_2d.is_closed
    
    @property
    def is_manifold(self) -> bool:
        return self.shape_2d.is_manifold
    
    @property
    def is_convex(self) -> bool:
        return self.shape_2d.is_convex
    
    def get_convex_hull(self) -> Optional[List[np.ndarray]]:
        return self.shape_2d.get_convex_hull()

class SignedDistanceShape3DRevolution(SignedDistanceShape3D):
    """
    A 3D shape created by revolving a 2D signed distance shape around an axis.   
    """
    def __init__(self, shape_2d: SignedDistanceShape2D, axis: np.ndarray = np.array([0, 1, 0])):
        self.shape_2d = shape_2d
        self.axis = axis / np.linalg.norm(axis)
    
    def get_distance(self, point: np.ndarray) -> float:
        raise NotImplementedError()
    
    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        raise NotImplementedError()
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError()
    
    @property
    def volume(self) -> float:
        raise NotImplementedError()
    
    @property
    def surface_area(self) -> float:
        raise NotImplementedError()
    
    @property
    def is_closed(self) -> bool:
        return self.shape_2d.is_closed
    
    @property
    def is_manifold(self) -> bool:
        return self.shape_2d.is_manifold
    
    @property
    def is_convex(self) -> bool:
        return self.shape_2d.is_convex
    
    def get_convex_hull(self) -> Optional[List[np.ndarray]]:
        return self.shape_2d.get_convex_hull()

class Circle(SignedDistanceShape2D, CorrespondingBoundingBox):
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

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        r = self.radius + padding

        local_bounds = np.array([
            [-r, -r, 0], [r, -r, 0],
            [-r, r, 0],  [r, r, 0]
        ])
        world_bounds = AABB.transform_local_bounds(transformation_matrix, local_bounds)

        min_point = np.min(world_bounds, axis=0)
        max_point = np.max(world_bounds, axis=0)
        return AABB(min_point, max_point)

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
    
class Square(SignedDistanceShape2D, CorrespondingBoundingBox):
    """
    A simple 2D square shape defined by a signed distance function.
    Centered at the origin with a given half-size.
    """
    def __init__(self, size: float = 1.0):
        self.half_size = size / 2

    def get_distance(self, point: np.ndarray) -> float:
        dx = abs(point[0]) - self.half_size
        dy = abs(point[1]) - self.half_size
        outside_dist = np.maximum(dx, dy)
        inside_dist = np.minimum(np.maximum(dx, dy), 0)
        return outside_dist + inside_dist

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Gradient approximation for square
        grad = np.zeros(2)
        if abs(point[0]) > self.half_size:
            grad[0] = np.sign(point[0])
        if abs(point[1]) > self.half_size:
            grad[1] = np.sign(point[1])
        norm = np.linalg.norm(grad)
        if norm > 0:
            return grad / norm
        return np.array([0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
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
        u = (point[0] + self.half_size) / (2 * self.half_size)
        v = (point[1] + self.half_size) / (2 * self.half_size)
        return u, v
    
    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        r = self.half_size + padding

        local_bounds = np.array([
            [-r, -r, 0], [r, -r, 0],
            [-r, r, 0],  [r, r, 0]
        ])
        world_bounds = AABB.transform_local_bounds(transformation_matrix, local_bounds)
        
        min_point = np.min(world_bounds, axis=0)
        max_point = np.max(world_bounds, axis=0)
        return AABB(min_point, max_point)
    
    @property
    def perimeter(self) -> float:
        return 8 * self.half_size

    @property
    def area(self) -> float:
        return (2 * self.half_size) ** 2

    @property
    def is_convex(self) -> bool:
        return True
    
    def get_convex_hull(self) -> List[np.ndarray]:
        points = [
            np.array([-self.half_size, -self.half_size, 0]),
            np.array([ self.half_size, -self.half_size, 0]),
            np.array([ self.half_size,  self.half_size, 0]),
            np.array([-self.half_size,  self.half_size, 0])
        ]
        return points

class Rectangle(SignedDistanceShape2D, CorrespondingBoundingBox):
    """
    A simple 2D square shape defined by a signed distance function.
    Centered at the origin with a given half-size.
    """
    def __init__(self, size: np.ndarray = np.array([1.0, 1.0])):
        self.half_size = size / 2

    def get_distance(self, point: np.ndarray) -> float:
        dx = abs(point[0]) - self.half_size
        dy = abs(point[1]) - self.half_size
        outside_dist = np.maximum(dx, dy)
        inside_dist = np.minimum(np.maximum(dx, dy), 0)
        return outside_dist + inside_dist

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Gradient approximation for square
        grad = np.zeros(2)
        if abs(point[0]) > self.half_size:
            grad[0] = np.sign(point[0])
        if abs(point[1]) > self.half_size:
            grad[1] = np.sign(point[1])
        norm = np.linalg.norm(grad)
        if norm > 0:
            return grad / norm
        return np.array([0.0, 1.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
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
    
    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        r = self.half_size + padding

        local_bounds = np.array([
            [-r[0], -r[1], 0], [r[0], -r[1], 0],
            [-r[0], r[1], 0],  [r[0], r[1], 0]
        ])
        world_bounds = AABB.transform_local_bounds(transformation_matrix, local_bounds)
        
        min_point = np.min(world_bounds, axis=0)
        max_point = np.max(world_bounds, axis=0)
        return AABB(min_point, max_point)
    
    @property
    def perimeter(self) -> float:
        return 4 * self.half_size[0] + 4 * self.half_size[1]

    @property
    def area(self) -> float:
        return (self.half_size[0] * self.half_size[1]) ** 2
    
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

class Triangle(SignedDistanceShape2D, CorrespondingBoundingBox):
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
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
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

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        # Transform the three vertices
        local_bounds = np.array([self.v0, self.v1, self.v2])
        world_bounds = AABB.transform_local_bounds(transformation_matrix, local_bounds)

        min_p = np.min(world_bounds, axis=0) - padding
        max_p = np.max(world_bounds, axis=0) + padding
        return AABB(min_p, max_p)
    
    @property
    def area(self) -> float:
        return float(0.5 * np.linalg.norm(np.cross(self.v1-self.v0, self.v2-self.v0)))
    
    @property
    def perimeter(self) -> float:
        return float(np.linalg.norm(self.v1-self.v0) + 
                np.linalg.norm(self.v2-self.v1) + 
                np.linalg.norm(self.v0-self.v2))
    
class Ellipse(SignedDistanceShape2D, CorrespondingBoundingBox):
    """
    A simple 2D ellipse shape defined by a signed distance function.
    Centered at the origin with given radii along x and y axes.
    """
    def __init__(self, radius_x: float = 0.5, radius_y: float = 0.3):
        self.radius_x = radius_x
        self.radius_y = radius_y

    def get_distance(self, point: np.ndarray) -> float:
        raise NotImplementedError()

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> List[float]:
        # Transform ray into unit circle space: scale O and D by (1/rx, 1/ry)
        # (ox/rx + t*dx/rx)^2 + (oy/ry + t*dy/ry)^2 = 1
        
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
        raise NotImplementedError()
    
    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        raise NotImplementedError()

    @property
    def perimeter(self) -> float:
        raise NotImplementedError()

    @property
    def area(self) -> float:
        raise NotImplementedError()
    
class Plane(SignedDistanceShape3D, CorrespondingBoundingBox):
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
        
        # Check if ray is not parallel to the plane
        if abs(denom) > 1e-6:
            t = -(np.dot(self.normal, ray.origin) + self.d) / denom
            if 0 <= t < max_t:
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

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        # A plane is infinite, returning a very large AABB
        inf = 1e10
        return AABB(np.array([-inf]*3), np.array([inf]*3))
    
class Sphere(SignedDistanceShape3D, CorrespondingBoundingBox):
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
    
    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        # Efficient sphere AABB transformation: Center translates, radius scales by max scale
        # Extract scale from matrix columns
        r = self.radius + padding

        local_bounds = np.array([
            [-r, -r, -r], [r, -r, -r],
            [-r, r, -r],  [r, r, -r],
            [-r, -r, r],  [r, -r, r],
            [-r, r, r],   [r, r, r]
        ])
        world_bounds = AABB.transform_local_bounds(transformation_matrix, local_bounds)
        
        min_point = np.min(world_bounds, axis=0)
        max_point = np.max(world_bounds, axis=0)
        return AABB(min_point, max_point)

    @property
    def volume(self) -> float:
        return (4/3) * np.pi * self.radius**3

    @property
    def surface_area(self) -> float:
        return 4 * np.pi * self.radius**2

class Cube(SignedDistanceShape3D, CorrespondingBoundingBox):
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

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        # Transform all 8 corners
        r = self.half_size + padding
        local_bounds = np.array([
            [-r,-r,-r], [r,-r,-r], [-r,r,-r], [r,r,-r],
            [-r,-r,r],  [r,-r,r],  [-r,r,r],  [r,r,r]
        ])
        world_bounds = AABB.transform_local_bounds(transformation_matrix, local_bounds)

        min_point = np.min(world_bounds, axis=0)
        max_point = np.max(world_bounds, axis=0)
        return AABB(min_point, max_point)

    @property
    def volume(self) -> float:
        return (self.half_size * 2) ** 3

class Cylinder(SignedDistanceShape3D, CorrespondingBoundingBox):
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

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        raise NotImplementedError()
    
    @property
    def volume(self) -> float:
        return np.pi * self.radius**2 * self.height

    @property
    def surface_area(self) -> float:
        raise NotImplementedError()

class Pyramid(SignedDistanceShape3D, CorrespondingBoundingBox):
    """
    A simple 3D pyramid shape defined by a signed distance function.
    Centered at the origin with a square base and a given height.
    """

    def __init__(self, base_size: float = 1.0, height: float = 1.0):
        self.base_half_size = base_size / 2
        self.height = height

    def get_distance(self, point: np.ndarray) -> float:
        raise NotImplementedError()

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        raise NotImplementedError()

    @property
    def volume(self) -> float:
        raise NotImplementedError()

    @property
    def surface_area(self) -> float:
        raise NotImplementedError()
    
class Cone(SignedDistanceShape3D, CorrespondingBoundingBox):
    """
    A simple 3D cone shape defined by a signed distance function.
    Centered at the origin with a given base radius and height.
    """

    def __init__(self, base_radius: float = 0.5, height: float = 1.0):
        self.base_radius = base_radius
        self.height = height

    def get_distance(self, point: np.ndarray) -> float:
        raise NotImplementedError()

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_transformed_aabb(self, transformation_matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        raise NotImplementedError()

    @property
    def volume(self) -> float:
        raise NotImplementedError()

    @property
    def surface_area(self) -> float:
        raise NotImplementedError()

class Torus(SignedDistanceShape3D, CorrespondingBoundingBox):
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

class Capsule(SignedDistanceShape3D, CorrespondingBoundingBox):
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