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
        
        :param matrix: A 4x4 transformation matrix as a numpy array.
        :param padding: A small padding value to expand the AABB.
        :return: An AABB instance representing the transformed bounding box.
        """
        raise NotImplementedError("AABB transformation not implemented for this shape.")

class SignedDistanceShape(SignedDistanceFunction, SignedDistanceGradient):
    """
    Abstract base class for shapes defined by Signed Distance Functions paired with their Gradients.
    """
    
    @abstractmethod
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> float:
        """
        Compute the intersection of a ray with the shape.
        
        :param ray: The ray to test for intersection.
        :param max_t: The maximum distance to consider for intersection.
        :return: The distance to the intersection point, or infinity if no intersection.
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

class SignedDistanceShape2D(SignedDistanceShape):
    """
    Abstract base class for 2D shapes defined by Signed Distance Functions and their gradients.
    Combines both SDF and SDG functionalities for 2D space.
    """
    
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
    pass

class SignedDistanceShape3DRevolution(SignedDistanceShape3D):
    pass

class Circle(SignedDistanceShape2D, CorrespondingBoundingBox):
    """
    A simple 2D circle shape defined by a signed distance function.
    Centered at the origin with a given radius.
    """

    def __init__(self, radius: float):
        self.radius = radius

    def get_distance(self, point: np.ndarray) -> float:
        return np.linalg.norm(point[:2]) - self.radius

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(point[:3])
        if dist == 0:
            return np.array([0.0, 1.0, 0])
        return point[:3] / dist

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> float:
        # Ray-circle intersection in 2D
        origin_2d = ray.origin[:2]
        direction_2d = ray.direction[:2] / np.linalg.norm(ray.direction[:2])

        a = np.dot(direction_2d, direction_2d)
        b = 2 * np.dot(origin_2d, direction_2d)
        c = np.dot(origin_2d, origin_2d) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return float('inf')  # No intersection

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        if 0 < t1 < max_t:
            return t1
        if 0 < t2 < max_t:
            return t2
        return float('inf')

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        angle = np.arctan2(point[1], point[0])
        u = (angle + np.pi) / (2 * np.pi)
        v = 0.5  # Circle has no height variation
        return u, v
    
    def get_transformed_aabb(self, matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        r = self.radius + padding

        local_bounds = np.array([
            [-r, -r, 0], [r, -r, 0],
            [-r, r, 0],  [r, r, 0]
        ])
        world_bounds = AABB().transform_local_bounds(matrix, local_bounds)
        
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
    def is_closed(self) -> bool:
        return True
    
    @property
    def is_manifold(self) -> bool:
        return True

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

    def __init__(self, size: float):
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
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> float:
        # Ray-square intersection in 2D (AABB method)
        origin_2d = ray.origin[:2]
        direction_2d = ray.direction[:2] / np.linalg.norm(ray.direction[:2])

        tmin = (-self.half_size - origin_2d) / direction_2d
        tmax = (self.half_size - origin_2d) / direction_2d

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        if t_exit >= t_enter and t_exit > 0 and t_enter < max_t:
            return max(t_enter, 0.0)
        
        return float('inf')

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        u = (point[0] + self.half_size) / (2 * self.half_size)
        v = (point[1] + self.half_size) / (2 * self.half_size)
        return u, v
    
    def transform_aabb(self, matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        r = self.half_size + padding

        local_bounds = np.array([
            [-r, -r, 0], [r, -r, 0],
            [-r, r, 0],  [r, r, 0]
        ])
        world_bounds = AABB().transform_local_bounds(matrix, local_bounds)
        
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
    def is_closed(self) -> bool:
        return True

    @property
    def is_manifold(self) -> bool:
        return True

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

    def __init__(self, size: np.ndarray):
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
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> float:
        # Ray-square intersection in 2D (AABB method)
        origin_2d = ray.origin[:2]
        direction_2d = ray.direction[:2] / np.linalg.norm(ray.direction[:2])

        tmin = (-self.half_size - origin_2d) / direction_2d
        tmax = (self.half_size - origin_2d) / direction_2d

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        if t_exit >= t_enter and t_exit > 0 and t_enter < max_t:
            return max(t_enter, 0.0)
        
        return float('inf')

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        u = (point[0] + self.half_size) / (2 * self.half_size)
        v = (point[1] + self.half_size) / (2 * self.half_size)
        return u, v
    
    def transform_aabb(self, matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        r = self.half_size + padding

        local_bounds = np.array([
            [-r[0], -r[1], 0], [r[0], -r[1], 0],
            [-r[0], r[1], 0],  [r[0], r[1], 0]
        ])
        world_bounds = AABB().transform_local_bounds(matrix, local_bounds)
        
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
    def is_closed(self) -> bool:
        return True

    @property
    def is_manifold(self) -> bool:
        return True

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
    Defined by three vertices in 2D space.
    """

    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def get_distance(self, point: np.ndarray) -> float:
        # Placeholder implementation
        return float('inf')

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        # Placeholder implementation
        return np.array([0.0, 0.0])
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> float:
        # Placeholder implementation
        return float('inf')

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        # Placeholder implementation
        return 0.0, 0.0
    
    def transform_aabb(self, matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        # Placeholder implementation
        return AABB(np.zeros(3), np.zeros(3))
    
class Sphere(SignedDistanceShape3D, CorrespondingBoundingBox):
    """
    A simple 3D sphere shape defined by a signed distance function.
    Centered at the origin with a given radius.
    """

    def __init__(self, radius: float):
        self.radius = radius

    def get_distance(self, point: np.ndarray) -> float:
        raise NotImplementedError()

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> float:
        raise NotImplementedError()

    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError()

    def transform_aabb(self, matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        raise NotImplementedError()

    @property
    def volume(self) -> float:
        raise NotImplementedError()

    @property
    def surface_area(self) -> float:
        raise NotImplementedError()

class Cube(SignedDistanceShape3D, CorrespondingBoundingBox):
    """
    A simple 3D cube shape defined by a signed distance function.
    Centered at the origin with a given half-size.
    """
    def __init__(self, size: float):
        self.half_size = size / 2

    def get_distance(self, point: np.ndarray) -> float:
        raise NotImplementedError()
    
    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def ray_intersect(self, ray: Ray, max_t: float = 1e30) -> float:
        raise NotImplementedError()
    
    def get_uv(self, point: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError()
    
    def transform_aabb(self, matrix: np.ndarray, padding: float = 1e-4) -> AABB:
        raise NotImplementedError()
    
    @property
    def volume(self) -> float:
        raise NotImplementedError()
    
    @property
    def surface_area(self) -> float:
        raise NotImplementedError()