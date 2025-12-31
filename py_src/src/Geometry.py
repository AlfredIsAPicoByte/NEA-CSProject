from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from CommonUtils import ray_world_to_local, world_to_local_point, local_to_world_point, normal_local_to_world
from PrimaryStructures import Transform, Ray
from Luminance import PBRMaterial

# Base Class
class Shape(ABC):
    """
    Abstract base for all shapes.
    Provides transform, PBRMaterial, and naming.
    """
    def __init__(self, name: str = "Shape", transform: Optional["Transform"] = None, 
                 PBRMaterial: Optional["PBRMaterial"] = None, **kwargs):
        self.name = name
        self.transform = transform or Transform(np.zeros(3), np.zeros(3), np.ones(3))
        self.PBRMaterial = PBRMaterial
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def origin(self) -> np.ndarray:
        return self.transform.position
    
    @abstractmethod
    def signed_distance(self, point: np.ndarray) -> float:
        """Signed distance for points (negative inside, positive outside)."""
        raise NotImplementedError
    
    @abstractmethod
    def signed_distance(self, ray: Ray) -> float:
        """Signed distance for rays (negative inside, positive outside)."""
        raise NotImplementedError

    def get_distance(self, point: np.ndarray) -> float:
        """Unsigned distance to surface."""
        return abs(self.signed_distance(point))

    def check_point_inside(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        """Point is inside if signed distance < -epsilon."""
        return self.signed_distance(point) < -epsilon

    def check_point_on_surface(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        """Point is on surface if |signed distance| <= epsilon."""
        return abs(self.signed_distance(point)) <= epsilon

    def get_closest_point(self, point: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Use gradient descent to find closest point on surface.
        Fallback for shapes without closed-form solution.
        """
        current = np.array(point, dtype=float)
        step_size = 0.1
        
        for _ in range(max_iterations):
            dist = self.signed_distance(current)
            if abs(dist) < 1e-6:
                return current
            
            # Approximate gradient via finite differences
            eps = 1e-5
            grad = np.array([
                (self.signed_distance(current + np.array([eps, 0, 0])) - dist) / eps,
                (self.signed_distance(current + np.array([0, eps, 0])) - dist) / eps,
                (self.signed_distance(current + np.array([0, 0, eps])) - dist) / eps,
            ])
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            else:
                break
            
            current = current - dist * grad
        
        return current
    
    @abstractmethod
    def get_normal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        """Default: perpendicular to normal (requires override for custom behavior)."""
        raise NotImplementedError
    
    def get_binormal(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        """Default: cross(normal, tangent)."""
        normal = self.get_normal(point)
        tangent = self.get_tangent(point)
        return np.cross(normal, tangent) / (np.linalg.norm(np.cross(normal, tangent)) + bias)

    @abstractmethod
    def check_ray_intersection(self, ray: Ray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        raise NotImplementedError
    
    def get_min_uniform_scale(self):
        # Assuming you store scale as a Vector3 or (sx, sy, sz)
        # We need the smallest scale component to ensure we don't overstep.
        s = self.transform.scale
        return min(s.x, s.y, s.z)

    def inverse_transform_point(self, p_world):
        # Translate -> Rotate -> Scale (Inverse Order)
        # Usually handled by a Matrix4 inverse
        return np.dot(self.transform.inverse_matrix, np.append(p_world, 1.0))[:3]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """2 for 2D, 3 for 3D."""
        raise NotImplementedError

    def translate(self, offset: np.ndarray):
        self.transform.translate(offset, space="global")

    def rotate(self, angle: float, axis: np.ndarray):
        self.transform.rotate(angle, axis, space="global")

    def enlarge(self, factor: np.ndarray):
        self.transform.enlarge(factor, space="global")
    
    def scale(self, factor: np.ndarray):
        self.enlarge(factor)

    def _repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

# 2D Shapes
class Shape2D(Shape):
    """Base for 2D shapes."""
    
    @property
    def dimensions(self) -> int:
        return 2

    @property
    def volume(self) -> float:
        return 0.0

    @property
    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def perimeter(self) -> float:
        raise NotImplementedError

class Circle(Shape2D):
    def __init__(self, center: np.ndarray, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.transform.position = np.asarray(center, dtype=float)
        if radius <= 0:
            raise ValueError("Radius must be > 0")
        self.radius = float(radius)
        self.transform.position = self.transform.position

    def signed_distance(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.transform.position) - self.radius

    def signed_distance(self, ray: Ray) -> bool:
        d = ray.orientation
        s = ray.origin - self.transform.position
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        return b ** 2 - 4 * a * c >= 0

    def get_ray_intersections(self, ray: Ray, bias: float = 1e-10) -> List[np.ndarray]:
        if not self.check_ray_intersection(ray):
            return []
        d = ray.orientation
        s = ray.origin - self.transform.position
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        if abs(discriminant) < bias:
            t = -b / (2 * a)
            return [ray.point_at(t)] if t >= 0 else []
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        ts = [t for t in [t1, t2] if t >= -bias]
        return [ray.point_at(t) for t in sorted(ts)]

    def get_normal(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        vec = point - self.transform.position
        return vec / (np.linalg.norm(vec) + bias)

    def get_tangent(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        normal = self.get_normal(point)
        # Perpendicular in 2D (assumes Z=0 plane)
        return np.array([-normal[1], normal[0], 0]) / (np.linalg.norm([-normal[1], normal[0], 0]) + bias)

    @property
    def area(self) -> float:
        from math import pi
        return pi * self.radius ** 2

    @property
    def perimeter(self) -> float:
        from math import pi
        return 2 * pi * self.radius

    def _repr__(self):
        return f"Circle(center={self.transform.position}, radius={self.radius})"

class Triangle(Shape2D):
    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.v1 = np.asarray(v1, dtype=float)
        self.v2 = np.asarray(v2, dtype=float)
        self.v3 = np.asarray(v3, dtype=float)
        self.transform.position = (self.v1 + self.v2 + self.v3) / 3
        self._validate()

    def validate(self, bias: float = 1e-6):
        """Check for degeneracy."""
        area = self.area
        if area < bias:
            raise ValueError("Triangle is degenerate (collinear or zero area)")

    def signed_distance(self, point: np.ndarray) -> float:
        """Unsigned distance to triangle (simplified: distance to closest edge/vertex)."""
        # For exact signed distance, would need plane equation + edge tests
        return self.get_distance(point)

    def get_distance(self, point: np.ndarray, bias: float = 1e-10) -> float:
        def point_to_segment_dist(p, a, b):
            ab = b - a
            ap = p - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < bias:
                return np.linalg.norm(ap)
            t = np.clip(np.dot(ap, ab) / ab_sq, 0, 1)
            return np.linalg.norm(p - (a + t * ab))
        
        d1 = point_to_segment_dist(point, self.v1, self.v2)
        d2 = point_to_segment_dist(point, self.v2, self.v3)
        d3 = point_to_segment_dist(point, self.v3, self.v1)
        return min(d1, d2, d3)

    def check_ray_intersection(self, ray: Ray, bias: float = 1e-10) -> bool:
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        h = np.cross(ray.orientation, edge2)
        a = np.dot(edge1, h)
        
        if abs(a) < bias:
            return False
        
        f = 1.0 / a
        s = ray.origin - self.v1
        u = f * np.dot(s, h)
        if u < 0 or u > 1:
            return False
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray.orientation, q)
        if v < 0 or u + v > 1:
            return False
        
        t = f * np.dot(edge2, q)
        return t > bias

    def get_ray_intersections(self, ray: Ray, bias: float = 1e-10) -> List[np.ndarray]:
        if not self.check_ray_intersection(ray):
            return []
        
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        h = np.cross(ray.orientation, edge2)
        a = np.dot(edge1, h)
        f = 1.0 / a
        s = ray.origin - self.v1
        u = f * np.dot(s, h)
        q = np.cross(s, edge1)
        v = f * np.dot(ray.orientation, q)
        t = f * np.dot(edge2, q)
        
        return [ray.point_at(t)] if t > bias else []

    def get_normal(self, point: np.ndarray, bias: float = 1e-10) -> np.ndarray:
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        normal = np.cross(edge1, edge2)
        return normal / (np.linalg.norm(normal) + bias)

    def get_tangent(self, point: np.ndarray, bias: float = 1e-10) -> np.ndarray:
        edge1 = self.v2 - self.v1
        return edge1 / (np.linalg.norm(edge1) + bias)

    @property
    def area(self) -> float:
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        return 0.5 * np.linalg.norm(np.cross(edge1, edge2))

    @property
    def perimeter(self) -> float:
        return (np.linalg.norm(self.v2 - self.v1) +
                np.linalg.norm(self.v3 - self.v2) +
                np.linalg.norm(self.v1 - self.v3))

    def _repr__(self):
        return f"Triangle(v1={self.v1}, v2={self.v2}, v3={self.v3})"

class Polygon(Shape2D):
    def __init__(self, vertices: List[np.ndarray], **kwargs):
        super().__init__(**kwargs)
        self.vertices = [np.asarray(v, dtype=float) for v in vertices]
        if len(self.vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices")
        self.transform.position = np.mean(self.vertices, axis=0)

    def signed_distance(self, point: np.ndarray) -> float:
        return self.get_distance(point)

    def get_distance(self, point: np.ndarray, bias: float = 1e-10) -> float:
        """Minimum distance to polygon edges or vertices."""
        def point_to_segment_dist(p, a, b):
            ab = b - a
            ap = p - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < bias:
                return np.linalg.norm(ap)
            t = np.clip(np.dot(ap, ab) / ab_sq, 0, 1)
            return np.linalg.norm(p - (a + t * ab))
        
        min_dist = float('inf')
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            dist = point_to_segment_dist(point, v1, v2)
            min_dist = min(min_dist, dist)
        return min_dist

    def check_ray_intersection(self, ray: Ray) -> bool:
        return len(self.get_ray_intersections(ray)) > 0

    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        # Simplified: check ray against each triangle formed by first vertex + edge
        intersections = []
        for i in range(1, len(self.vertices) - 1):
            tri = Triangle(self.vertices[0], self.vertices[i], self.vertices[i + 1])
            intersections.extend(tri.get_ray_intersections(ray))
        return intersections

    def get_normal(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        # Polygon normal: cross product of two edges
        edge1 = self.vertices[1] - self.vertices[0]
        edge2 = self.vertices[2] - self.vertices[0]
        normal = np.cross(edge1, edge2)
        return normal / (np.linalg.norm(normal) + bias)

    def get_tangent(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        edge = self.vertices[1] - self.vertices[0]
        return edge / (np.linalg.norm(edge) + bias)

    @property
    def area(self) -> float:
        """Shoelace formula for polygon area."""
        area = 0
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            area += np.cross(v1, v2)
        return abs(area) / 2

    @property
    def perimeter(self) -> float:
        perim = 0
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            perim += np.linalg.norm(v2 - v1)
        return perim

    def _repr__(self):
        return f"Polygon({len(self.vertices)} vertices)"

class Plane(Shape2D):
    def __init__(self, point: np.ndarray, normal: np.ndarray, bias: float = 1e-10, **kwargs):
        super().__init__(**kwargs)
        self.point = np.asarray(point, dtype=float)
        norm = np.linalg.norm(normal)
        if norm < bias:
            raise ValueError("Normal vector cannot be zero")
        self.normal = np.asarray(normal, dtype=float) / norm
        self.transform.position = self.point

    def signed_distance(self, point: np.ndarray) -> float:
        return np.dot(point - self.point, self.normal)

    def check_ray_intersection(self, ray: Ray, bias: float = 1e-10) -> bool:
        denom = np.dot(self.normal, ray.orientation)
        return abs(denom) > bias

    def get_ray_intersections(self, ray: Ray, bias: float = 1e-10) -> List[np.ndarray]:
        denom = np.dot(self.normal, ray.orientation)
        if abs(denom) < bias:
            return []
        
        t = np.dot(self.normal, self.point - ray.origin) / denom
        if t >= -bias:
            return [ray.point_at(t)]
        return []

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def get_tangent(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        if abs(self.normal[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(self.normal, arbitrary)
        return tangent / (np.linalg.norm(tangent) + bias)

    @property
    def area(self) -> float:
        return float('inf')

    @property
    def perimeter(self) -> float:
        return float('inf')

    def _repr__(self):
        return f"Plane(point={self.point}, normal={self.normal})"

class ClippedPlane(Plane):
    def __init__(self, point: np.ndarray, normal: np.ndarray, bounds: List[np.ndarray], **kwargs):
        super().__init__(point, normal, **kwargs)
        self.bounds = [np.asarray(b, dtype=float) for b in bounds]
    
    def check_ray_intersection(self, ray: Ray) -> bool:
        intersections = self.get_ray_intersections(ray)
        return len(intersections) > 0
    
    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        intersections = super().get_ray_intersections(ray)
        valid_points = []
        for pt in intersections:
            if self._point_in_bounds(pt):
                valid_points.append(pt)
        return valid_points
    
    def point_in_bounds(self, point: np.ndarray) -> bool:
        # Simple bounding box check (could be improved for arbitrary polygons)
        xs = [b[0] for b in self.bounds]
        ys = [b[1] for b in self.bounds]
        return (min(xs) <= point[0] <= max(xs)) and (min(ys) <= point[1] <= max(ys))

    @property
    def area(self) -> float:
        # Approximate area via polygon area formula
        area = 0
        n = len(self.bounds)
        for i in range(n):
            v1 = self.bounds[i]
            v2 = self.bounds[(i + 1) % n]
            area += v1[0] * v2[1] - v2[0] * v1[1]
        return abs(area) / 2

    @property
    def perimeter(self) -> float:
        perim = 0
        n = len(self.bounds)
        for i in range(n):
            v1 = self.bounds[i]
            v2 = self.bounds[(i + 1) % n]
            perim += np.linalg.norm(v2 - v1)
        return perim

    def _repr__(self):
        return f"ClippedPlane(point={self.point}, normal={self.normal}, bounds={len(self.bounds)} vertices)"

# 3D Shapes
class Shape3D(Shape):
    """Base for 3D shapes."""
    
    @property
    def dimensions(self) -> int:
        return 3

    @property
    @abstractmethod
    def volume(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def surface_area(self) -> float:
        raise NotImplementedError

    @property
    def area(self) -> float:
        return self.surface_area

    def convex_hull(self, resolution: int = 100) -> List[np.ndarray]:
        """Approximate convex hull (override for exact implementations)."""
        raise NotImplementedError

class Sphere(Shape3D):
    def __init__(self, center: np.ndarray, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.transform.position = np.asarray(center, dtype=float)
        if radius <= 0:
            raise ValueError("Radius must be > 0")
        self.radius = float(radius)
        self.transform.position = self.transform.position

    def signed_distance(self, point: np.ndarray) -> float:
        # Transform point into local/object space (handles position, rotation and scale)
        p_local = world_to_local_point(point, self.transform)
        sdf_local = np.linalg.norm(p_local) - self.radius
        # Approximate world-space distance by scaling with the smallest global scale
        scale = self.transform.get_global_scale()
        return sdf_local * float(np.min(scale))

    def check_ray_intersection(self, ray: Ray) -> bool:
        local_ray = ray_world_to_local(ray, self.transform)
        d = local_ray.orientation
        s = local_ray.origin
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        return b ** 2 - 4 * a * c >= 0

    def get_ray_intersections(self, ray: Ray, bias: float = 1e-10) -> List[np.ndarray]:
        if not self.check_ray_intersection(ray):
            return []

        local_ray = ray_world_to_local(ray, self.transform)
        d = local_ray.orientation
        s = local_ray.origin
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if abs(discriminant) < bias:
            t = -b / (2 * a)
            if t < -bias:
                return []
            local_pt = local_ray.point_at(t)
            return [local_to_world_point(local_pt, self.transform)]

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        ts = [t for t in [t1, t2] if t >= -bias]
        world_pts = []
        for t in sorted(ts):
            local_pt = local_ray.point_at(t)
            world_pts.append(local_to_world_point(local_pt, self.transform))
        return world_pts

    def get_normal(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        # Compute normal in local space then transform to world space correctly
        p_local = world_to_local_point(point, self.transform)
        local_n = p_local / (np.linalg.norm(p_local) + bias)
        return normal_local_to_world(local_n, self.transform)

    def get_tangent(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        # Compute tangent in local space and transform it to world
        p_local = world_to_local_point(point, self.transform)
        local_n = p_local / (np.linalg.norm(p_local) + bias)
        # pick local arbitrary axis
        if abs(local_n[0]) < 0.9:
            arb = np.array([1.0, 0.0, 0.0])
        else:
            arb = np.array([0.0, 1.0, 0.0])
        local_t = np.cross(local_n, arb)
        local_t = local_t / (np.linalg.norm(local_t) + bias)
        # convert tangent direction to world (linear part)
        lin = self.transform.model_matrix[:3, :3]
        world_t = lin @ local_t
        return world_t / (np.linalg.norm(world_t) + bias)

    @property
    def volume(self) -> float:
        from math import pi
        return (4/3) * pi * self.radius ** 3

    @property
    def surface_area(self) -> float:
        from math import pi
        return 4 * pi * self.radius ** 2

    def convex_hull(self, resolution: int = 100) -> List[np.ndarray]:
        points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(resolution):
            theta = 2 * np.pi * i / phi
            y = 1 - (i / (resolution - 1)) * 2
            r = np.sqrt(max(1 - y * y, 0))
            local_pt = self.radius * np.array([np.cos(theta) * r, y, np.sin(theta) * r])
            points.append(local_to_world_point(local_pt, self.transform))
        return points

    def _repr__(self):
        return f"Sphere(center={self.transform.position}, radius={self.radius})"

class Cube(Shape3D):
    def __init__(self, center: np.ndarray, side_length: float, **kwargs):
        super().__init__(**kwargs)
        self.transform.position = np.asarray(center, dtype=float)
        if side_length <= 0:
            raise ValueError("Side length must be > 0")
        self.side_length = float(side_length)

    def signed_distance(self, point: np.ndarray) -> float:
        # Work in local/object space for correct handling of rotation & scale
        p_local = world_to_local_point(point, self.transform)
        abs_p = np.abs(p_local)
        half = self.side_length / 2
        q = abs_p - half
        sdf_local = np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)
        scale = self.transform.get_global_scale()
        return sdf_local * float(np.min(scale))

    def check_ray_intersection(self, ray: Ray) -> bool:
        return len(self.get_ray_intersections(ray)) > 0
    
    def get_ray_intersections(self, ray: Ray, bias: float = 1e-10) -> List[np.ndarray]:
        # Transform ray to local/object space and intersect with axis-aligned cube centered at origin
        local_ray = ray_world_to_local(ray, self.transform)
        half = self.side_length / 2
        bounds_min = -np.ones(3) * half
        bounds_max = np.ones(3) * half

        t_min, t_max = -float('inf'), float('inf')
        for i in range(3):
            if abs(local_ray.orientation[i]) > bias:
                t1 = (bounds_min[i] - local_ray.origin[i]) / local_ray.orientation[i]
                t2 = (bounds_max[i] - local_ray.origin[i]) / local_ray.orientation[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif local_ray.origin[i] < bounds_min[i] or local_ray.origin[i] > bounds_max[i]:
                return []

        if t_min <= t_max and t_max >= -bias:
            pts = []
            if t_min >= -bias:
                pts.append(local_to_world_point(local_ray.point_at(t_min), self.transform))
            if t_max != t_min and t_max >= -bias:
                pts.append(local_to_world_point(local_ray.point_at(t_max), self.transform))
            return pts
        return []

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        # Compute normal in local space (axis-aligned cube) and transform to world
        p_local = world_to_local_point(point, self.transform)
        half = self.side_length / 2
        abs_p = np.abs(p_local)
        dominant = np.argmax(abs_p)
        local_n = np.zeros(3)
        local_n[dominant] = np.sign(p_local[dominant])
        return normal_local_to_world(local_n, self.transform)

    def get_tangent(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        normal = self.get_normal(point)
        if abs(normal[2]) < 0.9:
            arbitrary = np.array([0.0, 0.0, 1.0])
        else:
            arbitrary = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(normal, arbitrary)
        return tangent / (np.linalg.norm(tangent) + bias)

    @property
    def volume(self) -> float:
        return self.side_length ** 3

    @property
    def surface_area(self) -> float:
        return 6 * (self.side_length ** 2)

    def _repr__(self):
        return f"Cube(transform={self.transform}, side_length={self.side_length})"

class Cuboid(Shape3D):
    def __init__(self, center: np.ndarray, dimensions: Tuple[float, float, float], **kwargs):
        super().__init__(**kwargs)
        self.transform.position = np.asarray(center, dtype=float)
        
        # Validate dimensions
        if any(d <= 0 for d in dimensions):
            raise ValueError("All dimensions must be > 0")
            
        # Store as a tuple for immutability, or array for math
        self.dimensions_tuple = tuple(float(d) for d in dimensions)

    def signed_distance(self, point: np.ndarray) -> float:
        """
        Calculates the Signed Distance Function (SDF).
        NOTE: For non-uniform scaling, this returns a conservative lower bound,
        not the exact Euclidean distance.
        """
        p_local = np.abs(world_to_local_point(point, self.transform))
        # d is the extent (half-dimensions)
        d = np.array(self.dimensions_tuple) / 2
        
        # q = distance from the positive octant corner
        q = p_local - d
        
        # Standard Box SDF logic
        sdf_local = np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)
        
        # Apply scaling (conservative)
        scale = self.transform.get_global_scale()
        return sdf_local * float(np.min(scale))

    def check_ray_intersection(self, ray: Ray) -> bool:
        # Optimization: You could rewrite this to return True immediately 
        # inside the loop logic, but wrapping get_ray_intersections is cleaner.
        return len(self.get_ray_intersections(ray)) > 0

    def get_ray_intersections(self, ray: Ray, bias: float = 1e-10) -> List[np.ndarray]:
        local_ray = ray_world_to_local(ray, self.transform)
        half = np.array(self.dimensions_tuple) / 2
        bounds_min = -half
        bounds_max = half

        t_min, t_max = -float('inf'), float('inf')

        # Slab method for AABB intersection
        for i in range(3):
            # Check if ray is not parallel to the slab
            if abs(local_ray.orientation[i]) > bias:
                inv_d = 1.0 / local_ray.orientation[i]
                t1 = (bounds_min[i] - local_ray.origin[i]) * inv_d
                t2 = (bounds_max[i] - local_ray.origin[i]) * inv_d
                
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            
            # If ray is parallel, it must be inside the slab bounds
            elif local_ray.origin[i] < bounds_min[i] or local_ray.origin[i] > bounds_max[i]:
                return []

        # Check if a valid intersection interval exists
        if t_min <= t_max and t_max >= -bias:
            pts = []
            # Entry point (if not behind origin)
            if t_min >= -bias:
                pts.append(local_to_world_point(local_ray.point_at(t_min), self.transform))
            # Exit point (if distinct and valid)
            if t_max != t_min and t_max >= -bias:
                pts.append(local_to_world_point(local_ray.point_at(t_max), self.transform))
            return pts
            
        return []

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        p_local = world_to_local_point(point, self.transform)
        half = np.array(self.dimensions_tuple) / 2
        
        # Calculate deviation ratio to find the dominant axis
        # Dividing by half maps the box extents to a [-1, 1] range
        ratio = np.abs(p_local) / half
        dominant_axis = np.argmax(ratio)
        
        local_n = np.zeros(3)
        # Sign determines if it's the positive or negative face
        local_n[dominant_axis] = np.sign(p_local[dominant_axis])
        
        return normal_local_to_world(local_n, self.transform)
    
    @property
    def volume(self) -> float:
        w, h, d = self.dimensions_tuple
        return w * h * d

    @property
    def surface_area(self) -> float:
        w, h, d = self.dimensions_tuple
        return 2 * (w * h + h * d + w * d)

class Prism(Shape3D):
    def __init__(self, base_polygon: Polygon, height: float, **kwargs):
        super().__init__(**kwargs)
        self.base_polygon = base_polygon
        if height <= 0:
            raise ValueError("Height must be > 0")
        self.height = float(height)
        self.transform.position = base_polygon.transform.position

    @property
    def volume(self) -> float:
        return self.base_polygon.area * self.height

    @property
    def surface_area(self) -> float:
        return 2 * self.base_polygon.area + self.base_polygon.perimeter * self.height

    def signed_distance(self, point: np.ndarray) -> float:
        return self.get_distance(point)

    def check_ray_intersection(self, ray: Ray) -> bool:
        return len(self.get_ray_intersections(ray)) > 0

    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        # Simplified: triangulate base and extrude
        raise NotImplementedError("Prism ray intersection not yet implemented")

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Prism normal not yet implemented")

    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Prism tangent not yet implemented")

    def convex_hull(self, resolution: int = 100) -> List[np.ndarray]:
        raise NotImplementedError("Prism convex hull not yet implemented")

    def _repr__(self):
        return f"Prism(base_polygon={self.base_polygon}, height={self.height})"

class Pyramid(Shape3D):
    def __init__(self, base_polygon: Polygon, height: float, **kwargs):
        super().__init__(**kwargs)
        self.base_polygon = base_polygon
        if height <= 0:
            raise ValueError("Height must be > 0")
        self.height = float(height)
        self.apex = base_polygon.transform.position + np.array([0, height, 0])
        self.transform.position = base_polygon.transform.position

    @property
    def volume(self) -> float:
        return (1/3) * self.base_polygon.area * self.height

    @property
    def surface_area(self) -> float:
        return self.base_polygon.area  # + lateral faces (stub)

    def signed_distance(self, point: np.ndarray) -> float:
        return self.get_distance(point)

    def check_ray_intersection(self, ray: Ray) -> bool:
        return len(self.get_ray_intersections(ray)) > 0

    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        raise NotImplementedError("Pyramid ray intersection not yet implemented")

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pyramid normal not yet implemented")

    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pyramid tangent not yet implemented")

    def convex_hull(self, resolution: int = 100) -> List[np.ndarray]:
        raise NotImplementedError("Pyramid convex hull not yet implemented")

    def _repr__(self):
        return f"Pyramid(base_polygon={self.base_polygon}, height={self.height})"

class Capsule(Shape3D):
    def __init__(self, point1: np.ndarray, point2: np.ndarray, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.point1 = np.asarray(point1, dtype=float)
        self.point2 = np.asarray(point2, dtype=float)
        if radius <= 0:
            raise ValueError("Radius must be > 0")
        self.radius = float(radius)
        self.transform.position = (self.point1 + self.point2) / 2

    def signed_distance(self, point: np.ndarray, bias: float = 1e-10) -> float:
        """Distance from point to capsule surface."""
        ab = self.point2 - self.point1
        ap = point - self.point1
        ab_sq = np.dot(ab, ab)
        t = np.clip(np.dot(ap, ab) / max(ab_sq, bias), 0, 1)
        closest = self.point1 + t * ab
        return np.linalg.norm(point - closest) - self.radius

    def check_ray_intersection(self, ray: Ray) -> bool:
        return len(self.get_ray_intersections(ray)) > 0

    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        raise NotImplementedError("Capsule ray intersection not yet implemented")

    def get_normal(self, point: np.ndarray, bias: float = 1e-10) -> np.ndarray:
        ab = self.point2 - self.point1
        ap = point - self.point1
        t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), bias), 0, 1)
        closest = self.point1 + t * ab
        vec = point - closest
        return vec / (np.linalg.norm(vec) + bias)

    def get_tangent(self, point: np.ndarray, bias: float = 1e-10) -> np.ndarray:
        ab = self.point2 - self.point1
        ab = ab / (np.linalg.norm(ab) + bias)
        normal = self.get_normal(point)
        tangent = np.cross(normal, ab)
        return tangent / (np.linalg.norm(tangent) + bias)

    @property
    def volume(self) -> float:
        from math import pi
        cylinder_height = np.linalg.norm(self.point2 - self.point1)
        return pi * self.radius ** 2 * cylinder_height + (4/3) * pi * self.radius ** 3

    @property
    def surface_area(self) -> float:
        from math import pi
        cylinder_height = np.linalg.norm(self.point2 - self.point1)
        return 2 * pi * self.radius * cylinder_height + 4 * pi * self.radius ** 2

    def convex_hull(self, resolution: int = 100) -> List[np.ndarray]:
        raise NotImplementedError("Capsule convex hull not yet implemented")

    def _repr__(self):
        return f"Capsule(point1={self.point1}, point2={self.point2}, radius={self.radius})"

# VObject & Factories
@dataclass
class VObject:
    """Visual object combining shape, transform, PBRMaterial etc."""
    shape: Shape
    transform: Optional["Transform"] = None
    PBRMaterial: Optional["PBRMaterial"] = None
    name: str = "VObject"

    children: List["VObject"] = field(default_factory=list)
    parent: Optional["VObject"] = None

    def _post_init__(self):
        if self.transform is None:
            self.transform = Transform(np.zeros(3), np.zeros(3), np.ones(3))

    @property
    def position(self) -> np.ndarray:
        """Get world position considering parent hierarchy."""
        world_pos = self.transform.position
        if self.parent is not None:
            world_pos = world_pos + self.parent.position
        return world_pos
    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Set local position relative to parent."""
        value = np.asarray(value, dtype=float)
        if self.parent is not None:
            self.transform.position = value - self.parent.position
        else:
            self.transform.position = value
    @property
    def world_position(self) -> np.ndarray:
        """Get absolute world position."""
        world_pos = self.transform.position
        current = self.parent
        while current is not None:
            world_pos = world_pos + current.transform.position
            current = current.parent
        return world_pos
    @property
    def local_position(self) -> np.ndarray:
        """Get position relative to parent."""
        return self.transform.position
    @local_position.setter
    def local_position(self, value: np.ndarray) -> None:
        """Set position relative to parent."""
        self.transform.position = np.asarray(value, dtype=float)
    @property
    def rotation(self) -> np.ndarray:
        """Get local rotation."""
        return self.transform.rotation
    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        """Set local rotation."""
        self.transform.rotation = np.asarray(value, dtype=float)
    @property
    def scale(self) -> np.ndarray:
        """Get local scale."""
        return self.transform.sca
    @scale.setter
    def scale(self, value: np.ndarray) -> None:
        """Set local scale."""
        self.transform.scale = np.asarray(value, dtype=float)
    def translate(self, offset: np.ndarray, space: str = "local") -> None:
        """Translate in local or world space."""
        offset = np.asarray(offset, dtype=float)
        if space == "local":
            self.transform.translate(offset, space="local")
        else:
            self.transform.translate(offset, space="global")
    def rotate(self, angle: float, axis: np.ndarray, space: str = "local") -> None:
        """Rotate around axis in local or world space."""
        self.transform.rotate(angle, axis, space=space)
    def scale_by(self, factor: np.ndarray) -> None:
        """Scale uniformly or per-axis."""
        factor = np.asarray(factor, dtype=float)
        self.transform.enlarge(factor, space="local")

    def _repr__(self):
        return f"VObject(name={self.name}, shape={self.shape})"

class ShapeFactory(ABC):
    """Abstract factory for creating shapes."""
    @abstractmethod
    def create(self, **kwargs) -> Shape:
        raise NotImplementedError

class CircleFactory(ShapeFactory):
    def create(self, center: np.ndarray, radius: float, **kwargs) -> Circle:
        return Circle(center, radius, **kwargs)

class TriangleFactory(ShapeFactory):
    def create(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs) -> Triangle:
        return Triangle(v1, v2, v3, **kwargs)

class PolygonFactory(ShapeFactory):
    def create(self, vertices: List[np.ndarray], **kwargs) -> Polygon:
        return Polygon(vertices, **kwargs)

class SphereFactory(ShapeFactory):
    def create(self, center: np.ndarray, radius: float, **kwargs) -> Sphere:
        return Sphere(center, radius, **kwargs)

class CubeFactory(ShapeFactory):
    def create(self, center: np.ndarray, side_length: float, **kwargs) -> Cube:
        return Cube(center, side_length, **kwargs)

class PrismFactory(ShapeFactory):
    def create(self, base_polygon: Polygon, height: float, **kwargs) -> Prism:
        return Prism(base_polygon, height, **kwargs)

class PyramidFactory(ShapeFactory):
    def create(self, base_polygon: Polygon, height: float, **kwargs) -> Pyramid:
        return Pyramid(base_polygon, height, **kwargs)

class CapsuleFactory(ShapeFactory):
    def create(self, point1: np.ndarray, point2: np.ndarray, radius: float, **kwargs) -> Capsule:
        return Capsule(point1, point2, radius, **kwargs)

def sdf_unit_sphere(point: np.ndarray) -> float:
    return np.linalg.norm(point) - 1

def get_transformed_exit_point(
    ray_origin_world: np.ndarray, 
    ray_dir_world: np.ndarray, 
    object_transform_matrix: np.ndarray, 
    inverse_transform_matrix: np.ndarray,
    max_steps: int = 64,
    epsilon: float = 1e-4
) -> Optional[np.ndarray]:
    
    # --- STEP 1: Transform Ray to Local Space ---
    # 1. Transform Origin (Point: w=1.0)
    origin_4d = np.append(ray_origin_world, 1.0)
    origin_local_4d = inverse_transform_matrix @ origin_4d
    ray_origin_local = origin_local_4d[:3]
    
    # 2. Transform Direction (Vector: w=0.0)
    # We only need the rotation/scale 3x3 block for vectors
    ray_dir_local_raw = inverse_transform_matrix[:3, :3] @ ray_dir_world
    
    # 3. Normalize Direction
    # This is crucial: We are now working in "Unit Sphere Space".
    # The length of the raw vector encodes the scaling, but standard SDFs 
    # expect normalized directions to step correctly.
    ray_dir_local = ray_dir_local_raw / (np.linalg.norm(ray_dir_local_raw) + 1e-8)

    # --- STEP 2: Raymarch in Local Space (Unit Sphere) ---
    t = 0.0
    hit_entry = False

    # A. Find Entry Point (Front Face)
    for _ in range(max_steps):
        p = ray_origin_local + (ray_dir_local * t)
        dist = sdf_unit_sphere(p)
        
        # Optimization: If we are too far away, we missed
        if dist > 10.0: 
            return None 

        if dist < epsilon:
            hit_entry = True
            break
        t += dist

    # If we never hit the entry, we can't find an exit!
    if not hit_entry:
        return None

    # B. Find Exit Point (Back Face)
    # Push slightly inside the sphere to start the internal march
    t += epsilon * 2.0 
    
    for _ in range(max_steps):
        p = ray_origin_local + (ray_dir_local * t)
        
        # INVERTED SDF: We want distance to the "outer shell" from inside.
        # Inside a unit sphere, dist is negative. -dist makes it positive.
        dist = -sdf_unit_sphere(p)
        
        if dist < epsilon:
            # We hit the exit boundary!
            p_exit_local = p 
            
            # --- STEP 3: Transform Back to World Space ---
            # FIX: Must use homogeneous coordinates (4D)
            p_exit_local_4d = np.append(p_exit_local, 1.0)
            
            # Apply World Matrix
            p_exit_world_4d = object_transform_matrix @ p_exit_local_4d
            
            # Return 3D (x,y,z)
            return p_exit_world_4d[:3]
            
        t += dist
        
    return None

"""
Geometry Module: Defines geometric shapes, their properties, and interactions.
Provides base classes, mixins, and concrete implementations for 2D and 3D shapes, including ray intersection and surface property queries.
Also provides an inverted sdf method to find exist points for rays

Classes:
- Shape: Abstract base class for all shapes.
- Shape2D, Shape3D: Base classes for 2D and 3D shapes.
- Circle, Triangle, Polygon, Plane, ClippedPlane: Concrete 2D shape implementations.
- Sphere, Cube, Cuboid, Prism, Pyramid, Capsule: Concrete 3D shape implementations.
- GeometryMixin, RayIntersectionMixin, SurfacePropertiesMixin: Mixins for geometric queries.
- VObject: Combines shape with transform and PBRMaterial for rendering.
- ShapeFactory and its subclasses: Factories for creating shape instances. 
"""