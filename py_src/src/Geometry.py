from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from PrimaryStructures import Transform, Ray
from Luminance import Color, Material

# Base Classes & Mixins
class GeometryMixin(ABC):
    """Mixin providing default geometric query implementations."""
    
    @abstractmethod
    def SignedDistance(self, point: np.ndarray) -> float:
        """Signed distance (negative inside, positive outside)."""
        raise NotImplementedError

    def GetDistance(self, point: np.ndarray) -> float:
        """Unsigned distance to surface."""
        return abs(self.SignedDistance(point))

    def CheckPointInside(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        """Point is inside if signed distance < -epsilon."""
        return self.SignedDistance(point) < -epsilon

    def CheckPointOnSurface(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        """Point is on surface if |signed distance| <= epsilon."""
        return abs(self.SignedDistance(point)) <= epsilon

    def GetClosestPoint(self, point: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Use gradient descent to find closest point on surface.
        Fallback for shapes without closed-form solution.
        """
        current = np.array(point, dtype=float)
        step_size = 0.1
        
        for _ in range(max_iterations):
            dist = self.SignedDistance(current)
            if abs(dist) < 1e-6:
                return current
            
            # Approximate gradient via finite differences
            eps = 1e-5
            grad = np.array([
                (self.SignedDistance(current + np.array([eps, 0, 0])) - dist) / eps,
                (self.SignedDistance(current + np.array([0, eps, 0])) - dist) / eps,
                (self.SignedDistance(current + np.array([0, 0, eps])) - dist) / eps,
            ])
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            else:
                break
            
            current = current - dist * grad
        
        return current

class RayIntersectionMixin(ABC):
    """Mixin for ray-shape intersection tests."""
    
    @abstractmethod
    def CheckRayIntersection(self, ray: "Ray") -> bool:
        raise NotImplementedError

    @abstractmethod
    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        raise NotImplementedError

class SurfacePropertiesMixin(ABC):
    """Mixin for surface normal/tangent/binormal queries."""
    
    @abstractmethod
    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        """Default: perpendicular to normal (requires override for custom behavior)."""
        raise NotImplementedError
    
    def GetBinormal(self, point: np.ndarray) -> np.ndarray:
        """Default: cross(normal, tangent)."""
        normal = self.GetNormal(point)
        tangent = self.GetTangent(point)
        return np.cross(normal, tangent) / (np.linalg.norm(np.cross(normal, tangent)) + 1e-12)

class Shape(ABC):
    """
    Abstract base for all shapes.
    Provides transform, material, and naming.
    """
    def __init__(self, name: str = "Shape", transform: Optional["Transform"] = None, 
                 material: Optional["Material"] = None, **kwargs):
        self.name = name
        self.transform = transform or Transform(np.zeros(3), np.zeros(3), np.ones(3))
        self.material = material
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def origin(self) -> np.ndarray:
        return self.transform.position

    @abstractmethod
    def SignedDistance(self, point: np.ndarray) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """2 for 2D, 3 for 3D."""
        raise NotImplementedError

    def Translate(self, offset: np.ndarray) -> None:
        self.transform.translate(offset, space="global")

    def Rotate(self, angle: float, axis: np.ndarray) -> None:
        self.transform.rotate(angle, axis, space="global")

    def Scale(self, factor: np.ndarray) -> None:
        self.transform.enlarge(factor, space="global")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

# 2D Shapes

class Shape2D(Shape, GeometryMixin, RayIntersectionMixin, SurfacePropertiesMixin):
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
        self.center = np.asarray(center, dtype=float)
        if radius <= 0:
            raise ValueError("Radius must be > 0")
        self.radius = float(radius)
        self.transform.position = self.center

    def SignedDistance(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.center) - self.radius

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        d = ray.orientation
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        return b ** 2 - 4 * a * c >= 0

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        if not self.CheckRayIntersection(ray):
            return []
        d = ray.orientation
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        if abs(discriminant) < 1e-10:
            t = -b / (2 * a)
            return [ray.point_at(t)] if t >= 0 else []
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        ts = [t for t in [t1, t2] if t >= -1e-10]
        return [ray.point_at(t) for t in sorted(ts)]

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        vec = point - self.center
        return vec / (np.linalg.norm(vec) + 1e-12)

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        normal = self.GetNormal(point)
        # Perpendicular in 2D (assumes Z=0 plane)
        return np.array([-normal[1], normal[0], 0]) / (np.linalg.norm([-normal[1], normal[0], 0]) + 1e-12)

    @property
    def area(self) -> float:
        from math import pi
        return pi * self.radius ** 2

    @property
    def perimeter(self) -> float:
        from math import pi
        return 2 * pi * self.radius

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"

class Triangle(Shape2D):
    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.v1 = np.asarray(v1, dtype=float)
        self.v2 = np.asarray(v2, dtype=float)
        self.v3 = np.asarray(v3, dtype=float)
        self.transform.position = (self.v1 + self.v2 + self.v3) / 3
        self._validate()

    def _validate(self):
        """Check for degeneracy."""
        area = self.area
        if area < 1e-6:
            raise ValueError("Triangle is degenerate (collinear or zero area)")

    def SignedDistance(self, point: np.ndarray) -> float:
        """Unsigned distance to triangle (simplified: distance to closest edge/vertex)."""
        # For exact signed distance, would need plane equation + edge tests
        return self.GetDistance(point)

    def GetDistance(self, point: np.ndarray) -> float:
        def point_to_segment_dist(p, a, b):
            ab = b - a
            ap = p - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-10:
                return np.linalg.norm(ap)
            t = np.clip(np.dot(ap, ab) / ab_sq, 0, 1)
            return np.linalg.norm(p - (a + t * ab))
        
        d1 = point_to_segment_dist(point, self.v1, self.v2)
        d2 = point_to_segment_dist(point, self.v2, self.v3)
        d3 = point_to_segment_dist(point, self.v3, self.v1)
        return min(d1, d2, d3)

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        h = np.cross(ray.orientation, edge2)
        a = np.dot(edge1, h)
        
        if abs(a) < 1e-10:
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
        return t > 1e-10

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        if not self.CheckRayIntersection(ray):
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
        
        return [ray.point_at(t)] if t > 1e-10 else []

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        normal = np.cross(edge1, edge2)
        return normal / (np.linalg.norm(normal) + 1e-12)

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        edge1 = self.v2 - self.v1
        return edge1 / (np.linalg.norm(edge1) + 1e-12)

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

    def __repr__(self):
        return f"Triangle(v1={self.v1}, v2={self.v2}, v3={self.v3})"

class Polygon(Shape2D):
    def __init__(self, vertices: List[np.ndarray], **kwargs):
        super().__init__(**kwargs)
        self.vertices = [np.asarray(v, dtype=float) for v in vertices]
        if len(self.vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices")
        self.transform.position = np.mean(self.vertices, axis=0)

    def SignedDistance(self, point: np.ndarray) -> float:
        return self.GetDistance(point)

    def GetDistance(self, point: np.ndarray) -> float:
        """Minimum distance to polygon edges or vertices."""
        def point_to_segment_dist(p, a, b):
            ab = b - a
            ap = p - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-10:
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

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        return len(self.GetRayIntersections(ray)) > 0

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        # Simplified: check ray against each triangle formed by first vertex + edge
        intersections = []
        for i in range(1, len(self.vertices) - 1):
            tri = Triangle(self.vertices[0], self.vertices[i], self.vertices[i + 1])
            intersections.extend(tri.GetRayIntersections(ray))
        return intersections

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        # Polygon normal: cross product of two edges
        edge1 = self.vertices[1] - self.vertices[0]
        edge2 = self.vertices[2] - self.vertices[0]
        normal = np.cross(edge1, edge2)
        return normal / (np.linalg.norm(normal) + 1e-12)

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        edge = self.vertices[1] - self.vertices[0]
        return edge / (np.linalg.norm(edge) + 1e-12)

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

    def __repr__(self):
        return f"Polygon({len(self.vertices)} vertices)"

# 3D Shapes

class Shape3D(Shape, GeometryMixin, RayIntersectionMixin, SurfacePropertiesMixin):
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

    def ConvexHull(self, resolution: int = 100) -> List[np.ndarray]:
        """Approximate convex hull (override for exact implementations)."""
        raise NotImplementedError

class Sphere(Shape3D):
    def __init__(self, center: np.ndarray, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.center = np.asarray(center, dtype=float)
        if radius <= 0:
            raise ValueError("Radius must be > 0")
        self.radius = float(radius)
        self.transform.position = self.center

    def SignedDistance(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.center) - self.radius

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        d = ray.orientation
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        return b ** 2 - 4 * a * c >= 0

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        if not self.CheckRayIntersection(ray):
            return []
        
        d = ray.orientation
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        if abs(discriminant) < 1e-10:
            t = -b / (2 * a)
            return [ray.point_at(t)] if t >= -1e-10 else []
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        ts = [t for t in [t1, t2] if t >= -1e-10]
        return [ray.point_at(t) for t in sorted(ts)]

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        vec = point - self.center
        return vec / (np.linalg.norm(vec) + 1e-12)

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        normal = self.GetNormal(point)
        if abs(normal[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(normal, arbitrary)
        return tangent / (np.linalg.norm(tangent) + 1e-12)

    @property
    def volume(self) -> float:
        from math import pi
        return (4/3) * pi * self.radius ** 3

    @property
    def surface_area(self) -> float:
        from math import pi
        return 4 * pi * self.radius ** 2

    def ConvexHull(self, resolution: int = 100) -> List[np.ndarray]:
        points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(resolution):
            theta = 2 * np.pi * i / phi
            y = 1 - (i / (resolution - 1)) * 2
            r = np.sqrt(max(1 - y * y, 0))
            points.append(self.center + self.radius * np.array([np.cos(theta) * r, y, np.sin(theta) * r]))
        return points

    def __repr__(self):
        return f"Sphere(center={self.center}, radius={self.radius})"

class Cube(Shape3D):
    def __init__(self, center: np.ndarray, side_length: float, **kwargs):
        super().__init__(**kwargs)
        self.center = np.asarray(center, dtype=float)
        if side_length <= 0:
            raise ValueError("Side length must be > 0")
        self.side_length = float(side_length)
        self.transform.position = self.center

    def SignedDistance(self, point: np.ndarray) -> float:
        """Cube SDF using absolute coordinates."""
        p = np.abs(point - self.center)
        q = p - self.side_length / 2
        return np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        return len(self.GetRayIntersections(ray)) > 0

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        """AABB ray intersection."""
        half = self.side_length / 2
        bounds_min = self.center - half
        bounds_max = self.center + half
        
        t_min, t_max = 0, float('inf')
        for i in range(3):
            if abs(ray.orientation[i]) > 1e-10:
                t1 = (bounds_min[i] - ray.origin[i]) / ray.orientation[i]
                t2 = (bounds_max[i] - ray.origin[i]) / ray.orientation[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif ray.origin[i] < bounds_min[i] or ray.origin[i] > bounds_max[i]:
                return []
        
        if t_min <= t_max and t_max >= -1e-10:
            result = []
            if t_min >= -1e-10:
                result.append(ray.point_at(t_min))
            if t_max != t_min and t_max >= -1e-10:
                result.append(ray.point_at(t_max))
            return result
        return []

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        """Normal pointing outward from closest face."""
        p = point - self.center
        half = self.side_length / 2
        abs_p = np.abs(p)
        dominant = np.argmax(abs_p)
        normal = np.zeros(3)
        normal[dominant] = np.sign(p[dominant])
        return normal

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        normal = self.GetNormal(point)
        if abs(normal[2]) < 0.9:
            arbitrary = np.array([0.0, 0.0, 1.0])
        else:
            arbitrary = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(normal, arbitrary)
        return tangent / (np.linalg.norm(tangent) + 1e-12)

    @property
    def volume(self) -> float:
        return self.side_length ** 3

    @property
    def surface_area(self) -> float:
        return 6 * (self.side_length ** 2)

    def ConvexHull(self, resolution: int = 8) -> List[np.ndarray]:
        half = self.side_length / 2
        points = []
        for dx in [-half, half]:
            for dy in [-half, half]:
                for dz in [-half, half]:
                    points.append(self.center + np.array([dx, dy, dz]))
        return points

    def __repr__(self):
        return f"Cube(center={self.center}, side_length={self.side_length})"

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

    def SignedDistance(self, point: np.ndarray) -> float:
        return self.GetDistance(point)

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        return len(self.GetRayIntersections(ray)) > 0

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        # Simplified: triangulate base and extrude
        raise NotImplementedError("Prism ray intersection not yet implemented")

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Prism normal not yet implemented")

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Prism tangent not yet implemented")

    def ConvexHull(self, resolution: int = 100) -> List[np.ndarray]:
        raise NotImplementedError("Prism convex hull not yet implemented")

    def __repr__(self):
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

    def SignedDistance(self, point: np.ndarray) -> float:
        return self.GetDistance(point)

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        return len(self.GetRayIntersections(ray)) > 0

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        raise NotImplementedError("Pyramid ray intersection not yet implemented")

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pyramid normal not yet implemented")

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pyramid tangent not yet implemented")

    def ConvexHull(self, resolution: int = 100) -> List[np.ndarray]:
        raise NotImplementedError("Pyramid convex hull not yet implemented")

    def __repr__(self):
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

    def SignedDistance(self, point: np.ndarray) -> float:
        """Distance from point to capsule surface."""
        ab = self.point2 - self.point1
        ap = point - self.point1
        ab_sq = np.dot(ab, ab)
        t = np.clip(np.dot(ap, ab) / max(ab_sq, 1e-10), 0, 1)
        closest = self.point1 + t * ab
        return np.linalg.norm(point - closest) - self.radius

    def CheckRayIntersection(self, ray: "Ray") -> bool:
        return len(self.GetRayIntersections(ray)) > 0

    def GetRayIntersections(self, ray: "Ray") -> List[np.ndarray]:
        raise NotImplementedError("Capsule ray intersection not yet implemented")

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        ab = self.point2 - self.point1
        ap = point - self.point1
        t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-10), 0, 1)
        closest = self.point1 + t * ab
        vec = point - closest
        return vec / (np.linalg.norm(vec) + 1e-12)

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        ab = self.point2 - self.point1
        ab = ab / (np.linalg.norm(ab) + 1e-12)
        normal = self.GetNormal(point)
        tangent = np.cross(normal, ab)
        return tangent / (np.linalg.norm(tangent) + 1e-12)

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

    def ConvexHull(self, resolution: int = 100) -> List[np.ndarray]:
        raise NotImplementedError("Capsule convex hull not yet implemented")

    def __repr__(self):
        return f"Capsule(point1={self.point1}, point2={self.point2}, radius={self.radius})"

# VObject & Factories
@dataclass
class VObject:
    """Visual object combining shape, transform, material, and name."""

    shape: Shape
    transform: Optional["Transform"] = None
    material: Optional["Material"] = None
    texture: Optional[str] = None
    name: str = "VObject"

    def __post_init__(self):
        if self.transform is None:
            self.transform = Transform(np.zeros(3), np.zeros(3), np.ones(3))

    @property
    def position(self) -> np.ndarray:
        return self.transform.position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self.transform.position = np.asarray(value, dtype=float)
        self.shape.transform.position = self.transform.position

    def __repr__(self):
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

"""
Geometry Module: Defines geometric shapes, their properties, and interactions.
Provides base classes, mixins, and concrete implementations for 2D and 3D shapes,
including ray intersection and surface property queries.

Classes:
- Shape: Abstract base class for all shapes.
- Shape2D, Shape3D: Base classes for 2D and 3D shapes.
- Circle, Triangle, Polygon: Concrete 2D shape implementations.
- Sphere, Cube, Prism, Pyramid, Capsule: Concrete 3D shape implementations.
- GeometryMixin, RayIntersectionMixin, SurfacePropertiesMixin: Mixins for geometric queries.
- VObject: Combines shape with transform and material for rendering.
- ShapeFactory and its subclasses: Factories for creating shape instances. 
"""