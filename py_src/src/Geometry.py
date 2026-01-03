from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass, field

from CommonUtils import unit
from PrimaryStructures import Transform, Ray, TracingRay
from Luminance import PBRMaterial

class Shape(ABC):
    """
    Abstract base for all shapes.
    Provides transform, material, and naming.
    """
    def __init__(
            self,
            transform: Optional["Transform"] = None, 
            material: Optional["PBRMaterial"] = None,
            name: str = "Shape",
            **kwargs
        ):
        # Default transform if none provided
        self.transform = transform if transform else Transform(np.zeros(3), np.array([0, 0, 1]), np.ones(3)) 
        self.material = material
        self.name = name
        
        # dynamic attribute assignment
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def origin(self) -> np.ndarray:
        return self.transform.position
    
    @classmethod
    @abstractmethod
    def unit_signed_distance(cls, point: np.ndarray) -> float:
        """
        Signed distance for a unit version of the shape.
        (e.g., Unit Sphere is radius 1 at origin).
        """
        raise NotImplementedError

    @abstractmethod
    def signed_distance(self, point: np.ndarray) -> float:
        """
        Signed distance for points in world space.
        (negative inside, positive outside).
        """
        raise NotImplementedError
    
    def inverse_signed_distance(
        self,
        origin_world: np.ndarray, 
        dir_world: np.ndarray,
        max_steps: int = 64,
        max_distance: float = 10,
        epsilon: float = 1e-4
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # --- STEP 1: Transform Ray to Local Space ---
        # 1. Transform Origin (Point: w=1.0)
        origin_local = self.transform.inverse_transform_point(origin_world)
        
        # 2. Transform Direction (Vector: w=0.0)
        # We only need the rotation/scale 3x3 block for vectors
        dir_local = unit(self.transform.inverse_transform_direction(dir_world))

        # --- STEP 2: Raymarch in Local Space (Unit Sphere) ---
        t = 0.0
        entry_point_local = np.zeros(3)
        hit_entry = False

        # A. Find Entry Point (Front Face)
        for _ in range(max_steps):
            entry_point_local = origin_local + (dir_local * t)
            dist = -self.unit_signed_distance(entry_point_local)
            
            # Optimization: If we are too far away, we missed
            if dist > max_distance:
                return None 

            if dist < epsilon:
                hit_entry = True
                break
            t += dist

        # If we never hit the entry, we can't find an exit!
        if not hit_entry:
            return None
        entry_point = self.transform.transform_point(entry_point_local)

        # B. Find Exit Point (Back Face)
        # Push slightly inside the sphere to start the internal march
        t += epsilon * 2.0
        exit_point_local = np.array([0.0, 0.0, 0.0])
        
        for _ in range(max_steps):
            exit_point_local = origin_local + (dir_local * t)
            
            # INVERTED SDF: We want distance to the "outer shell" from inside.
            # Inside a unit sphere, dist is negative. -dist makes it positive.
            dist = -self.unit_signed_distance(exit_point_local)
            
            if dist < epsilon:
                # Apply World Matrix
                exit_point = self.transform.transform_point(exit_point_local)
                
                # Return 3D (x,y,z)
                return (entry_point, exit_point)
                
            t += dist
            
        return None
    
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
        
        # Adaptive step size loop
        for _ in range(max_iterations):
            dist = self.signed_distance(current)
            if abs(dist) < 1e-6:
                return current
            
            # Approximate gradient via finite differences
            eps = 1e-5
            dx = (self.signed_distance(current + np.array([eps, 0, 0])) - dist) / eps
            dy = (self.signed_distance(current + np.array([0, eps, 0])) - dist) / eps
            dz = (self.signed_distance(current + np.array([0, 0, eps])) - dist) / eps
            
            grad = np.array([dx, dy, dz])
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            else:
                break
            
            # Move towards surface
            current = current - dist * grad
        
        return current
    
    def get_min_uniform_scale(self):
        # Assuming you store scale as a Vector3 or (sx, sy, sz)
        # We need the smallest scale component to ensure we don't overstep.
        s = self.transform.scale
        return min(s[0], s[1], s[2])

    @abstractmethod
    def get_normal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        """
        Get a vector tangent to the surface at the point.
        """
        raise NotImplementedError
    
    def get_binormal(self, point: np.ndarray, bias: float = 1e-12) -> np.ndarray:
        """
        Bitangent/Binormal vector. 
        Calculated via cross product of Normal and Tangent.
        """
        normal = self.get_normal(point)
        tangent = self.get_tangent(point)
        bn = np.cross(normal, tangent)
        return bn / (np.linalg.norm(bn) + bias)

    @abstractmethod
    def check_ray_intersection(self, ray: "Ray") -> bool:
        """Return True if the Ray intersects the shape."""
        raise NotImplementedError

    @abstractmethod
    def get_ray_intersections(self, ray: "Ray") -> List[np.ndarray]:
        """Return a list of all intersection points sorted by distance."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """2 for 2D, 3 for 3D."""
        raise NotImplementedError

    # --- Transformation Wrappers ---
    
    def translate(self, offset: np.ndarray):
        self.transform.translate(offset, space="global")

    def rotate(self, angle: float, axis: np.ndarray):
        self.transform.rotate(angle, axis, space="global")

    def enlarge(self, factor: Union[float, np.ndarray]):
        self.transform.enlarge(factor, space="global")
    
    def scale(self, factor: Union[float, np.ndarray]):
        self.enlarge(factor)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', pos={self.origin})"

# 2D Shapes
class Shape2D(Shape):
    """Base for 2D shapes (Planes, Disks, Triangles)."""
    
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
    def __init__(self, center: np.ndarray, normal: np.ndarray, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.transform.position = np.asarray(center, dtype=float)
        self.normal = unit(np.asarray(normal, dtype=float))
        if radius <= 0: raise ValueError("Radius must be > 0")
        self.radius = float(radius)

    @classmethod
    def unit_signed_distance(cls, point: np.ndarray) -> float:
        """Unit Disk SDF (Radius 1, XY Plane)."""
        xy_dist = np.linalg.norm(point[:2])
        return np.sqrt(max(0.0, xy_dist - 1.0)**2 + point[2]**2)

    def signed_distance(self, point: np.ndarray) -> float:
        p = point - self.transform.position
        dist_plane = np.dot(p, self.normal)
        p_proj = p - dist_plane * self.normal
        # Distance to edge of disk
        dist_edge = max(0.0, np.linalg.norm(p_proj) - self.radius)
        return np.sqrt(dist_edge**2 + dist_plane**2)

    def convex_hull(self, resolution: int = 32) -> List[np.ndarray]:
        """
        Generates a polygon approximation of the circle.
        Resolution determines the number of vertices.
        """
        # 1. Create Basis Vectors (Orthonormal to Normal)
        if abs(self.normal[2]) < 0.9:
            tangent = np.cross(self.normal, np.array([0, 0, 1]))
        else:
            tangent = np.cross(self.normal, np.array([1, 0, 0]))
        tangent = unit(tangent)
        bitangent = np.cross(self.normal, tangent)

        # 2. Generate points around the circle
        hull_points = []
        for i in range(resolution):
            theta = 2 * np.pi * i / resolution
            # P = C + r * (cos(t)*Tangent + sin(t)*Bitangent)
            offset = self.radius * (np.cos(theta) * tangent + np.sin(theta) * bitangent)
            hull_points.append(self.transform.position + offset)
        
        return hull_points

    # ... (Include previous check_ray_intersection, get_ray_intersections, get_normal, etc.) ...
    def check_ray_intersection(self, ray: Ray) -> bool:
        # (Implementation from previous response)
        denom = np.dot(self.normal, ray.orientation)
        if abs(denom) < 1e-6: return False
        t = np.dot(self.transform.position - ray.origin, self.normal) / denom
        if t < 0: return False
        return np.linalg.norm(ray.point_at(t) - self.transform.position)**2 <= self.radius**2

    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        if self.check_ray_intersection(ray):
            denom = np.dot(self.normal, ray.orientation)
            t = np.dot(self.transform.position - ray.origin, self.normal) / denom
            return [ray.point_at(t)]
        return []

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        # Tangent along the circle edge direction
        radial = point - self.transform.position
        if np.linalg.norm(radial) < 1e-6: return unit(np.cross(self.normal, np.array([1,0,0])))
        return unit(np.cross(self.normal, radial))

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2

    @property
    def perimeter(self) -> float:
        return 2 * np.pi * self.radius


class Triangle(Shape2D):
    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.v1 = np.asarray(v1, dtype=float)
        self.v2 = np.asarray(v2, dtype=float)
        self.v3 = np.asarray(v3, dtype=float)
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        self.normal = unit(np.cross(edge1, edge2))
        self.transform.position = (self.v1 + self.v2 + self.v3) / 3.0

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

    @classmethod
    def unit_signed_distance(cls, point: np.ndarray) -> float:
        return max(abs(point[2]), max(point[0], max(point[1], 1 - point[0] - point[1])) if point[0]>0 and point[1]>0 else 0)

    def convex_hull(self, resolution: int = 0) -> List[np.ndarray]:
        """Triangle convex hull is just its vertices."""
        return [self.v1, self.v2, self.v3]

    def signed_distance(self, point: np.ndarray) -> float:
        # (Implementation from previous response)
        # ... logic for distance to triangle ...
        return 0.0 # Placeholder: Insert full logic from previous step

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

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        return unit(self.v2 - self.v1)

    @property
    def area(self) -> float:
        return 0.5 * np.linalg.norm(np.cross(self.v2 - self.v1, self.v3 - self.v1))

    @property
    def perimeter(self) -> float:
        return (np.linalg.norm(self.v2 - self.v1) + np.linalg.norm(self.v3 - self.v2) + np.linalg.norm(self.v1 - self.v3))

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

class Shape3D(Shape):
    """Base for 3D shapes (Spheres, Cubes, Meshes)."""
    
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
        """Alias for surface area to maintain API compatibility."""
        return self.surface_area

    def convex_hull(self, resolution: int = 100) -> List[np.ndarray]:
        """
        Approximate convex hull. 
        Should return a list of vertices defining the hull.
        """
        raise NotImplementedError

class Sphere(Shape3D):
    def __init__(self, center: np.ndarray, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.transform.position = np.asarray(center, dtype=float)
        self.radius = float(radius)

    @classmethod
    def unit_signed_distance(cls, point: np.ndarray) -> float:
        return np.linalg.norm(point) - 1.0

    def signed_distance(self, point: np.ndarray) -> float:
        # World space approximation
        d = np.linalg.norm(point - self.transform.position) - self.radius
        return d * min(self.transform.scale[0], self.transform.scale[0], self.transform.scale[0])

    def convex_hull(self, resolution: int = 100) -> List[np.ndarray]:
        """
        Generates points on the sphere surface using a Fibonacci Lattice.
        This provides an evenly distributed point cloud.
        """
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden Ratio
        
        for i in range(resolution):
            y = 1 - (i / float(resolution - 1)) * 2  # y goes from 1 to -1
            radius_at_y = np.sqrt(max(0, 1 - y * y)) # radius at y
            
            theta = 2 * np.pi * i / phi
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            # 1. Create local point scaled by radius
            local_pt = np.array([x, y, z]) * self.radius
            
            # 2. Transform to world space (handles rotation/position)
            world_pt = self.transform.transform_point(local_pt)
            points.append(world_pt)
            
        return points

    # Standard sphere methods
    def check_ray_intersection(self, ray: Ray) -> bool:
        # Solve quadratic for a, b, c
        oc = ray.origin - self.transform.position
        d = ray.orientation
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c = np.dot(oc, oc) - (self.radius ** 2)

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False

        sqrt_disc = np.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)

        # If both intersections are behind the ray origin, no hit
        return not (t0 < 0.0 and t1 < 0.0)

    def get_ray_intersections(self, ray: Ray) -> List[np.ndarray]:
        oc = ray.origin - self.transform.position
        d = ray.orientation
        a = np.dot(d, d)
        b = 2.0 * np.dot(oc, d)
        c = np.dot(oc, oc) - (self.radius ** 2)

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return []

        sqrt_disc = np.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)

        points: List[np.ndarray] = []
        if t0 >= 0.0:
            points.append(ray.point_at(t0))
        if t1 >= 0.0:
            points.append(ray.point_at(t1))
        return points

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return unit(point - self.transform.position)

    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        n = self.get_normal(point)
        arb = np.array([0, 1, 0]) if abs(n[1]) < 0.9 else np.array([1, 0, 0])
        return unit(np.cross(n, arb))

    @property
    def volume(self) -> float:
        return (4/3) * np.pi * self.radius**3

    @property
    def surface_area(self) -> float:
        return 4 * np.pi * self.radius**2


class Cube(Shape3D):
    def __init__(self, center: np.ndarray, side_length: float, **kwargs):
        super().__init__(**kwargs)
        self.transform.position = np.asarray(center, dtype=float)
        self.side_length = float(side_length)

    @classmethod
    def unit_signed_distance(cls, point: np.ndarray) -> float:
        # Unit Cube (side 1)
        d = np.abs(point) - 0.5
        return np.linalg.norm(np.maximum(d, 0.0)) + min(max(d[0], max(d[1], d[2])), 0.0)

    def signed_distance(self, point: np.ndarray) -> float:
        # OBB Distance
        local_p = self.transform.inverse_transform_point(point)
        d = np.abs(local_p) - (self.side_length / 2.0)
        dist = np.linalg.norm(np.maximum(d, 0.0)) + min(max(d[0], max(d[1], d[2])), 0.0)
        return dist * self.get_min_uniform_scale()

    def convex_hull(self, resolution: int = 0) -> List[np.ndarray]:
        """
        Returns the 8 corners of the Cube (OBB).
        Resolution is ignored as a cube strictly has 8 vertices.
        """
        half = self.side_length / 2.0
        corners = []
        
        # Iterate all combinations of +/- half_size
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    local_pt = np.array([x * half, y * half, z * half])
                    world_pt = self.transform.transform_point(local_pt)
                    corners.append(world_pt)
                    
        return corners

    def check_ray_intersection(self, ray: Ray, local_min: float = -0.5, local_max: float = 0.5) -> bool:
        """
        Checks intersection using the Slab Method against a local AABB.
        Defaults to a unit cube centered at origin (-0.5 to 0.5).
        """
        # 1. Transform Ray to Local Space
        # Note: We do NOT normalize the local_dir. This allows 't' to match world units.
        local_origin = self.inverse_transform_point(ray.origin)
        local_dir = self.inverse_transform_direction(ray.direction, normalize=False)
        
        # 2. Slab Method Setup
        # Avoid division by zero: replace 0 with a tiny epsilon or use numpy's inf handling
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / local_dir
            
        t1 = (local_min - local_origin) * inv_dir
        t2 = (local_max - local_origin) * inv_dir

        # 3. Find intersection interval
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)
        
        # Largest 'entry' time and smallest 'exit' time across all axes
        t_enter = np.max(t_min)
        t_exit  = np.min(t_max)

        # 4. Check validity
        # Hit if t_exit >= t_enter and the hit is in front of the ray (t_exit >= 0)
        return t_exit >= t_enter and t_exit >= 0

    def get_ray_intersections(self, ray: Ray, local_min: float = -0.5, local_max: float = 0.5) -> List[np.ndarray]:
        """
        Returns a list of intersection points (in World Space) using the Slab Method.
        Returns 0, 1, or 2 points sorted by distance.
        """
        # 1. Transform Ray to Local Space
        local_origin = self.inverse_transform_point(ray.origin)
        local_dir = self.inverse_transform_direction(ray.direction, normalize=False)

        # 2. Slab Method
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / local_dir
        
        t1 = (local_min - local_origin) * inv_dir
        t2 = (local_max - local_origin) * inv_dir
        
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)
        
        t_enter = np.max(t_min)
        t_exit  = np.min(t_max)
        
        # 3. Validate Intersection
        if t_exit < t_enter or t_exit < 0:
            return []
            
        intersections = []
        
        # 4. Calculate World Points
        # Because we didn't normalize local_dir, these 't' values apply directly to the World Ray.
        # P_world = Origin_world + t * Direction_world
        
        # Check entry point (must be >= 0 to be valid)
        if t_enter >= 0:
            p_enter = ray.origin + (ray.direction * t_enter)
            intersections.append(p_enter)
            
        # Check exit point (only if it's distinct from entry, e.g. not a grazing edge, and valid)
        # Using a small epsilon for float comparison
        if t_exit >= 0 and (len(intersections) == 0 or abs(t_exit - t_enter) > 1e-6):
            p_exit = ray.origin + (ray.direction * t_exit)
            intersections.append(p_exit)
            
        return intersections

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        # Box normal logic
        local_p = self.transform.inverse_transform_point(point)
        dominant = np.argmax(np.abs(local_p))
        n = np.zeros(3); n[dominant] = np.sign(local_p[dominant])
        return self.transform.transform_normal(n)

    def get_tangent(self, point: np.ndarray) -> np.ndarray:
        n = self.get_normal(point)
        t = np.cross(n, np.array([0,1,0])) if abs(n[1]) < 0.9 else np.cross(n, np.array([1,0,0]))
        return unit(t)

    @property
    def volume(self) -> float:
        return self.side_length ** 3

    @property
    def surface_area(self) -> float:
        return 6 * (self.side_length ** 2)

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
    """
    Visual object combining shape, transform, and material.
    Acts as a Node in the Scene Graph.
    """
    shape: Optional["Shape"] = None # Optional so we can have empty "Group" nodes
    transform: Optional["Transform"] = None
    material: Optional["PBRMaterial"] = None
    name: str = "VObject"

    children: List["VObject"] = field(default_factory=list)
    parent: Optional["VObject"] = field(default=None, repr=False) # Exclude from repr to avoid recursion

    def __post_init__(self):
        if self.transform is None:
            # Assuming Transform is defined
            self.transform = Transform(np.zeros(3), np.array([0, 0, 1]), np.ones(3))
        
        # Ensure the shape (if present) shares this transform or follows it
        # Depending on your architecture, you might want: 
        if self.shape:
             self.shape.transform = self.transform

    def add_child(self, child: "VObject"):
        """Properly links a child to this parent."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: "VObject"):
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def __repr__(self):
        shape_name = self.shape.name if self.shape else "None"
        return f"VObject(name='{self.name}', shape={shape_name}, children={len(self.children)})"

class ShapeFactory(ABC):
    """Abstract factory for creating shapes."""
    @abstractmethod
    def create(self, **kwargs) -> Shape:
        raise NotImplementedError

class CircleFactory(ShapeFactory):
    def create(self, center: np.ndarray, radius: float, **kwargs) -> Circle:
        # Circle ctor expects (center, normal, radius); default normal is +Z
        return Circle(center, np.array([0.0, 0.0, 1.0]), radius, **kwargs)

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

# --- 1. AABB Helper Class (The "Box" logic) ---
class AABB:
    """
    Axis-Aligned Bounding Box.
    Used for quick rejection tests before checking complex geometry.
    """
    __slots__ = ['min_point', 'max_point']

    def __init__(self, min_point: np.ndarray, max_point: np.ndarray):
        self.min_point = min_point
        self.max_point = max_point

    def intersect(self, ray: TracingRay) -> float:
        """
        Slab Method for Ray/AABB intersection.
        Returns distance to entry, or infinity if miss.
        """
        # We use the inverse direction to replace division with multiplication
        # This handles division by zero gracefully (results in +/- inf)
        inv_dir = 1.0 / (ray.orientation + 1e-9) 
        
        t0 = (self.min_point - ray.origin) * inv_dir
        t1 = (self.max_point - ray.origin) * inv_dir

        tmin = np.maximum(np.minimum(t0, t1), 0.0)
        tmax = np.minimum(np.maximum(t0, t1), 1e30)

        # Find largest entry time and smallest exit time across all axes
        t_enter = np.max(tmin)
        t_exit = np.min(tmax)

        if t_exit >= t_enter:
            return t_enter
        
        return float('inf')

    @staticmethod
    def from_object(obj: VObject) -> 'AABB':
        """
        Calculates the world-space AABB for a given object.
        """
        # Get the object's local bounds (e.g., Sphere is [-r, -r, -r] to [r, r, r])
        # This assumes your Shape classes have a 'get_bounds()' method.
        # Fallback: Approximate with a unit cube scaled by transform
        
        # 1. Get Transform Matrix
        matrix = obj.transform.get_global_matrix()
        
        # 2. Define the 8 corners of a generic Unit Cube (-0.5 to 0.5) or Shape bounds
        # Note: You should add get_local_bounds() to your Shape classes for tighter fits.
        corners = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1],  [1, -1, 1],  [-1, 1, 1],  [1, 1, 1]
        ]) * 0.5 # Unit cube logic
        
        # If shape is sphere, radius is usually 1.0 before scaling
        if hasattr(obj.shape, 'radius'):
            corners *= 2.0 # Scale unit cube to wrap radius 1.0 sphere

        # 3. Transform corners to world space
        # (Append 1 for homogeneous coords)
        ones = np.ones((8, 1))
        corners_4d = np.hstack([corners, ones]) 
        world_corners = (matrix @ corners_4d.T).T[:, :3]

        # 4. Find min/max of transformed corners
        min_p = np.min(world_corners, axis=0) - 0.01 # Small padding
        max_p = np.max(world_corners, axis=0) + 0.01
        
        return AABB(min_p, max_p)

    @staticmethod
    def union(box_a: 'AABB', box_b: 'AABB') -> 'AABB':
        return AABB(
            np.minimum(box_a.min_point, box_b.min_point),
            np.maximum(box_a.max_point, box_b.max_point)
        )

# --- 2. The Node Structure ---
class BVHNode:
    def __init__(self, objects: list[VObject]):
        self.left: BVHNode | None = None
        self.right: BVHNode | None = None
        self.box: AABB | None = None
        self.objects: list[VObject] = [] # Only leaf nodes have objects

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