from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, List, Tuple, cast
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
            name: str = "Shape",
            **kwargs
        ):
        self.name = name
        
        # dynamic attribute assignment
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def signed_distance(self, local_point: np.ndarray) -> float:
        """
        Signed distance for a point in LOCAL space.
        (negative inside, positive outside).
        """
        raise NotImplementedError
    
    def unit_signed_distance(self, local_point: np.ndarray) -> float:
        """
        Optional helper. Usually aliases signed_distance unless 
        the shape has specific 'unit' logic distinct from its parameters.
        """
        return self.signed_distance(local_point)

    # --- 2. Derivatives (Local Space) ---

    def get_normal(self, local_point: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """
        Calculates the normal in LOCAL Space using Finite Differences.
        Can be overridden by shapes with analytical normals (like Sphere).
        """
        # Gradient approximation
        dx = np.array([epsilon, 0, 0])
        dy = np.array([0, epsilon, 0])
        dz = np.array([0, 0, epsilon])
        
        d = self.signed_distance(local_point)
        
        grad = np.array([
            self.signed_distance(local_point + dx) - d,
            self.signed_distance(local_point + dy) - d,
            self.signed_distance(local_point + dz) - d
        ])
        
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 0 else np.array([0.0, 1.0, 0.0])

    @abstractmethod
    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        """
        Get UV coordinates (0-1) for texture mapping.
        """
        raise NotImplementedError

    # --- 3. Analytical Intersection (Local Space) ---

    @abstractmethod
    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        """
        Analytical intersection (Ray-Sphere, etc).
        Returns list of hit points in LOCAL SPACE.
        """
        raise NotImplementedError

    # --- 4. Volumetric Helpers (Local Space) ---

    def get_entry_exit_points(
        self,
        local_ray: "Ray", 
        max_steps: int = 64,
        max_dist: float = 20.0,
        epsilon: float = 1e-3
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Raymarches through the SDF to find entry and exit points.
        Useful for volumetric rendering or transparency.
        
        Expects: local_ray (transformed by caller).
        Returns: (entry_point_local, exit_point_local) or None.
        """
        
        # A. Find Entry (March Forward)
        t = 0.0
        entry_t = -1.0
        
        for _ in range(max_steps):
            p = local_ray.point_at(t)
            d = self.signed_distance(p)
            
            if d < epsilon:
                entry_t = t
                break
            
            t += d
            if t > max_dist:
                return None # Missed completely

        if entry_t < 0:
            return None

        # B. Find Exit (March "Inside")
        # To find exit, we push slightly inside and march using inverted distance
        # or simply fixed steps until d > 0 again.
        
        # Simple method: Continue marching until d > epsilon again
        # (This is a simplified approach; true "interior marching" needs inverted SDF)
        exit_t = entry_t
        t = entry_t + epsilon * 2.0 # Step inside
        
        for _ in range(max_steps):
            p = local_ray.point_at(t)
            d = self.signed_distance(p) # Negative inside
            
            # If d becomes positive, we have exited
            if d > epsilon:
                exit_t = t
                break
                
            # Inside the shape, d is negative. 
            # We want to move towards the boundary (d=0).
            # We move by |d| (abs distance to surface).
            step = abs(d) 
            # Safety for deep interior
            if step < epsilon: step = epsilon 
            
            t += step
            if t > max_dist:
                break
        
        return (local_ray.point_at(entry_t), local_ray.point_at(exit_t))

    # --- 5. Geometric Queries ---

    def check_point_inside(self, local_point: np.ndarray, epsilon: float = 1e-6) -> bool:
        return self.signed_distance(local_point) < -epsilon

    def get_closest_point(self, local_point: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Gradient descent to find closest surface point.
        """
        current = np.array(local_point, dtype=float)
        
        for _ in range(max_iterations):
            dist = self.signed_distance(current)
            if abs(dist) < 1e-6:
                return current
            
            grad = self.get_normal(current)
            # Move exactly 'dist' along the gradient towards surface
            current = current - dist * grad
        
        return current

    def __repr__(self):
        return f"Shape({self.name})"

# 2D Shapes
class Shape2D(Shape):
    """
    Base for 2D shapes (Planes, Disks, Triangles).
    Strictly defined on the Local XY Plane (z=0) where possible.
    """
    @property
    def dimensions(self) -> int:
        return 2

    # A 2D shape has no volume
    def volume(self) -> float:
        return 0.0
    
    # 2D Shapes are essentially 0-thickness in Z
    def unit_signed_distance(self, local_point: np.ndarray) -> float:
        # Default fallback: Treat as flat plane with thickness 0
        return abs(local_point[2])

class Circle(Shape2D):
    def __init__(self, radius: float, **kwargs):
        """
        A Disk on the XY plane.
        To position/rotate it, apply a Transform to the VObject.
        """
        super().__init__(**kwargs)
        if radius <= 0: raise ValueError("Radius must be > 0")
        self.radius = float(radius)

    def signed_distance(self, local_point: np.ndarray) -> float:
        """SDF for a Disk on the XY plane."""
        # 2D distance from center (0,0)
        q = np.linalg.norm(local_point[:2]) - self.radius
        
        # Sqrt( d_planar^2 + z^2 )
        # We use max(q, 0) because inside the disk, planar distance is 0 
        # (unless we want to treat it as a thin plate)
        return np.sqrt(max(q, 0.0)**2 + local_point[2]**2)

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # Intersect with Plane Z=0
        if abs(local_ray.orientation[2]) < 1e-6:
            return []
            
        t = -local_ray.origin[2] / local_ray.orientation[2]
        
        if t < 0: return []
        
        p = local_ray.point_at(t)
        
        # Check if point is within radius
        if np.dot(p[:2], p[:2]) <= self.radius**2:
            return [p]
            
        return []

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        # Normal is always +Z in local space
        # (Double-sided rendering might flip this at the shader level)
        return np.array([0.0, 0.0, 1.0])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Planar mapping
        u = (local_point[0] / self.radius) * 0.5 + 0.5
        v = (local_point[1] / self.radius) * 0.5 + 0.5
        return np.array([u, v])

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2

class Triangle(Shape2D):
    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Calculate Centroid (World Space)
        centroid = (v1 + v2 + v3) / 3.0
        
        # 2. Store Vertices relative to Centroid (Local Space)
        # Note: We are NOT rotating the triangle to lie flat on XY here to keep it simple.
        # It exists in 3D local space, just centered.
        self.v1 = v1 - centroid
        self.v2 = v2 - centroid
        self.v3 = v3 - centroid
        
        # Pre-calculate normal and edge vectors for intersection tests
        self.edge1 = self.v2 - self.v1
        self.edge2 = self.v3 - self.v1
        self.normal = np.cross(self.edge1, self.edge2)
        self.normal /= np.linalg.norm(self.normal)

        # NOTE: The VObject creating this shape MUST set its transform.position to 'centroid'

    def signed_distance(self, p: np.ndarray) -> float:
        # Basic UDTriangle (Unsigned Distance) logic
        # Adapted from Inigo Quilez
        ba = self.v2 - self.v1; pa = p - self.v1
        cb = self.v3 - self.v2; pb = p - self.v2
        ac = self.v1 - self.v3; pc = p - self.v3
        
        nor = self.normal

        # Clamp to edges
        d = min(min(
            np.dot(ba * np.clip(np.dot(ba, pa) / np.dot(ba, ba), 0.0, 1.0) - pa, 
                   ba * np.clip(np.dot(ba, pa) / np.dot(ba, ba), 0.0, 1.0) - pa),
            np.dot(cb * np.clip(np.dot(cb, pb) / np.dot(cb, cb), 0.0, 1.0) - pb, 
                   cb * np.clip(np.dot(cb, pb) / np.dot(cb, cb), 0.0, 1.0) - pb)),
            np.dot(ac * np.clip(np.dot(ac, pc) / np.dot(ac, ac), 0.0, 1.0) - pc, 
                   ac * np.clip(np.dot(ac, pc) / np.dot(ac, ac), 0.0, 1.0) - pc))
        
        d = np.sqrt(d)
        
        # If we are effectively "above/below" the triangle, use normal distance
        # (Simplification: Real 3D signed distance to a flat polygon is complex)
        return d

    def get_ray_intersections(self, ray: "Ray") -> List[np.ndarray]:
        # Moller-Trumbore algorithm (Local Space)
        h = np.cross(ray.orientation, self.edge2)
        a = np.dot(self.edge1, h)

        if abs(a) < 1e-8:
            return []

        f = 1.0 / a
        s = ray.origin - self.v1
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return []

        q = np.cross(s, self.edge1)
        v = f * np.dot(ray.orientation, q)

        if v < 0.0 or u + v > 1.0:
            return []

        t = f * np.dot(self.edge2, q)

        if t > 1e-8:
            return [ray.point_at(t)]
            
        return []

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        return self.normal

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Barycentric UVs would require storing UVs per vertex.
        # Fallback to planar projection.
        return local_point[:2]

    @property
    def area(self) -> float:
        return float(0.5 * np.linalg.norm(np.cross(self.edge1, self.edge2)))

class Polygon(Shape2D):
    def __init__(self, vertices: List[np.ndarray], **kwargs):
        super().__init__(**kwargs)
        self.vertices = [np.asarray(v, dtype=float) for v in vertices]
        if len(self.vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices")

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
        return float(min_dist)

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
        return float(abs(area) / 2)

    @property
    def perimeter(self) -> float:
        perim = 0
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % len(self.vertices)]
            perim += np.linalg.norm(v2 - v1)
        return float(perim)

    def _repr__(self):
        return f"Polygon({len(self.vertices)} vertices)"

class Plane(Shape2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def signed_distance(self, local_point: np.ndarray) -> float:
        # Distance to Z=0 plane is just Z height
        return local_point[2]

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        denom = local_ray.orientation[2]
        if abs(denom) < 1e-6:
            return []
            
        t = -local_ray.origin[2] / denom
        if t >= 0:
            return [local_ray.point_at(t)]
        return []

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Infinite tiling UVs
        return local_point[:2]

class ClippedPlane(Shape2D):
    def __init__(self, clip_polygon: List[np.ndarray], **kwargs):
        """
        clip_polygon: List of 2D points (x, y) defining the clip boundary 
                      on the local plane.
        """
        super().__init__(**kwargs)
        if len(clip_polygon) < 3:
            raise ValueError("Clip polygon must have at least 3 vertices")
        
        # Ensure points are 2D arrays
        self.clip_polygon = [np.array(v[:2]) for v in clip_polygon]

    def signed_distance(self, local_point: np.ndarray) -> float:
        # 1. Distance to the infinite plane
        dist_plane = local_point[2]
        
        # 2. Distance to the 2D polygon edge (SDF for 2D Poly)
        # (Simplified: exact 2D poly SDF is expensive, usually we check bounds)
        # If we just want intersection, exact SDF isn't strictly required unless raymarching.
        
        # Return plane distance for basic behavior, 
        # but the Ray Intersection will handle the clipping.
        return dist_plane

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # 1. Intersect Infinite Plane
        denom = local_ray.orientation[2]
        if abs(denom) < 1e-6: return []
            
        t = -local_ray.origin[2] / denom
        if t < 0: return []
        
        hit_point = local_ray.point_at(t)
        
        # 2. Check Point-in-Polygon (2D)
        if self._is_point_in_poly(hit_point[:2]):
            return [hit_point]
            
        return []

    def _is_point_in_poly(self, p2d: np.ndarray) -> bool:
        # Ray casting algorithm for Point in Polygon
        inside = False
        n = len(self.clip_polygon)
        x, y = p2d
        
        for i in range(n):
            x1, y1 = self.clip_polygon[i]
            x2, y2 = self.clip_polygon[(i + 1) % n]
            
            if ((y1 > y) != (y2 > y)) and \
               (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-10) + x1):
                inside = not inside
                
        return inside
    
    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        return local_point[:2]

class Shape3D(Shape):
    """
    Base for 3D shapes (Spheres, Boxes, Meshes).
    Defined strictly in Local Space.
    """
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

    def convex_hull(self) -> List[np.ndarray]:
        """
        Returns a list of vertices defining the shape's convex hull.
        Returns LOCAL coordinates. 
        (The VObject is responsible for transforming these to World Space).
        """
        raise NotImplementedError
    
class Sphere(Shape3D):
    def __init__(self, radius: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if radius <= 0: raise ValueError("Radius must be > 0")
        self.radius = float(radius)

    # --- SDF (Local Space) ---
    def signed_distance(self, local_point: np.ndarray) -> float:
        # Simple distance from origin minus radius
        return float(np.linalg.norm(local_point) - self.radius)

    # --- Analytical Intersection (Local Space) ---
    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # Ray-Sphere intersection at (0,0,0)
        # quadratic: t^2(d.d) + 2t(o.d) + (o.o - r^2) = 0
        
        # d is often normalized in local space, but if we have non-uniform scale 
        # passed down, it might not be. We assume standard ray behavior here.
        
        oc = local_ray.origin  # since center is 0,0,0
        
        a = np.dot(local_ray.orientation, local_ray.orientation)
        b = 2.0 * np.dot(oc, local_ray.orientation)
        c = np.dot(oc, oc) - self.radius**2
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return []
            
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        hits = []
        if t1 >= 0: hits.append(local_ray.point_at(t1))
        if t2 >= 0: hits.append(local_ray.point_at(t2))
        
        return hits

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        # Normal of a sphere at origin is just the normalized point position
        dist = np.linalg.norm(local_point)
        if dist < 1e-6: return np.array([0, 1, 0]) # Degenerate case
        return local_point / dist

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Spherical UV mapping
        # u = 0.5 + arctan2(z, x) / 2pi
        # v = 0.5 + arcsin(y) / pi
        n = local_normal
        u = 0.5 + np.arctan2(n[2], n[0]) / (2 * np.pi)
        v = 0.5 + np.arcsin(n[1]) / np.pi
        return np.array([u, v])

    def convex_hull(self, resolution: int = 12) -> List[np.ndarray]:
        # Returns an Icosahedron or low-res sphere approximation (Local Space)
        # Using Fibonacci Lattice for even distribution
        points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(resolution):
            y = 1 - (i / float(resolution - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append(np.array([x, y, z]) * self.radius)
        return points

    @property
    def volume(self) -> float:
        return (4/3) * np.pi * self.radius**3

    @property
    def surface_area(self) -> float:
        return 4 * np.pi * self.radius**2

class Cube(Shape3D):
    def __init__(self, size: Union[float, np.ndarray] = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        # Internal storage is half-extents (distance from center to edge)
        if isinstance(size, (int, float)):
            self.half_size = np.array([size, size, size]) / 2.0
        else:
            self.half_size = np.asarray(size, dtype=float) / 2.0

    # --- SDF (Local Space) ---
    def signed_distance(self, local_point: np.ndarray) -> float:
        # Signed Distance to a Box (Inigo Quilez)
        # d = |p| - r
        q = np.abs(local_point) - self.half_size
        
        # Outside distance + Inside distance
        return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

    # --- Analytical Intersection (Local Space) ---
    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # Slab Method (AABB intersection)
        
        # Prepare bounds
        b_min = -self.half_size
        b_max = self.half_size
        
        # Avoid div by zero
        inv_dir = np.zeros_like(local_ray.orientation)
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / local_ray.orientation
        
        # Handle Inf cases for orthogonal rays
        inv_dir[np.isinf(inv_dir)] = 1e30 # Large number, not Inf
        
        t1 = (b_min - local_ray.origin) * inv_dir
        t2 = (b_max - local_ray.origin) * inv_dir
        
        t_min = np.maximum(np.minimum(t1, t2), 0.0) # Element-wise max/min
        t_max = np.minimum(np.maximum(t1, t2), 1e30)
        
        # Find largest entry and smallest exit
        t_enter = np.max(t_min)
        t_exit = np.min(t_max)
        
        if t_exit < t_enter or t_exit < 0:
            return []
            
        hits = []
        if t_enter >= 0: hits.append(local_ray.point_at(t_enter))
        # Only add exit if it's distinct (and we assume solid, so usually we just want entry)
        # But for transparency, we might want both.
        if t_exit > t_enter: hits.append(local_ray.point_at(t_exit))
            
        return hits

    def get_normal(self, local_point: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
        # Normalize point relative to box dimensions
        # Add tiny epsilon to avoid division by zero
        p = local_point / (self.half_size + epsilon)
        
        # Softmax-ish approach for cleaner edges, or just argmax
        abs_p = np.abs(p)
        max_axis = np.argmax(abs_p)
        
        normal = np.zeros(3)
        # Using sign of the dominant axis
        normal[max_axis] = np.sign(p[max_axis])
        
        return normal

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Box mapping (Triplanar-ish)
        # Map UV based on the dominant face
        n = np.abs(local_normal)
        if n[0] > n[1] and n[0] > n[2]:
            return local_point[1:3] # YZ plane
        elif n[1] > n[0] and n[1] > n[2]:
            return local_point[::2] # XZ plane
        else:
            return local_point[0:2] # XY plane

    def convex_hull(self) -> List[np.ndarray]:
        # Returns the 8 corners of the box (Local Space)
        corners = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    corners.append(np.array([x, y, z]) * self.half_size)
        return corners

    @property
    def volume(self) -> float:
        # L * W * H
        size = self.half_size * 2
        return size[0] * size[1] * size[2]

    @property
    def surface_area(self) -> float:
        # 2(lw + lh + wh)
        s = self.half_size * 2
        return 2 * (s[0]*s[1] + s[1]*s[2] + s[2]*s[0])

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
        return float(np.linalg.norm(point - closest) - self.radius)

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
        return float(pi * self.radius ** 2 * cylinder_height + (4/3) * pi * self.radius ** 3)

    @property
    def surface_area(self) -> float:
        from math import pi
        cylinder_height = np.linalg.norm(self.point2 - self.point1)
        return float(2 * pi * self.radius * cylinder_height + 4 * pi * self.radius ** 2)

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
    transform: Optional['transform'] = None
    material: Optional["PBRMaterial"] = None
    name: str = "VObject"

    children: List["VObject"] = field(default_factory=list)
    parent: Optional["VObject"] = field(default=None, repr=False) # Exclude from repr to avoid recursion

    def __post_init__(self):
        if self.transform is None:
            # Assuming Transform is defined
            self.transform = Transform.identity()
        
        # If no material provided on the VObject itself, inherit from the shape
        if self.material is None and self.shape is not None and hasattr(self.shape, 'material'):
            self.material = getattr(self.shape, 'material', None)

    @property
    def world_transform(self) -> 'transform':
        """
        Calculates the absolute World Space transform by traversing up the hierarchy.
        Usage: Use this property in your Renderer/Intersection loop.
        """
        if self.parent:
            # Combine Parent(World) * Self(Local)
            # Assuming your Transform class supports composition via multiplication
            return self.parent.world_transform * self.transform
        return self.transform

    def add_child(self, child: "VObject"):
        """Properly links a child to this parent."""
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: "VObject"):
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def flatten(self) -> List["VObject"]:
        """
        Returns a flat list of this object and all its descendants.
        Useful for passing a simple list to the Renderer.
        """
        flat_list = [self]
        for child in self.children:
            flat_list.extend(child.flatten())
        return flat_list

    def __hash__(self):
        # Uses the object's unique memory address
        return id(self)

    def __eq__(self, other):
        # Checks if they are literally the exact same instance in memory
        return self is other

    def __repr__(self):
        shape_name = self.shape.name if self.shape else "None"
        return f"VObject(name='{self.name}', shape={shape_name}, material={self.material}, children={len(self.children)})"

class ShapeFactory(ABC):
    """Abstract factory for creating shapes."""
    @abstractmethod
    def create(self, **kwargs) -> Shape:
        raise NotImplementedError

class CircleFactory(ShapeFactory):
    def create(self, center: np.ndarray, radius: float, **kwargs) -> Circle: # type: ignore
        # Circle ctor expects (center, normal, radius); default normal is +Z
        return Circle(center, np.array([0.0, 0.0, 1.0]), radius, **kwargs)

class TriangleFactory(ShapeFactory):
    def create(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs) -> Triangle: # type: ignore
        return Triangle(v1, v2, v3, **kwargs)

class PolygonFactory(ShapeFactory):
    def create(self, vertices: List[np.ndarray], **kwargs) -> Polygon: # type: ignore
        return Polygon(vertices, **kwargs) # type: ignore

class SphereFactory(ShapeFactory):
    def create(self, center: np.ndarray, radius: float, **kwargs) -> Sphere: # type: ignore
        return Sphere(center, radius, **kwargs)

class CubeFactory(ShapeFactory):
    def create(self, center: np.ndarray, side_length: float, **kwargs) -> Cube: # type: ignore
        return Cube(center, side_length, **kwargs)

class PrismFactory(ShapeFactory):
    def create(self, base_polygon: Polygon, height: float, **kwargs) -> Prism: # type: ignore
        return Prism(base_polygon, height, **kwargs) # type: ignore

class PyramidFactory(ShapeFactory):
    def create(self, base_polygon: Polygon, height: float, **kwargs) -> Pyramid: # type: ignore
        return Pyramid(base_polygon, height, **kwargs) # type: ignore

class CapsuleFactory(ShapeFactory):
    def create(self, point1: np.ndarray, point2: np.ndarray, radius: float, **kwargs) -> Capsule: # type: ignore
        return Capsule(point1, point2, radius, **kwargs) # type: ignore

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

    def intersect(self, ray: TracingRay, max_t: float =  1e30, bias: float = 1e-9) -> float:
        """
        Slab Method for Ray/AABB intersection.
        Returns distance to entry, or infinity if miss.
        """
        # We use the inverse direction to replace division with multiplication
        # This handles division by zero gracefully (results in +/- inf)
        inv_dir = 1.0 / (ray.orientation + bias) 
        
        t0 = (self.min_point - ray.origin) * inv_dir
        t1 = (self.max_point - ray.origin) * inv_dir

        tmin = np.maximum(np.minimum(t0, t1), 0.0)
        tmax = np.minimum(np.maximum(t0, t1), max_t)

        # Find largest entry time and smallest exit time across all axes
        t_enter = np.max(tmin)
        t_exit = np.min(tmax)

        if t_exit >= t_enter:
            return t_enter
        
        return float('inf')

    @staticmethod
    def from_object(obj: VObject, padding: float = 1e-2) -> 'AABB':
        """
        Calculates the world-space AABB for a given object.
        """
        # Get the object's local bounds (e.g., Sphere is [-r, -r, -r] to [r, r, r])
        # This assumes your Shape classes have a 'get_bounds()' method.
        # Fallback: Approximate with a unit cube scaled by transform
        
        # 1. Get Transform Matrix
        obj_transform = cast(Transform, getattr(obj, 'transform', Transform.identity()))
        matrix = obj_transform.get_global_matrix()
        
        # 2. Define the 8 corners of a cube localy
        shape = getattr(obj, "shape", None)
        local_corners = None
        
        if shape is not None:
            # 1. Handle Cubes / Meshes (Anything with corners)
            if hasattr(shape, "convex_hull"):
                local_corners = np.array(shape.convex_hull())
            
            # 2. Handle Spheres (Look for radius)
            elif hasattr(shape, "radius"):
                # Create a box that fully encloses the sphere
                r = float(shape.radius)
                local_corners = np.array([
                    [-r, -r, -r], [r, -r, -r], [-r, r, -r], [r, r, -r],
                    [-r, -r, r],  [r, -r, r],  [-r, r, r],  [r, r, r]
                ])

        # C. Fallback: Unit Cube (-0.5 to 0.5)
        if local_corners is None:
            local_corners = np.array([
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], 
                [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],  [0.5, -0.5, 0.5],  
                [-0.5, 0.5, 0.5],  [0.5, 0.5, 0.5]
            ])

        # 2. Transform to World Space
        # Convert to homogeneous coordinates (N, 4)
        ones = np.ones((len(local_corners), 1))
        corners_4d = np.hstack([local_corners, ones]) 
        
        # Apply Matrix (Scale, Rotate, Translate)
        world_corners = (matrix @ corners_4d.T).T[:, :3]

        # 4. Find min/max of transformed corners
        min_p = np.min(world_corners, axis=0) - padding # Small padding
        max_p = np.max(world_corners, axis=0) + padding
        
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