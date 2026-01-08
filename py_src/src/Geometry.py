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
    Base for 2D shapes (Planes, Disks).
    Strictly defined on the Local XZ Plane (y=0) where possible (Y-Up).
    """
    @property
    def dimensions(self) -> int:
        return 2

    def volume(self) -> float:
        return 0.0
    
    # 2D Shapes are essentially 0-thickness in Y
    def unit_signed_distance(self, local_point: np.ndarray) -> float:
        # Default fallback: Treat as flat plane with thickness 0 along Y
        return abs(local_point[1])

class Circle(Shape2D):
    def __init__(self, **kwargs):
        """
        A Disk on the XZ plane (Y=0).
        """
        super().__init__(**kwargs)
        self.radius = 1

    def signed_distance(self, local_point: np.ndarray) -> float:
        """SDF for a Disk on the XZ plane."""
        # 2D distance from center (0,0) in X/Z
        # We index [0] (x) and [2] (z)
        flat_dist = np.linalg.norm([local_point[0], local_point[2]]) - self.radius
        
        # Sqrt( d_planar^2 + y^2 )
        return np.sqrt(max(flat_dist, 0.0)**2 + local_point[1]**2)

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # Intersect with Plane Y=0
        if abs(local_ray.orientation[1]) < 1e-6:
            return []
            
        t = -local_ray.origin[1] / local_ray.orientation[1]
        
        if t < 0: return []
        
        p = local_ray.point_at(t)
        
        # Check if point is within radius (using X and Z)
        if (p[0]**2 + p[2]**2) <= self.radius**2:
            return [p]
            
        return []

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        # Normal is always +Y in local space
        return np.array([0.0, 1.0, 0.0])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Planar mapping using X and Z
        u = (local_point[0] / self.radius) * 0.5 + 0.5
        v = (local_point[2] / self.radius) * 0.5 + 0.5
        return np.array([u, v])

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2

class Plane(Shape2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def signed_distance(self, local_point: np.ndarray) -> float:
        # Distance to Y=0 plane is just Y height
        return local_point[1]

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # Intersect Y component
        denom = local_ray.orientation[1]
        if abs(denom) < 1e-6:
            return []
            
        t = -local_ray.origin[1] / denom
        if t >= 0:
            return [local_ray.point_at(t)]
        return []

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Tiling UVs using X and Z
        return np.array([local_point[0], local_point[2]])

class ClippedPlane(Shape2D):
    def __init__(self, clip_polygon: List[np.ndarray], **kwargs):
        """
        clip_polygon: List of 2D points (x, z) defining the clip boundary 
                      on the local XZ plane.
        """
        super().__init__(**kwargs)
        if len(clip_polygon) < 3:
            raise ValueError("Clip polygon must have at least 3 vertices")
        
        # Ensure points are 2D arrays
        self.clip_polygon = [np.array(v[:2]) for v in clip_polygon]

    def signed_distance(self, local_point: np.ndarray) -> float:
        # Distance to the infinite plane Y=0
        return local_point[1]

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # 1. Intersect Infinite Plane Y=0
        denom = local_ray.orientation[1]
        if abs(denom) < 1e-6: return []
            
        t = -local_ray.origin[1] / denom
        if t < 0: return []
        
        hit_point = local_ray.point_at(t)
        
        # 2. Check Point-in-Polygon (Using X and Z coordinates)
        # pass X and Z to the 2D check
        if self._is_point_in_poly(np.array([hit_point[0], hit_point[2]])):
            return [hit_point]
            
        return []

    def _is_point_in_poly(self, p2d: np.ndarray) -> bool:
        # Ray casting algorithm for Point in Polygon
        inside = False
        n = len(self.clip_polygon)
        x, z = p2d # Interpreted as X and Z
        
        for i in range(n):
            x1, z1 = self.clip_polygon[i]
            x2, z2 = self.clip_polygon[(i + 1) % n]
            
            if ((z1 > z) != (z2 > z)) and \
               (x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-10) + x1):
                inside = not inside
                
        return inside
    
    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        return np.array([local_point[0], local_point[2]])

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
        raise NotImplementedError
    
class Triangle(Shape3D):
    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        
        # Vertices in Local Space
        self.v1 = np.asarray(v1, dtype=float)
        self.v2 = np.asarray(v2, dtype=float)
        self.v3 = np.asarray(v3, dtype=float)
        
        # Edges
        self.edge1 = self.v2 - self.v1
        self.edge2 = self.v3 - self.v1
        self.edge3 = self.v3 - self.v2 # Used for area calc logic
        
        # Normal
        self.normal = np.cross(self.edge1, self.edge2)
        norm_mag = np.linalg.norm(self.normal)
        if norm_mag > 1e-12:
            self.normal /= norm_mag
        else:
            self.normal = np.array([0.0, 1.0, 0.0]) # Degenerate triangle fallback

        # Cached vectors for SDF optimization
        self.ba = self.v2 - self.v1
        self.cb = self.v3 - self.v2
        self.ac = self.v1 - self.v3

    def signed_distance(self, p: np.ndarray) -> float:
        """
        Calculates distance to the face if projected inside, 
        or distance to the nearest edge if projected outside.
        """
        pa = p - self.v1
        pb = p - self.v2
        pc = p - self.v3
        
        # 1. Project point to plane to check if inside the triangle "cone"
        # Uses cross product to determine which side of the edge vector the point lies on
        nor = self.normal
        c1 = np.dot(np.cross(self.ba, nor), pa)
        c2 = np.dot(np.cross(self.cb, nor), pb)
        c3 = np.dot(np.cross(self.ac, nor), pc)

        if c1 <= 0.0 and c2 <= 0.0 and c3 <= 0.0:
            # Inside the triangle face: Distance is distance to plane
            return float(abs(np.dot(nor, pa)))
        
        # 2. Outside: Distance to closest edge segment
        d1 = np.dot(self.ba * np.clip(np.dot(self.ba, pa) / np.dot(self.ba, self.ba), 0.0, 1.0) - pa, 
                    self.ba * np.clip(np.dot(self.ba, pa) / np.dot(self.ba, self.ba), 0.0, 1.0) - pa)
        d2 = np.dot(self.cb * np.clip(np.dot(self.cb, pb) / np.dot(self.cb, self.cb), 0.0, 1.0) - pb, 
                    self.cb * np.clip(np.dot(self.cb, pb) / np.dot(self.cb, self.cb), 0.0, 1.0) - pb)
        d3 = np.dot(self.ac * np.clip(np.dot(self.ac, pc) / np.dot(self.ac, self.ac), 0.0, 1.0) - pc, 
                    self.ac * np.clip(np.dot(self.ac, pc) / np.dot(self.ac, self.ac), 0.0, 1.0) - pc)

        return float(np.sqrt(min(d1, min(d2, d3))))

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # Möller–Trumbore intersection algorithm
        h = np.cross(local_ray.orientation, self.edge2)
        a = np.dot(self.edge1, h)

        if abs(a) < 1e-8: return [] # Ray is parallel to triangle

        f = 1.0 / a
        s = local_ray.origin - self.v1
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0: return []

        q = np.cross(s, self.edge1)
        v = f * np.dot(local_ray.orientation, q)

        if v < 0.0 or u + v > 1.0: return []

        t = f * np.dot(self.edge2, q)

        if t > 1e-8:
            return [local_ray.point_at(t)]
        return []

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        return self.normal

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Barycentric UV mapping could go here, for now simpler planar projection
        return local_point[:2]

    # --- Shape3D Implementation ---

    @property
    def volume(self) -> float:
        # A 2D plane in 3D space has zero volume
        return 0.0

    @property
    def surface_area(self) -> float:
        # 0.5 * |AB x AC|
        return float(0.5 * np.linalg.norm(np.cross(self.edge1, self.edge2)))

    def convex_hull(self) -> List[np.ndarray]:
        # The convex hull of a triangle is just its 3 vertices
        return [self.v1, self.v2, self.v3]

class Polygon(Shape3D):
    def __init__(self, vertices: List[np.ndarray], **kwargs):
        super().__init__(**kwargs)
        self.vertices = [np.asarray(v, dtype=float) for v in vertices]
        if len(self.vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices")

        # Create sub-triangles (Fan Triangulation - assumes CONVEX polygon)
        # If your polygons are concave, you must use Ear Clipping here.
        self._triangles: List[Triangle] = []
        for i in range(1, len(self.vertices) - 1):
            self._triangles.append(
                Triangle(self.vertices[0], self.vertices[i], self.vertices[i + 1])
            )
        
        # Store normal from the first triangle (assuming planar polygon)
        self.normal = self._triangles[0].normal

    def signed_distance(self, local_point: np.ndarray) -> float:
        # The distance to the polygon is the minimum distance to any of its triangles
        min_d = float("inf")
        for tri in self._triangles:
            d = tri.signed_distance(local_point)
            if d < min_d:
                min_d = d
        return min_d

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        all_hits = []
        for tri in self._triangles:
            hits = tri.get_ray_intersections(local_ray)
            all_hits.extend(hits)
        
        # Sort by distance
        if all_hits:
            all_hits.sort(key=lambda p: np.linalg.norm(p - local_ray.origin))
        return all_hits

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        return self.normal
        
    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        return local_point[:2]

    # --- Shape3D Implementation ---

    @property
    def volume(self) -> float:
        return 0.0

    @property
    def surface_area(self) -> float:
        # Sum of all sub-triangle areas
        return sum(tri.surface_area for tri in self._triangles)

    def convex_hull(self) -> List[np.ndarray]:
        # For a convex polygon, the hull is simply the vertices.
        return self.vertices
    
class Sphere(Shape3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.radius = 1

    def signed_distance(self, local_point: np.ndarray) -> float:
        return float(np.linalg.norm(local_point) - self.radius)

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        oc = local_ray.origin
        a = np.dot(local_ray.orientation, local_ray.orientation)
        b = 2.0 * np.dot(oc, local_ray.orientation)
        c = np.dot(oc, oc) - self.radius**2
        
        discriminant = b*b - 4*a*c
        if discriminant < 0: return []
            
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        hits = []
        if t1 >= 0: hits.append(local_ray.point_at(t1))
        if t2 >= 0: hits.append(local_ray.point_at(t2))
        return hits

    def get_normal(self, local_point: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(local_point)
        if dist < 1e-6: return np.array([0, 1, 0])
        return local_point / dist

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Spherical UV mapping (Y-Up)
        # Y is Latitude (poles at +1, -1), X/Z is Longitude
        n = local_normal
        # arctan2(z, x) corresponds to angle on XZ plane
        u = 0.5 + np.arctan2(n[2], n[0]) / (2 * np.pi)
        # arcsin(y) corresponds to angle from equator up to pole
        v = 0.5 + np.arcsin(n[1]) / np.pi
        return np.array([u, v])

    def convex_hull(self, resolution: int = 12) -> List[np.ndarray]:
        points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(resolution):
            y = 1 - (i / float(resolution - 1)) * 2 # Y goes from 1 to -1
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
        if isinstance(size, (int, float)):
            self.half_size = np.array([size, size, size]) / 2.0
        else:
            self.half_size = np.asarray(size, dtype=float) / 2.0

    def signed_distance(self, local_point: np.ndarray) -> float:
        q = np.abs(local_point) - self.half_size
        return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        b_min = -self.half_size
        b_max = self.half_size
        
        inv_dir = np.zeros_like(local_ray.orientation)
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / local_ray.orientation
        inv_dir[np.isinf(inv_dir)] = 1e30 
        
        t1 = (b_min - local_ray.origin) * inv_dir
        t2 = (b_max - local_ray.origin) * inv_dir
        
        t_min = np.maximum(np.minimum(t1, t2), 0.0)
        t_max = np.minimum(np.maximum(t1, t2), 1e30)
        
        t_enter = np.max(t_min)
        t_exit = np.min(t_max)
        
        if t_exit < t_enter or t_exit < 0:
            return []
            
        hits = []
        if t_enter >= 0: hits.append(local_ray.point_at(t_enter))
        if t_exit > t_enter: hits.append(local_ray.point_at(t_exit))
            
        return hits

    def get_normal(self, local_point: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
        p = local_point / (self.half_size + epsilon)
        abs_p = np.abs(p)
        max_axis = np.argmax(abs_p)
        normal = np.zeros(3)
        normal[max_axis] = np.sign(p[max_axis])
        return normal

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Box mapping (Triplanar-ish) for Y-Up
        n = np.abs(local_normal)
        # Dominant Y (Top/Bottom) -> Map XZ
        if n[1] > n[0] and n[1] > n[2]:
             return np.array([local_point[0], local_point[2]])
        # Dominant X (Left/Right) -> Map ZY
        elif n[0] > n[1] and n[0] > n[2]:
            return np.array([local_point[2], local_point[1]])
        # Dominant Z (Front/Back) -> Map XY
        else:
            return np.array([local_point[0], local_point[1]])

    def convex_hull(self) -> List[np.ndarray]:
        corners = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    corners.append(np.array([x, y, z]) * self.half_size)
        return corners

    @property
    def volume(self) -> float:
        size = self.half_size * 2
        return size[0] * size[1] * size[2]

    @property
    def surface_area(self) -> float:
        s = self.half_size * 2
        return 2 * (s[0]*s[1] + s[1]*s[2] + s[2]*s[0])

class Pyramid(Shape3D):
    def __init__(self, base_polygon: Polygon, height: float, **kwargs):
        super().__init__(**kwargs)
        self.base_polygon = base_polygon
        if height <= 0:
            raise ValueError("Height must be > 0")
        self.height = float(height)

class Prism(Shape3D):
    def __init__(self, base_polygon: Polygon, height: float, **kwargs):
        super().__init__(**kwargs)
        self.base_polygon = base_polygon
        if height <= 0:
            raise ValueError("Height must be > 0")
        self.height = float(height)

class Cylinder(Shape3D):
    def __init__(self, radius: float = 1.0, height: float = 2.0, **kwargs):
        """
        Vertical Cylinder aligned along the Y-axis.
        Range: y is from -height/2 to +height/2.
        """
        super().__init__(**kwargs)
        self.radius = float(radius)
        self.height = float(height)
        self.half_height = height / 2.0

    def signed_distance(self, p: np.ndarray) -> float:
        # Distance to infinite cylinder on (x, z)
        d_axial = np.linalg.norm(p[[0, 2]]) - self.radius
        # Distance from vertical bounds
        d_vertical = abs(p[1]) - self.half_height
        
        # Exterior distance (Euclidean)
        outside = np.linalg.norm(np.maximum(np.array([d_axial, d_vertical]), 0.0))
        # Interior distance (Negative, closest edge)
        inside = min(max(d_axial, d_vertical), 0.0)
        
        return float(outside + inside)

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        hits = []
        ox, oy, oz = local_ray.origin
        dx, dy, dz = local_ray.orientation
        
        # 1. Intersect Infinite Cylinder (x^2 + z^2 = r^2)
        # quadratic: a*t^2 + b*t + c = 0
        a = dx*dx + dz*dz
        
        if a > 1e-6:
            b = 2.0 * (ox*dx + oz*dz)
            c = (ox*ox + oz*oz) - self.radius**2
            
            disc = b*b - 4*a*c
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / (2.0*a)
                t2 = (-b + sqrt_disc) / (2.0*a)
                
                for t in [t1, t2]:
                    if t >= 0:
                        p = local_ray.point_at(t)
                        # Check height bounds
                        if abs(p[1]) <= self.half_height:
                            hits.append(p)

        # 2. Intersect Caps (Planes y = +/- half_height)
        if abs(dy) > 1e-6:
            for y_cap in [-self.half_height, self.half_height]:
                t = (y_cap - oy) / dy
                if t >= 0:
                    p = local_ray.point_at(t)
                    # Check radius bounds (x^2 + z^2 <= r^2)
                    if (p[0]**2 + p[2]**2) <= self.radius**2:
                        hits.append(p)
        
        # Sort by distance from origin
        hits.sort(key=lambda p: np.linalg.norm(p - local_ray.origin))
        return hits

    def get_normal(self, p: np.ndarray) -> np.ndarray:
        # Determine if we are on the caps or the side
        # (This logic favors the side if exactly on the edge)
        dist_axis = np.linalg.norm(p[[0, 2]])
        dist_cap_top = abs(p[1] - self.half_height)
        dist_cap_btm = abs(p[1] + self.half_height)
        
        # Epsilon for edge cases
        eps = 1e-4
        
        if dist_axis < self.radius - eps:
            # On caps
            return np.array([0.0, 1.0, 0.0]) if p[1] > 0 else np.array([0.0, -1.0, 0.0])
        else:
            # On side (Normal is in XZ plane)
            n = np.array([p[0], 0.0, p[2]])
            norm = np.linalg.norm(n)
            return n / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

    def get_uv(self, p: np.ndarray, n: np.ndarray) -> np.ndarray:
        # Cylindrical mapping
        # If normal is roughly vertical, use Planar mapping
        if abs(n[1]) > 0.9:
            # Caps: Map XZ to UV
            return (p[[0, 2]] / self.radius) * 0.5 + 0.5
        else:
            # Side: Unrap
            u = (np.arctan2(p[0], p[2]) / (2 * np.pi)) + 0.5
            v = (p[1] / self.height) + 0.5
            return np.array([u, v])

    @property
    def volume(self) -> float:
        return np.pi * (self.radius**2) * self.height

    @property
    def surface_area(self) -> float:
        # 2*pi*r*h + 2*pi*r^2
        return (2 * np.pi * self.radius * self.height) + (2 * np.pi * self.radius**2)

    def convex_hull(self) -> List[np.ndarray]:
        # Approximation: Two circles of points
        points = []
        segments = 12
        for y in [-self.half_height, self.half_height]:
            for i in range(segments):
                theta = (i / segments) * 2 * np.pi
                x = np.cos(theta) * self.radius
                z = np.sin(theta) * self.radius
                points.append(np.array([x, y, z]))
        return points


class Capsule(Shape3D):
    def __init__(self, radius: float = 0.5, height: float = 2.0, **kwargs):
        """
        Capsule aligned along Y-axis. 
        Total height is (cylinder_height + 2 * radius).
        Defined by segment A=(0, -h/2, 0) to B=(0, h/2, 0).
        """
        super().__init__(**kwargs)
        self.radius = float(radius)
        # The 'height' parameter here refers to the cylindrical segment length
        self.segment_height = float(height)
        self.half_seg = self.segment_height / 2.0

    def signed_distance(self, p: np.ndarray) -> float:
        # Vector from p to the axis line
        # Since axis is Y, closest point on infinite axis is (0, p.y, 0)
        # Clamped to segment:
        y_clamped = np.clip(p[1], -self.half_seg, self.half_seg)
        closest_on_segment = np.array([0.0, y_clamped, 0.0])
        
        dist = np.linalg.norm(p - closest_on_segment)
        return float(dist - self.radius)

    def get_ray_intersections(self, local_ray: "Ray") -> List[np.ndarray]:
        # Analytic intersection is union of Cylinder + 2 Spheres
        hits = []
        
        # 1. Cylinder part (Infinite XZ intersection clipped by Y)
        ox, oy, oz = local_ray.origin
        dx, dy, dz = local_ray.orientation
        
        a = dx*dx + dz*dz
        if a > 1e-6:
            b = 2.0 * (ox*dx + oz*dz)
            c = (ox*ox + oz*oz) - self.radius**2
            disc = b*b - 4*a*c
            if disc >= 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / (2.0*a)
                t2 = (-b + sqrt_disc) / (2.0*a)
                for t in [t1, t2]:
                    if t >= 0:
                        p = local_ray.point_at(t)
                        # strictly inside the segment range
                        if abs(p[1]) <= self.half_seg:
                            hits.append(p)
                            
        # 2. Sphere Caps (Top and Bottom)
        # Sphere centers at (0, -h/2, 0) and (0, h/2, 0)
        for y_center in [-self.half_seg, self.half_seg]:
            center = np.array([0.0, y_center, 0.0])
            oc = local_ray.origin - center
            
            # Sphere quadratic
            sa = np.dot(local_ray.orientation, local_ray.orientation)
            sb = 2.0 * np.dot(oc, local_ray.orientation)
            sc = np.dot(oc, oc) - self.radius**2
            
            s_disc = sb*sb - 4*sa*sc
            if s_disc >= 0:
                s_sqrt = np.sqrt(s_disc)
                st1 = (-sb - s_sqrt) / (2.0*sa)
                st2 = (-sb + s_sqrt) / (2.0*sa)
                
                for t in [st1, st2]:
                    if t >= 0:
                        p = local_ray.point_at(t)
                        # Check if this hit is on the "outer" hemisphere part
                        # i.e., abs(y) > half_seg
                        if abs(p[1]) >= self.half_seg:
                            hits.append(p)

        hits.sort(key=lambda p: np.linalg.norm(p - local_ray.origin))
        return hits

    def get_normal(self, p: np.ndarray) -> np.ndarray:
        # Gradient of SDF
        y_clamped = np.clip(p[1], -self.half_seg, self.half_seg)
        closest_on_segment = np.array([0.0, y_clamped, 0.0])
        normal = p - closest_on_segment
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else np.array([0, 0, 1])

    def get_uv(self, p: np.ndarray, n: np.ndarray) -> np.ndarray:
        # Spherical coordinates, but stretched vertically for the cylinder part
        u = 0.5 + np.arctan2(n[2], n[0]) / (2 * np.pi)
        v = (p[1] + self.half_seg + self.radius) / (self.segment_height + 2*self.radius)
        return np.array([u, v])

    @property
    def volume(self) -> float:
        # Cylinder vol + Sphere vol
        cyl_vol = np.pi * (self.radius**2) * self.segment_height
        sphere_vol = (4/3) * np.pi * (self.radius**3)
        return cyl_vol + sphere_vol

    @property
    def surface_area(self) -> float:
        # Cylinder area + Sphere area
        cyl_area = 2 * np.pi * self.radius * self.segment_height
        sphere_area = 4 * np.pi * (self.radius**2)
        return cyl_area + sphere_area
        
    def convex_hull(self) -> List[np.ndarray]:
        # Similar to Cylinder but with points at the pole caps
        points = Cylinder(self.radius, self.segment_height).convex_hull()
        points.append(np.array([0, self.half_seg + self.radius, 0]))
        points.append(np.array([0, -self.half_seg - self.radius, 0]))
        return points

# VObject & Factories
@dataclass
class VObject:
    """
    Visual object combining shape, transform, and material.
    Acts as a Node in the Scene Graph.
    """
    transform: 'Transform' = field(default_factory=lambda: Transform.identity())
    shape: Optional['Shape'] = None # Optional so we can have empty "Group" nodes
    material: Optional['PBRMaterial'] = None
    name: str = "VObject"

    bounds: Optional['AABB'] = None
    children: List['VObject'] = field(default_factory=list)
    parent: Optional['VObject'] = field(default=None, repr=False) # Exclude from repr to avoid recursion

    @property
    def world_transform(self) -> 'Transform':
        """
        Calculates the absolute World Space transform by traversing up the hierarchy.
        Usage: Use this property in your Renderer/Intersection loop.
        """
        if self.parent:
            # Combine Parent(World) * Self(Local)
            # Assuming your Transform class supports composition via multiplication
            t_m = self.parent.world_transform * self.transform
            if isinstance(t_m, 'Transform'):
                return t_m
            return Transform(*Transform.decompose_matrix(t_m))
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
        Returns a flat list of this object and all descendants using an iterative stack.
        Safe for very deep hierarchies.
        """
        result = []
        # Start with the current object
        stack = [self]
        
        while stack:
            current = stack.pop()
            result.append(current)
            
            # Add children to the stack. 
            # We reverse them so they pop in the original order (optional, aesthetic).
            for child in reversed(current.children):
                stack.append(child)
                
        return result

    def flatten_with_transforms(self, parent_matrix: Optional[np.ndarray]) -> List[Tuple["VObject", np.ndarray]]:
        """
        Returns a list of (object, absolute_world_matrix).
        Calculates world matrices on the way down to avoid re-traversing up later.
        """
        # 1. Calculate World Matrix for self
        # If no parent matrix provided, use Identity (or self's local matrix if root)
        local_mat = self.transform.get_global_matrix()
        
        if parent_matrix is not None:
            world_mat = parent_matrix @ local_mat
        else:
            world_mat = local_mat

        # 2. Add self to list
        flat_list: List[Tuple["VObject", np.ndarray]] = [(self, world_mat)]

        # 3. Pass calculated world_mat down to children
        for child in self.children:
            flat_list.extend(child.flatten_with_transforms(world_mat))
        
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
    def create(self, radius: float, **kwargs) -> Circle: # type: ignore
        # Circle ctor expects (center, normal, radius); default normal is +Z
        return Circle(radius, **kwargs)

class TriangleFactory(ShapeFactory):
    def create(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, **kwargs) -> Triangle: # type: ignore
        return Triangle(v1, v2, v3, **kwargs)

class PolygonFactory(ShapeFactory):
    def create(self, vertices: List[np.ndarray], **kwargs) -> Polygon: # type: ignore
        return Polygon(vertices, **kwargs) # type: ignore

class SphereFactory(ShapeFactory):
    def create(self, radius: float, **kwargs) -> Sphere: # type: ignore
        return Sphere(radius, **kwargs)

class CubeFactory(ShapeFactory):
    def create(self, size: float | np.ndarray, **kwargs) -> Cube: # type: ignore
        return Cube(size, **kwargs)

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