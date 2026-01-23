from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, List, Tuple

from src.Data.Ray import Ray

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
    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
        """
        Analytical intersection (Ray-Sphere, etc).
        Returns list of hit points in LOCAL SPACE.
        """
        raise NotImplementedError

    # --- 4. Volumetric Helpers (Local Space) ---

    def get_entry_exit_points(
        self,
        local_ray: Ray, 
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

    def get_normal(self, *args, **kwargs) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    @property
    def dimensions(self) -> int:
        return 2

    @property
    @abstractmethod
    def perimeter(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError
    
    @property
    def volume(self) -> float:
        return 0.0
    
    # 2D Shapes are essentially 0-thickness in Y
    def unit_signed_distance(self, local_point: np.ndarray) -> float:
        # Default fallback: Treat as flat plane with thickness 0 along Y
        return abs(local_point[1])

    def get_uv(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.array([local_point[0], local_point[2]])

class Circle(Shape2D):
    def __init__(self, **kwargs):
        """
        A Disk on the XZ plane (Y=0).
        """
        self.radius = 1
        super().__init__(**kwargs)

    def signed_distance(self, local_point: np.ndarray) -> float:
        """SDF for a Disk on the XZ plane."""
        # 2D distance from center (0,0) in X/Z
        # We index [0] (x) and [2] (z)
        flat_dist = np.linalg.norm([local_point[0], local_point[2]]) - self.radius
        
        # Sqrt( d_planar^2 + y^2 )
        return np.sqrt(max(flat_dist, 0.0)**2 + local_point[1]**2)

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
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

    def get_uv(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Planar mapping using X and Z
        u = (local_point[0] / self.radius) * 0.5 + 0.5
        v = (local_point[2] / self.radius) * 0.5 + 0.5
        return np.array([u, v])

    @property
    def area(self) -> float:
        return np.pi * self.radius ** 2

    @property
    def perimeter(self) -> float:
        return 2 * np.pi * self.radius

class Square(Shape2D):
    def __init__(self, **kwargs):
        """
        A Square on the XZ plane (Y=0).
        """
        self.size = 1.0
        super().__init__(**kwargs)

    def signed_distance(self, local_point: np.ndarray) -> float:
        # 2D distance from center in X/Z
        dx = abs(local_point[0]) - (self.size / 2)
        dz = abs(local_point[2]) - (self.size / 2)
        
        # Outside distance
        outside_dist = np.sqrt(max(dx, 0.0)**2 + max(dz, 0.0)**2)
        
        # Inside distance
        inside_dist = min(max(dx, dz), 0.0)
        
        return outside_dist + inside_dist + abs(local_point[1])

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
        # Intersect with Plane Y=0
        if abs(local_ray.orientation[1]) < 1e-6:
            return []
            
        t = -local_ray.origin[1] / local_ray.orientation[1]
        
        if t < 0: return []
        
        p = local_ray.point_at(t)
        
        # Check if point is within square bounds
        half_size = self.size / 2
        if abs(p[0]) <= half_size and abs(p[2]) <= half_size:
            return [p]
            
        return []

    def get_uv(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        u = (local_point[0] / self.size) + 0.5
        v = (local_point[2] / self.size) + 0.5
        return np.array([u, v])

    @property
    def area(self) -> float:
        return self.size ** 2

    @property
    def perimeter(self) -> float:
        return 4 * self.size

class Plane(Shape2D):
    """
    Infinite Plane on the XZ plane (Y=0).
    """
    def signed_distance(self, local_point: np.ndarray) -> float:
        # Distance to Y=0 plane is just Y height
        return local_point[1]

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
        # Intersect Y component
        denom = local_ray.orientation[1]
        if abs(denom) < 1e-6:
            return []
            
        t = -local_ray.origin[1] / denom
        if t >= 0:
            return [local_ray.point_at(t)]
        return []

    @property
    def perimeter(self) -> float:
        return float('inf')

    @property
    def area(self) -> float:
        return float('inf')

class ClippedPlane(Shape2D):
    """
    Plane clipped to a polygonal boundary on the XZ plane.
    """
    def __init__(self, clip_polygon: List[np.ndarray], **kwargs):
        """
        :param clip_polygon: List of 2D points (x, z) defining the clip boundary 
        """
        super().__init__(**kwargs)
        if len(clip_polygon) < 3:
            raise ValueError("Clip polygon must have at least 3 vertices")
        
        # Ensure points are 2D arrays
        self.clip_polygon = [np.array(v[:2]) for v in clip_polygon]

    def signed_distance(self, local_point: np.ndarray) -> float:
        # Distance to the infinite plane Y=0
        return local_point[1]

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
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

    @property
    def perimeter(self) -> float:
        if not self.clip_polygon:
            return 0.0
        perimeter = 0.0
        n = len(self.clip_polygon)
        for i in range(n):
            p1 = self.clip_polygon[i]
            p2 = self.clip_polygon[(i + 1) % n]
            perimeter += np.linalg.norm(p2 - p1)
        return float(perimeter)

    @property
    def area(self) -> float:
        if not self.clip_polygon or len(self.clip_polygon) < 3:
            return 0.0
        # Shoelace formula for polygon area
        area = 0.0
        n = len(self.clip_polygon)
        for i in range(n):
            x1, z1 = self.clip_polygon[i]
            x2, z2 = self.clip_polygon[(i + 1) % n]
            area += x1 * z2 - x2 * z1
        return abs(area) / 2.0

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

    @abstractmethod
    def convex_hull(self) -> List[np.ndarray]:
        raise NotImplementedError
    
class Triangle(Shape3D):
    """
    A 3D Triangle defined by 3 vertices in Local Space.
    """
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

    def signed_distance(self, local_point: np.ndarray) -> float:
        """
        Calculates distance to the face if projected inside, 
        or distance to the nearest edge if projected outside.
        """
        pa = local_point - self.v1
        pb = local_point - self.v2
        pc = local_point - self.v3
        
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

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
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

    def get_normal(self, *args, **kwargs) -> np.ndarray:
        return self.normal

    def get_uv(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
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
    """
    A 3D Polygon defined by a list of vertices in Local Space.
    """
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

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
        all_hits = []
        for tri in self._triangles:
            hits = tri.get_ray_intersections(local_ray)
            all_hits.extend(hits)
        
        # Sort by distance
        if all_hits:
            all_hits.sort(key=lambda p: np.linalg.norm(p - local_ray.origin))
        return all_hits

    def get_normal(self, *args, **kwargs) -> np.ndarray:
        return self.normal

    def get_uv(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
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

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
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

    def get_normal(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        dist = np.linalg.norm(local_point)
        if dist < 1e-6: return np.array([0, 1, 0])
        return local_point / dist

    def get_uv(self, local_normal: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Spherical UV mapping (Y-Up)
        # Y is Latitude (poles at +1, -1), X/Z is Longitude
        n = local_normal
        # arctan2(z, x) corresponds to angle on XZ plane
        u = 0.5 + np.arctan2(n[2], n[0]) / (2 * np.pi)
        # arcsin(y) corresponds to angle from equator up to pole
        v = 0.5 + np.arcsin(n[1]) / np.pi
        return np.array([u, v])

    def convex_hull(self, resolution: int = 128) -> List[np.ndarray]:
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

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
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
    def __init__(self, base: Shape2D, height: float, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        if height <= 0:
            raise ValueError("Height must be > 0")
        self.height = float(height)

    @property
    def volume(self) -> float:
        return (1/3) * self.base.area * self.height

    @property
    def surface_area(self) -> float:
        # Approximation: base + lateral assuming slant height = height / 2
        lateral = self.base.perimeter * (self.height / 2) / 2
        return self.base.area + lateral

    def signed_distance(self, local_point: np.ndarray) -> float:
        # Approximation for pyramid SDF
        base_dist = self.base.signed_distance(np.array([local_point[0], 0, local_point[2]]))
        y_from_base = (local_point[1] + self.height/2) / self.height
        scale = max(0, min(1, y_from_base))
        scaled_dist = base_dist * (1 - scale)
        y_dist = abs(local_point[1]) - self.height / 2
        return max(scaled_dist, y_dist)

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
        # Analytical intersection is complex for general base, return empty
        return []

    def get_normal(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Approximation
        return np.array([0.0, 1.0, 0.0])

    def get_uv(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        return local_point[:2]

    def convex_hull(self) -> List[np.ndarray]:
        # Approximation
        return []

class Prism(Shape3D):
    def __init__(self, base: Shape2D, height: float, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        if height <= 0:
            raise ValueError("Height must be > 0")
        self.height = float(height)

    @property
    def volume(self) -> float:
        return self.base.area * self.height

    @property
    def surface_area(self) -> float:
        lateral = self.base.perimeter * self.height
        return 2 * self.base.area + lateral

    def signed_distance(self, local_point: np.ndarray) -> float:
        base_dist = self.base.signed_distance(np.array([local_point[0], 0, local_point[2]]))
        y_dist = abs(local_point[1]) - self.height / 2
        return max(base_dist, y_dist)

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
        # Analytical intersection is complex for general base, return empty
        return []

    def get_normal(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Approximation
        return np.array([0.0, 1.0, 0.0])

    def get_uv(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        return local_point[:2]

    def convex_hull(self) -> List[np.ndarray]:
        # Approximation
        return []

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

    def signed_distance(self, local_point: np.ndarray) -> float:
        # Distance to infinite cylinder on (x, z)
        d_axial = np.linalg.norm(local_point[[0, 2]]) - self.radius
        # Distance from vertical bounds
        d_vertical = abs(local_point[1]) - self.half_height
        
        # Exterior distance (Euclidean)
        outside = np.linalg.norm(np.maximum(np.array([d_axial, d_vertical]), 0.0))
        # Interior distance (Negative, closest edge)
        inside = min(max(d_axial, d_vertical), 0.0)
        
        return float(outside + inside)

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
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

    def get_normal(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Determine if we are on the caps or the side
        # (This logic favors the side if exactly on the edge)
        dist_axis = np.linalg.norm(local_point[[0, 2]])
        dist_cap_top = abs(local_point[1] - self.half_height)
        dist_cap_btm = abs(local_point[1] + self.half_height)
        
        # Epsilon for edge cases
        eps = 1e-4
        
        if dist_axis < self.radius - eps:
            # On caps
            return np.array([0.0, 1.0, 0.0]) if local_point[1] > 0 else np.array([0.0, -1.0, 0.0])
        else:
            # On side (Normal is in XZ plane)
            n = np.array([local_point[0], 0.0, local_point[2]])
            norm = np.linalg.norm(n)
            return n / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Cylindrical mapping
        # If normal is roughly vertical, use Planar mapping
        if abs(local_normal[1]) > 0.9:
            # Caps: Map XZ to UV
            return (local_point[[0, 2]] / self.radius) * 0.5 + 0.5
        else:
            # Side: Unrap
            u = (np.arctan2(local_point[0], local_point[2]) / (2 * np.pi)) + 0.5
            v = (local_point[1] / self.height) + 0.5
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

    def signed_distance(self, local_point: np.ndarray) -> float:
        # Vector from local_point to the axis line
        # Since axis is Y, closest point on infinite axis is (0, local_point.y, 0)
        # Clamped to segment:
        y_clamped = np.clip(local_point[1], -self.half_seg, self.half_seg)
        closest_on_segment = np.array([0.0, y_clamped, 0.0])
        
        dist = np.linalg.norm(local_point - closest_on_segment)
        return float(dist - self.radius)

    def get_ray_intersections(self, local_ray: Ray) -> List[np.ndarray]:
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

    def get_normal(self, local_point: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Gradient of SDF
        y_clamped = np.clip(local_point[1], -self.half_seg, self.half_seg)
        closest_on_segment = np.array([0.0, y_clamped, 0.0])
        normal = local_point - closest_on_segment
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else np.array([0, 0, 1])

    def get_uv(self, local_point: np.ndarray, local_normal: np.ndarray) -> np.ndarray:
        # Spherical coordinates, but stretched vertically for the cylinder part
        u = 0.5 + np.arctan2(local_normal[2], local_normal[0]) / (2 * np.pi)
        v = (local_point[1] + self.half_seg + self.radius) / (self.segment_height + 2*self.radius)
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
