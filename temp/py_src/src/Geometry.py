import numpy as np 

from PrimaryStructures import Ray, Transform

"""

"""

class Shape:
    ts: np.ndarray = np.zeros(3)  # translation vectror
    rs: np.ndarray = np.zeros(3)  # rotation vector (Euler angles)
    sf: float = 1.0  # scale factor

    def __init__(self, **kwargs):
        self.name: str = kwargs.get("name", "Default Name")
        self.origin: np.ndarray = kwargs.get("origin", np.zeros(3))
        

        for key, value in kwargs.items():
            setattr(self, key, value)

    def CheckPointOnEdge(self, point: np.ndarray, epsilon: float) -> bool:
        raise NotImplementedError("CheckPointOnEdge method must be implemented in subclasses")
    
    def CheckPointInside(self, point: np.ndarray, epsilon: float) -> bool:
        raise NotImplementedError("CheckPointInside method must be implemented in subclasses")
    
    def GetDistance(self, point: np.ndarray) -> float:
        raise NotImplementedError("GetDistance method must be implemented in subclasses")

    def GetClosestPoint(self, point: np.ndarray) -> float:
        raise NotImplementedError("GetClosestPoint method must be implemented in subclasses")

    def CheckRayIntersection(self, ray: Ray) -> bool:
        raise NotImplementedError("CheckRayIntersection method must be implemented in subclasses")
    
    def GetRayIntersections(self, ray: Ray) -> list[np.ndarray]:
        raise NotImplementedError("GetRayIntersections method must be implemented in subclasses")

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GetNormal method must be implemented in subclasses")

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GetTangent method must be implemented in subclasses")
    
    @property
    def normals(self) -> list[np.ndarray]:
        raise NotImplementedError("Normals property must be implemented in subclasses")
    
    @property
    def tangents(self) -> list[np.ndarray]:
        raise NotImplementedError("Tangents property must be implemented in subclasses")

    @property
    def area(self) -> float:
        raise NotImplementedError("Area property must be implemented in subclasses")

    @property
    def perimeter(self) -> float:
        raise NotImplementedError("Perimeter property must be implemented in subclasses")
    
    @property
    def dimensions(self) -> int:
        raise NotImplementedError("Dimensions property must be implemented in subclasses")
    
    @property
    def volume(self) -> float:
        raise NotImplementedError("Volume property must be implemented in subclasses")
    
    def Translate(self, translation: np.ndarray) -> None:
        self.origin += translation

    def Scale(self, scale_factor: float) -> None:
        raise NotImplementedError("Scale method must be implemented in subclasses")

    def Rotate(self, rotation_matrix: np.ndarray) -> None:
        raise NotImplementedError("Rotate method must be implemented in subclasses")

    def ResetTransform(self) -> None:
        raise NotImplementedError("ResetTransform method must be implemented in subclasses")
    
    def ApplyTransform(self, transform: Transform) -> None:
        raise NotImplementedError("ApplyTransform method must be implemented in subclasses")

    def __repr__(self):
        return f"Shape()"

### TODO: Update the bellow classes to use all the new methods

class Circle(Shape):
    def __init__(self, center: np.ndarray, radius: float, name: str = "Circle"):
        super().__init__(name=name)
        self.center = center
        if radius <= 0:
            raise AttributeError("The radius of the Circle must be greater than 0")
        self.radius = radius

        self.origin = center
    
    def CheckPointInside(self, point: np.ndarray, epsilon: float = 0) -> bool:
        return self.radius - epsilon < np.linalg.norm(point - self.center) <= self.radius + epsilon
    
    # Using the quadratic equation to find intersection with circles
    # P(t) = dt + s
    # ||P(t)|| = r
    # (d * t + s)^2 = r^2
    
    # d * dt^2 + 2d * st + s^2 - r^2 = 0
    #
    # a = d.Dot(d)
    # b = 2 * d.Dot(s)
    # c = s.Dot(s) - r^2
    # at^2 + bt + c = 0
    
    # t = (-b ± sqrt(b^2 - 4ac)) / 2a
    # b/2 = d.Dot(s) 
    # t = (-2b ± sqrt(4b^2 - 4ac)) / 2a
    
    # Discriminant = b^2 - ac
    # b^2-ac < 0 ==> no intersection
    # b^2-ac = 0 ==> one intersection
    # b^2-ac > 0 ==> two intersections
    
    # the clossest intersection is the one with the smallest t
    # t = -b - sqrt(b^2 - ac) / a
        
    def CheckRayIntersection(self, ray: Ray) -> bool:
        d = ray.orientation 
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        if discriminant < 0:
            return False
        else:
            return True

    def GetRayIntersections(self, ray: Ray) -> list[np.ndarray]:
        if not self.CheckRayIntersection(ray):
            return []
        
        d = ray.orientation 
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant == 0:
            t = -b / (2 * a)
            return np.array([ray.point_at(t)]) if t >= 0 else []
        else:
            sqrt_disc = discriminant ** 0.5
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            # For rays starting inside, t=0 is a valid intersection (origin on circle)
            ts = [t for t in [t1, t2] if t >= 0 or np.isclose(t, 0)]
            return np.array([ray.point_at(t) for t in sorted(ts)])

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        if not self.CheckPoint(point, 0.01):
            raise ValueError("Point is not on the circle")
        vec = point - self.center
        return vec / np.linalg.norm(vec)

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        if not self.CheckPoint(point, 0.01):
            raise ValueError("Point is not on the circle")
        normal = self.GetNormal(point)
        # For 2D circle in XY plane, tangent is perpendicular to normal
        # If 3D, use cross product with Z axis if normal is in XY
        if normal.shape[0] == 2:
            return np.array([-normal[1], normal[0]])
        elif normal.shape[0] == 3:
            # Tangent in XY plane: cross with Z axis
            z_axis = np.array([0, 0, 1])
            tangent = np.cross(z_axis, normal)
            return tangent / np.linalg.norm(tangent)
        else:
            raise ValueError("Unsupported dimension for tangent computation")
        
    def CheckPointOnEdge(self, point: np.ndarray, epsilon: float) -> bool:
        dist = np.linalg.norm(point - self.center)
        return abs(dist - self.radius) <= epsilon
    
    def GetDistance(self, point: np.ndarray) -> float:
        return abs(np.linalg.norm(point - self.center) - self.radius)

    def GetClosestPoint(self, point: np.ndarray) -> float:
        direction = point - self.center
        direction_normalized = direction / np.linalg.norm(direction)
        return self.center + direction_normalized * self.radius

    @property
    def normals(self, resolution:int = 100) -> list[np.ndarray]:
        return [np.array([np.cos(theta), np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, num=resolution, endpoint=False)]
    
    @property
    def tangents(self, resolution:int = 100) -> list[np.ndarray]:
        return [np.array([-np.sin(theta), np.cos(theta)]) for theta in np.linspace(0, 2 * np.pi, num=resolution, endpoint=False)]

    @property
    def area(self) -> float:
        from math import pi
        return pi * self.radius ** 2

    @property
    def perimeter(self) -> float:
        from math import pi
        return 2 * pi * self.radius

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    @property
    def circumference(self) -> float:
        return self.Perimeter()
    
    @property
    def dimensions(self) -> int:
        return 2
    
    @property
    def volume(self) -> float:
        return 0.0
    
    def Scale(self, scale_factor: float) -> None:
        if scale_factor <= 0:
            raise ValueError("Scale factor must be greater than 0")
        self.sf *= scale_factor

        self.radius *= scale_factor

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"


class Triangle(Shape):
    def __init__(self, vertex1: np.ndarray, vertex2: np.ndarray, vertex3: np.ndarray, name: str = "Triangle"):
        super().__init__(name=name)
        self.vertex1 = np.asarray(vertex1, dtype=float)
        self.vertex2 = np.asarray(vertex2, dtype=float)
        self.vertex3 = np.asarray(vertex3, dtype=float)

        self.origin = (vertex1 + vertex2 + vertex3) / 3

        self.CheckDegeneracy()

    def CheckDegeneracy(self, epsilon: float = 1e-6) -> bool:
        """Check if any edge of the triangle is degenerate or area is near zero (collinear)."""
        edge1 = np.linalg.norm(self.vertex2 - self.vertex1)
        edge2 = np.linalg.norm(self.vertex3 - self.vertex2)
        edge3 = np.linalg.norm(self.vertex1 - self.vertex3)
        # If any edge is degenerate (zero length), triangle is degenerate
        if edge1 < epsilon or edge2 < epsilon or edge3 < epsilon:
            raise AttributeError("The triangle's verticies create edges that have near-zero lengths. The triangle is degenrate")
        # If area is near zero, triangle is degenerate (collinear)
        area = self.area
        if area < epsilon:
            raise AttributeError("The triangle has a near-zero Area. The triangle is degenrate")
    
    # Using a geometric solution to find intersection with triangles
    # P(t) = O + Rt
    # D = -(Ax )

    def CheCheckPointInsideck(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        # Barycentric technique
        v0 = self.vertex2 - self.vertex1
        v1 = self.vertex3 - self.vertex1
        v2 = point - self.vertex1

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= -epsilon) and (v >= -epsilon) and (u + v <= 1 + epsilon)

    def CheckRayIntersection(self, ray: Ray, epsilon: float = 1e-6) -> bool:
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex1
        h = np.cross(ray.orientation , edge2)
        a = np.dot(edge1, h)
        
        if -epsilon < a < epsilon:
            return False  # Ray is parallel to triangle
        
        f = 1.0 / a
        s = ray.origin - self.vertex1
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return False
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray.orientation , q)
        
        if v < 0.0 or u + v > 1.0:
            return False
        
        t = f * np.dot(edge2, q)
        
        if t > epsilon:  # Intersection with the triangle
            return True
        else:  # Line intersection but not a ray intersection
            return False
        
    def GetRayIntersections(self, ray: Ray, epsilon: float = 1e-6) -> np.ndarray | None:
        if not self.CheckIntersection(ray, epsilon):
            return None
        
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex1
        h = np.cross(ray.orientation , edge2)
        a = np.dot(edge1, h)
        
        f = 1.0 / a
        s = ray.origin - self.vertex1
        u = f * np.dot(s, h)
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray.orientation , q)
        
        t = f * np.dot(edge2, q)
        
        if t > epsilon:
            return ray.point_at(t)
        else:
            return None

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        if not self.CheckPoint(point):
            raise ValueError("Point is not on the triangle")
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex1
        return np.cross(edge1, edge2) / np.linalg.norm(np.cross(edge1, edge2))

    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        if not self.CheckPoint(point):
            raise ValueError("Point is not on the triangle")
        edge1 = self.vertex2 - self.vertex1
        return edge1 / np.linalg.norm(edge1)
    
    def CheckPointOnEdge(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        def point_on_segment(p, v1, v2, eps):
            d1 = np.linalg.norm(p - v1)
            d2 = np.linalg.norm(p - v2)
            segment_length = np.linalg.norm(v2 - v1)
            return abs((d1 + d2) - segment_length) <= eps
        
        return (point_on_segment(point, self.vertex1, self.vertex2, epsilon) or
                point_on_segment(point, self.vertex2, self.vertex3, epsilon) or
                point_on_segment(point, self.vertex3, self.vertex1, epsilon))
    
    def GetDistance(self, point: np.ndarray) -> float:
        # Distance to edges or vertices
        def point_to_segment_distance(p, v1, v2):
            seg_vec = v2 - v1
            pt_vec = p - v1
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq == 0:
                return np.linalg.norm(pt_vec)
            t = max(0, min(1, np.dot(pt_vec, seg_vec) / seg_len_sq))
            projection = v1 + t * seg_vec
            return np.linalg.norm(p - projection)
        
        d1 = point_to_segment_distance(point, self.vertex1, self.vertex2)
        d2 = point_to_segment_distance(point, self.vertex2, self.vertex3)
        d3 = point_to_segment_distance(point, self.vertex3, self.vertex1)
        return min(d1, d2, d3)
    
    def GetClosestPoint(self, point: np.ndarray) -> float:
        # Closest point on edges or vertices
        def point_to_segment_closest(p, v1, v2):
            seg_vec = v2 - v1
            pt_vec = p - v1
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq == 0:
                return v1
            t = max(0, min(1, np.dot(pt_vec, seg_vec) / seg_len_sq))
            return v1 + t * seg_vec
        
        candidates = [
            point_to_segment_closest(point, self.vertex1, self.vertex2),
            point_to_segment_closest(point, self.vertex2, self.vertex3),
            point_to_segment_closest(point, self.vertex3, self.vertex1)
        ]
        closest_point = min(candidates, key=lambda p: np.linalg.norm(point - p))
        return closest_point
    
    @property
    def normals(self) -> list[np.ndarray]:
        normal = self.GetNormal((self.vertex1 + self.vertex2 + self.vertex3) / 3)
        return [normal]
    
    @property
    def tangents(self) -> list[np.ndarray]:
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex2
        edge3 = self.vertex1 - self.vertex3
        return [edge / np.linalg.norm(edge) for edge in [edge1, edge2, edge3]]

    @property
    def area(self) -> float:
        # Heron's formula
        a = np.linalg.norm(self.vertex1 - self.vertex2)
        b = np.linalg.norm(self.vertex2 - self.vertex3)
        c = np.linalg.norm(self.vertex3 - self.vertex1)
        s = (a + b + c) / 2
        from math import sqrt
        return sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))

    @property
    def perimeter(self) -> float:
        a = np.linalg.norm(self.vertex1 - self.vertex2)
        b = np.linalg.norm(self.vertex2 - self.vertex3)
        c = np.linalg.norm(self.vertex3 - self.vertex1)
        return a + b + c

    @property
    def side_lengths(self) -> tuple[float, float, float]:
        a = np.linalg.norm(self.vertex1 - self.vertex2)
        b = np.linalg.norm(self.vertex2 - self.vertex3)
        c = np.linalg.norm(self.vertex3 - self.vertex1)
        return (a, b, c)
    
    @property
    def dimensions(self) -> int:
        return 2
    
    @property
    def volume(self) -> float:
        return 0.0

    def __repr__(self):
        return f"Triangle(vertex1={self.vertex1}, vertex2={self.vertex2}, vertex3={self.vertex3})"

class Polygon(Shape):
    def __init__(self, vertices: list[np.ndarray], name: str = "Polygon"):
        super().__init__(name=name)
        self.vertices = vertices
        self.origin = np.mean(vertices, axis=0)
    
    def CheckPoint(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        pass

    def CheckIntersection(self, ray: Ray) -> bool:
        pass

    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        pass
    
    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        pass

    @property
    def normals(self) -> list[np.ndarray]:
        pass

    @property
    def tangents(self) -> list[np.ndarray]:
        pass

    @property
    def area(self) -> list[np.ndarray]:
        pass
    
    @property
    def perimeter(self) -> list[np.ndarray]:
        pass

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def volume(self) -> float:
        return 0.0

    def __repr__(self):
        return f"Polygon(vertices={self.vertices})"
    
class Shape3D(Shape):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @property
    def dimensions(self) -> int:
        return 3
    
    @property
    def volume(self) -> float:
        raise NotImplementedError("Volume property must be implemented in 3D shape subclasses")
    
    @property
    def surface_area(self) -> float:
        raise NotImplementedError("Surface area property must be implemented in 3D shape subclasses")
    
    @property
    def surface_normals(self) -> list[np.ndarray]:
        raise NotImplementedError("Surface normals property must be implemented in 3D shape subclasses")
    
    @property
    def surface_tangents(self) -> list[np.ndarray]:
        raise NotImplementedError("Surface tangents property must be implemented in 3D shape subclasses")

    def ConvexHull(self, resolution: Optional[int] = 100) -> list[np.ndarray]:
        """Compute the vertices of the convex hull of the 3D shape.
            Arguments:
                resolution (int, optional): The number of points to use for approximating curved surfaces (points per face).
            Returns:
                list[np.ndarray]: A list of points representing the convex hull vertices.
        """
        raise NotImplementedError("ConvexHull method must be implemented in 3D shape subclasses")

    def __repr__(self):
        return f"Shape3D()"
    
class Cube(Shape3D):
    def __init__(self, center: np.ndarray, side_length: float, name: str = "Cube"):
        super().__init__(name=name)
        self.center = center
        if side_length <= 0:
            raise AttributeError("The side length of the Cube must be greater than 0")
        self.side_length = side_length

        self.origin = center
        self.rotation = np.zeros(3)
    
    @property
    def volume(self) -> float:
        return self.side_length ** 3
    
    @property
    def surface_area(self) -> float:
        return 6 * (self.side_length ** 2)
    
    def CheckPoint(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        half_side = self.side_length / 2
        return all(abs(point[i] - self.center[i]) <= half_side + epsilon for i in range(3))

    def ConvexHull(self) -> list[np.ndarray]:
        points = []
        half_side = self.side_length / 2

        for dx in [-half_side, half_side]:
            for dy in [-half_side, half_side]:
                for dz in [-half_side, half_side]:
                    points.append(self.center + np.array([dx, dy, dz]))
        
        def transform():
            pass  # Placeholder for transformation logic if needed in future

        return points

    def CheckRayIntersection(self, ray):
        return super().CheckRayIntersection(ray)

    def GetRayIntersections(self, ray: Ray) -> list[np.ndarray]:
        return super().GetRayIntersections(ray)
    
    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        return super().GetNormal(point)
    
    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        return super().GetTangent(point)

    def GetBinormal(self, point: np.ndarray) -> np.ndarray:
        return super().GetBinormal(point)
    
    def CheckPointInside(self, point, epsilon):
        return super().CheckPointInside(point, epsilon)
    
    def CheckPointOnEdge(self, point, epsilon):
        return super().CheckPointOnEdge(point, epsilon)

    def GetDistance(self, point: np.ndarray) -> float:
        return super().GetDistance(point)
    
    def GetClosestPoint(self, point: np.ndarray) -> float:
        return super().GetClosestPoint(point)

    def ApplyTransform(self, transform: Transform) -> None:
        self.center = transform.apply_to_point(self.center)
        # Assuming uniform scaling for side length
        uniform_scale = np.mean(transform.scale)
        self.side_length *= uniform_scale
        self.rotation += transform.rotation

    def __repr__(self):
        return f"Cube(center={self.center}, side_length={self.side_length})"
    
class Sphere(Shape3D):
    def __init__(self, center: np.ndarray, radius: float, name: str = "Sphere"):
        super().__init__(name=name)
        self.center = center
        if radius <= 0:
            raise AttributeError("The radius of the Sphere must be greater than 0")
        self.radius = radius

        self.origin = center
    
    @property
    def volume(self) -> float:
        from math import pi
        return (4/3) * pi * self.radius ** 3
    
    @property
    def surface_area(self) -> float:
        from math import pi
        return 4 * pi * self.radius ** 2
    
    def CheckPoint(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        return abs(np.linalg.norm(point - self.center) - self.radius) <= epsilon
    
    def CheckRayIntersection(self, ray) -> bool:
        d = ray.orientation 
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        return discriminant >= 0


    def GetRayIntersections(self, ray: Ray) -> list[np.ndarray]:
        d = ray.orientation 
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return []
        elif discriminant == 0:
            t = -b / (2 * a)
            return np.array([ray.point_at(t)]) if t >= 0 else []
        else:
            sqrt_disc = discriminant ** 0.5
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            ts = [t for t in [t1, t2] if t >= 0]
            return np.array([ray.point_at(t) for t in sorted(ts)])
        
    def CheckPointOnEdge(self, point, epsilon):
        return abs(np.linalg.norm(point - self.center) - self.radius) <= epsilon
    
    def CheckPointInside(self, point, epsilon):
        return np.linalg.norm(point - self.center) < self.radius - epsilon
    
    def GetDistance(self, point: np.ndarray) -> float:
        return abs(np.linalg.norm(point - self.center) - self.radius)
    
    def GetClosestPoint(self, point: np.ndarray) -> float:
        direction = point - self.center
        direction_normalized = direction / np.linalg.norm(direction)
        return self.center + direction_normalized * self.radius
    
    def GetNormal(self, point: np.ndarray) -> np.ndarray:
        if not self.CheckPoint(point, 0.01):
            raise ValueError("Point is not on the sphere")
        vec = point - self.center
        return vec / np.linalg.norm(vec)
    
    def GetTangent(self, point: np.ndarray) -> np.ndarray:
        if not self.CheckPoint(point, 0.01):
            raise ValueError("Point is not on the sphere")
        normal = self.GetNormal(point)
        # Find a vector not parallel to normal
        if abs(normal[0]) < 0.9:
            arbitrary = np.array([1, 0, 0])
        else:
            arbitrary = np.array([0, 1, 0])
        tangent = np.cross(normal, arbitrary)
        return tangent / np.linalg.norm(tangent)
    
    def ApplyTransform(self, transform: Transform) -> None:
        self.center = transform.apply_to_point(self.center)
        # Assuming uniform scaling for radius
        uniform_scale = np.mean(transform.scale)
        self.radius *= uniform_scale

    def GetBinormal(self, point: np.ndarray) -> np.ndarray:
        if not self.CheckPoint(point, 0.01):
            raise ValueError("Point is not on the sphere")
        normal = self.GetNormal(point)
        tangent = self.GetTangent(point)
        binormal = np.cross(normal, tangent)
        return binormal / np.linalg.norm(binormal)

    def ConvexHull(self, resolution: int = 100) -> list[np.ndarray]:
        points = []
        phi = (1 + 5 ** 0.5) / 2  # golden ratio
        for i in range(resolution):
            theta = 2 * np.pi * i / phi
            y = 1 - (i / float(resolution - 1)) * 2  # y goes from 1 to -1
            radius = (1 - y * y) ** 0.5  # radius at y

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append(self.center + self.radius * np.array([x, y, z]))
        return points

    def __repr__(self):
        return f"Sphere(center={self.center}, radius={self.radius})"
    
class Prisim(Shape3D):
    def __init__(self, base_polygon: Polygon, height: float, name: str = "Prisim"):
        super().__init__(name=name)
        self.base_polygon = base_polygon
        if height <= 0:
            raise AttributeError("The height of the Prisim must be greater than 0")
        self.height = height

        self.origin = base_polygon.origin
    
    @property
    def volume(self) -> float:
        return self.base_polygon.area * self.height
    
    @property
    def surface_area(self) -> float:
        # Placeholder implementation
        return 2 * self.base_polygon.area + self.base_polygon.perimeter * self.height
    
    def CheckPoint(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        pass

    def ConvexHull(self) -> list[np.ndarray]:
        pass

    def __repr__(self):
        return f"Prisim(base_polygon={self.base_polygon}, height={self.height})"
    
class Pyramid(Shape3D):
    def __init__(self, base_polygon: Polygon, height: float, name: str = "Pyramid"):
        super().__init__(name=name)
        self.base_polygon = base_polygon
        if height <= 0:
            raise AttributeError("The height of the Pyramid must be greater than 0")
        self.height = height

        self.origin = base_polygon.origin
    
    @property
    def volume(self) -> float:
        return (1/3) * self.base_polygon.area * self.height
    
    @property
    def surface_area(self) -> float:
        # Placeholder implementation
        return self.base_polygon.area  # + lateral area calculation needed
    
    def CheckPoint(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        pass

    def ConvexHull(self) -> list[np.ndarray]:
        pass

    def __repr__(self):
        return f"Pyramid(base_polygon={self.base_polygon}, height={self.height})"

class Capsule(Shape3D):
    def __init__(self, point1: np.ndarray, point2: np.ndarray, radius: float, name: str = "Capsule"):
        super().__init__(name=name)
        self.point1 = point1
        self.point2 = point2
        if radius <= 0:
            raise AttributeError("The radius of the Capsule must be greater than 0")
        self.radius = radius

        self.origin = (point1 + point2) / 2
    
    @property
    def volume(self) -> float:
        from math import pi
        cylinder_height = np.linalg.norm(self.point2 - self.point1)
        cylinder_volume = pi * self.radius ** 2 * cylinder_height
        sphere_volume = (4/3) * pi * self.radius ** 3
        return cylinder_volume + sphere_volume
    
    @property
    def surface_area(self) -> float:
        from math import pi
        cylinder_height = np.linalg.norm(self.point2 - self.point1)
        cylinder_area = 2 * pi * self.radius * cylinder_height
        sphere_area = 4 * pi * self.radius ** 2
        return cylinder_area + sphere_area
    
    def CheckPoint(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
        pass

    def ConvexHull(self, resolution = 100) -> list[np.ndarray]:
        pass

    def __repr__(self):
        return f"Capsule(point1={self.point1}, point2={self.point2}, radius={self.radius})"

class VObject:
    def __init__(self, shape: Shape, transform: Transform, material=None, name: str = "VObject"):
        self.shape = shape
        self.transform = transform
        self.material = material
        self.name = name

    @property
    def position(self) -> np.ndarray:
        return self.transform.position
    @position.setter
    def position(self, value: np.ndarray) -> None:
        self.transform.position = value
    @property
    def rotation(self) -> np.ndarray:
        return self.transform.rotation
    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        self.transform.rotation = value
    @property
    def scale(self) -> np.ndarray:
        return self.transform.scale
    @scale.setter
    def scale(self, value: np.ndarray) -> None:
        self.transform.scale = value

    def ApplyTransform(self) -> None:
        self.shape.origin = self.position
        # Note: For simplicity, only updating origin. Full implementation would transform all vertices/points.
    
    def __repr__(self):
        return f"VObject(name={self.name}, shape={self.shape}, transform={self.transform}, material={self.material})"

from typing import Any, Dict, Type, Optional
from abc import ABC, abstractmethod

class ShapeFactory(ABC):
    @abstractmethod
    def create(self, **kwargs: Any) -> Shape:
        return Shape(**kwargs)

    def __repr__(self):
        return f"ShapeFactory()"

class CircleFactory(ShapeFactory):
    def create(self, center: np.ndarray, radius: float, name: str = "Circle") -> Circle:
        return Circle(center=center, radius=radius, name=name)
    
    def __repr__(self):
        return f"CircleFactory()"
    
class SphereFactory(ShapeFactory):
    def create(self, center: np.ndarray, radius: float, name: str = "Sphere") -> Sphere:
        return Sphere(center=center, radius=radius, name=name)
    
    def __repr__(self):
        return f"SphereFactory()"
    
class TriangleFactory(ShapeFactory):
    def create(self, vertex1: np.ndarray, vertex2: np.ndarray, vertex3: np.ndarray, name: str = "Triangle") -> Triangle:
        return Triangle(vertex1=vertex1, vertex2=vertex2, vertex3=vertex3, name=name)
    
    def __repr__(self):
        return f"TriangleFactory()"
    
class PolygonFactory(ShapeFactory):
    def create(self, *vertices) -> Polygon:
        # If a single iterable of vertices was passed, handle that too.
        if len(vertices) == 1 and hasattr(vertices[0], '__iter__') and not isinstance(vertices[0], (str, bytes)):
            verts = vertices[0]
        else:
            verts = vertices
        return Polygon(verts)
    
class CubeFactory(ShapeFactory):
    def create(self, center: np.ndarray, side_length: float, name: str = "Cube") -> Cube:
        return Cube(center=center, side_length=side_length, name=name)
    
    def __repr__(self):
        return f"CubeFactory()"
    
class PrismFactory(ShapeFactory):
    def create(self, base_polygon: Polygon, height: float, name: str = "Prism") -> Prisim:
        return Prisim(base_polygon=base_polygon, height=height, name=name)
    
    def __repr__(self):
        return f"PrismFactory()"
    
class PyramidFactory(ShapeFactory):
    def create(self, base_polygon: Polygon, height: float, name: str = "Pyramid") -> Pyramid:
        return Pyramid(base_polygon=base_polygon, height=height, name=name)
    
    def __repr__(self):
        return f"PyramidFactory()"
    
class CapsuleFactory(ShapeFactory):
    def create(self, point1: np.ndarray, point2: np.ndarray, radius: float, name: str = "Capsule") -> Capsule:
        return Capsule(point1=point1, point2=point2, radius=radius, name=name)
    
    def __repr__(self):
        return f"CapsuleFactory()"