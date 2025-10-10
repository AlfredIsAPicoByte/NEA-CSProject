from src.PrimaryStructures import Ray
import numpy as np

"""

"""

class Shape:
    def __init__(self, **kwargs):
        self.name: str = kwargs.get("name", "Default Name")

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

    def __repr__(self):
        return f"Shape()"

### TODO: Update the bellow classes to use the new methods

class Circle(Shape):
    def __init__(self, center: np.ndarray, radius: float, name: str = "Circle"):
        super().__init__(name=name)
        self.center = center
        self.radius = radius
    
    def CheckPoint(self, point: np.ndarray, epsilon: float = 0) -> bool:
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
        
    def CheckIntersection(self, ray: Ray) -> bool:
        d = ray.direction
        s = ray.origin - self.center
        a = np.dot(d, d)
        b = 2 * np.dot(d, s)
        c = np.dot(s, s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        if discriminant < 0:
            return False
        else:
            return True

    def GetIntersection(self, ray: Ray) -> list[np.ndarray]:
        if not self.CheckIntersection(ray):
            return []
        
        d = ray.direction
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

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"


class Triangle(Shape):
    def __init__(self, vertex1: np.ndarray, vertex2: np.ndarray, vertex3: np.ndarray, name: str = "Triangle"):
        super().__init__(name=name)
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.vertex3 = vertex3

        if self.CheckDegeneracy():
            raise ValueError("Triangle is degenerate")

    def CheckDegeneracy(self, epsilon: float = 1e-6) -> bool:
        """Check if any edge of the triangle is degenerate or area is near zero (collinear)."""
        edge1 = np.linalg.norm(self.vertex2 - self.vertex1)
        edge2 = np.linalg.norm(self.vertex3 - self.vertex2)
        edge3 = np.linalg.norm(self.vertex1 - self.vertex3)
        # If any edge is degenerate (zero length), triangle is degenerate
        if edge1 < epsilon or edge2 < epsilon or edge3 < epsilon:
            return True
        # If area is near zero, triangle is degenerate (collinear)
        area = self.area
        if area < epsilon:
            return True
        return False
    
    # Using a geometric solution to find intersection with triangles
    # P(t) = O + Rt
    # D = -(Ax )

    def CheckPoint(self, point: np.ndarray, epsilon: float = 1e-6) -> bool:
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

    def CheckIntersection(self, ray: Ray, epsilon: float = 1e-6) -> bool:
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex1
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        
        if -epsilon < a < epsilon:
            return False  # Ray is parallel to triangle
        
        f = 1.0 / a
        s = ray.origin - self.vertex1
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return False
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        
        if v < 0.0 or u + v > 1.0:
            return False
        
        t = f * np.dot(edge2, q)
        
        if t > epsilon:  # Intersection with the triangle
            return True
        else:  # Line intersection but not a ray intersection
            return False
        
    def GetIntersection(self, ray: Ray, epsilon: float = 1e-6) -> np.ndarray | None:
        if not self.CheckIntersection(ray, epsilon):
            return None
        
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex1
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        
        f = 1.0 / a
        s = ray.origin - self.vertex1
        u = f * np.dot(s, h)
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        
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