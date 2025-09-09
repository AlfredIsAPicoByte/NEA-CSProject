from src.Basic import Vector, Matrix, Ray

class Shape:
    def __init__(self, name: str = "Shape"):
        self.name = name
    
    def CheckPoint(self, point: Vector) -> bool:
        raise NotImplementedError("CheckPoint method must be implemented in subclasses")

    def CheckIntersection(self, ray: Ray) -> bool:
        raise NotImplementedError("CheckIntersection method must be implemented in subclasses")

    def GetNormal(self, point: Vector) -> Vector:
        raise NotImplementedError("GetNormal method must be implemented in subclasses")

    def GetTangent(self, point: Vector) -> Vector:
        raise NotImplementedError("GetTangent method must be implemented in subclasses")
    
    def Area(self) -> float:
        raise NotImplementedError("Area method must be implemented in subclasses")

    def Perimeter(self) -> float:
        raise NotImplementedError("Perimeter method must be implemented in subclasses")

    def __repr__(self):
        return f"Shape()"

class Circle(Shape):
    def __init__(self, center: Vector, radius: float, name: str = "Circle"):
        super().__init__(name)
        self.center = center
        self.radius = radius
    
    def CheckPoint(self, point: Vector, uncertainty: float = 0) -> bool:
        return self.radius - uncertainty < (point - self.center).Magnitude() <= self.radius + uncertainty
    
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
        a = d.Dot(d)
        b = 2 * d.Dot(s)
        c = s.Dot(s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        if discriminant < 0:
            return False
        else:
            return True

    def GetIntersection(self, ray: Ray) -> list[Vector]:
        if not self.CheckIntersection(ray):
            return []
        
        d = ray.direction
        s = ray.origin - self.center
        a = d.Dot(d)
        b = 2 * d.Dot(s)
        c = s.Dot(s) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        
        t = -b - (discriminant) / a
        return ray.PointAtParameter(t)

    def GetNormal(self, point: Vector) -> Vector:
        if not self.CheckPoint(point, 0.01):
            raise ValueError("Point is not on the circle")
        return (point - self.center).Normalize()

    def GetTangent(self, point: Vector) -> Vector:
        if not self.CheckPoint(point, 0.01):
            raise ValueError("Point is not on the circle")
        return (point - self.center).Normalize()

    def Area(self) -> float:
        from math import pi
        return pi * self.radius ** 2

    def Perimeter(self) -> float:
        from math import pi
        return 2 * pi * self.radius

    def Diameter(self) -> float:
        return 2 * self.radius

    def Circumference(self) -> float:
        return self.Perimeter()

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"


class Triangle(Shape):
    def __init__(self, vertex1: Vector, vertex2: Vector, vertex3: Vector, name: str = "Triangle"):
        super().__init__(name)
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.vertex3 = vertex3
    
    # Using a geometric solution to find intersection with triangles
    # P(t) = O + Rt
    # D = -(Ax )

    def CheckPoint(self, point: Vector) -> bool:
        pass

    def CheckIntersection(self, ray: Ray) -> bool:
        pass

    def GetNormal(self, point: Vector) -> Vector:
        if not self.CheckPoint(point):
            raise ValueError("Point is not on the triangle")
        edge1 = self.vertex2 - self.vertex1
        edge2 = self.vertex3 - self.vertex1
        return edge1.Cross(edge2).Normalize()

    def Area(self) -> float:
        # Heron's formula
        a = (self.vertex1 - self.vertex2).Magnitude()
        b = (self.vertex2 - self.vertex3).Magnitude()
        c = (self.vertex3 - self.vertex1).Magnitude()
        s = (a + b + c) / 2
        from math import sqrt
        return sqrt(s * (s - a) * (s - b) * (s - c))

    def Perimeter(self) -> float:
        a = (self.vertex1 - self.vertex2).Magnitude()
        b = (self.vertex2 - self.vertex3).Magnitude()
        c = (self.vertex3 - self.vertex1).Magnitude()
        return a + b + c

    def SideLengths(self) -> tuple[float, float, float]:
        a = (self.vertex1 - self.vertex2).Magnitude()
        b = (self.vertex2 - self.vertex3).Magnitude()
        c = (self.vertex3 - self.vertex1).Magnitude()
        return (a, b, c)

    def __repr__(self):
        return f"Triangle(vertex1={self.vertex1}, vertex2={self.vertex2}, vertex3={self.vertex3})"

class Polygon(Shape):
    def __init__(self, vertices: list[Vector], name: str = "Polygon"):
        super().__init__(name)
        self.vertices = vertices
    
    def CheckPoint(self, point: Vector) -> bool:
        pass

    def CheckIntersection(self, ray: Ray) -> bool:
        pass

    def GetNormal(self, point: Vector) -> Vector:
        pass

    def Area(self) -> float:
        pass

    def Perimeter(self) -> float:
        pass

    def __repr__(self):
        return f"Polygon(vertices={self.vertices})"