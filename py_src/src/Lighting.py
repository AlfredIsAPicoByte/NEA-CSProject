from src.Basic import Vector, Ray
import random

class Color:
    def __init__(self, r: float, g: float, b: float, alpha: float = 1.0):
        """
        Color values are clamped between 0.0 and 1.0
        0.0 = no intensity, 1.0 = full intensity
        Alpha is the opacity of the color, 0.0 = fully transparent, 1.0 = fully opaque
        """
        self.r = clamp(r)
        self.g = clamp(g)
        self.b = clamp(b)
        self.alpha = clamp(alpha)

    def __mul__(self, other):
        if isinstance(other, Color):
            return Color(self.r * other.r, self.g * other.g, self.b * other.b, self.alpha * other.alpha)
        elif isinstance(other, (int, float)):
            return Color(self.r * other, self.g * other, self.b * other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Color):
            return Color(self.r + other.r, self.g + other.g, self.b + other.b, self.alpha + other.alpha)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Color):
            return Color(self.r - other.r, self.g - other.g, self.b - other.b, self.alpha - other.alpha)
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, Color):
            return Color(self.r / other.r, self.g / other.g, self.b / other.b, self.alpha / other.alpha)
        elif isinstance(other, (int, float)):
            return Color(self.r / other, self.g / other, self.b / other)
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Color):
            return (self.r == other.r and self.g == other.g and self.b == other.b and self.alpha == other.alpha)
        return NotImplemented

    def __repr__(self):
        return f"Color({self.r}, {self.g}, {self.b})"

def clamp(value, min_value: float|int = 0.0, max_value: float|int = 1.0):
    return max(min_value, min(value, max_value))

class RayColor(Color, Ray):
    def __init__(self, origin: Vector, direction: Vector, color: Color):
        """
        Combines a Ray and a Color into one object.
        """
        Ray.__init__(self, origin, direction.Normalize())
        Color.__init__(self, color.r, color.g, color.b, color.alpha)

    def __repr__(self):
        return f"RayColor(origin={self.origin}, direction={self.direction}, color={self.color})"

class SimpleMaterial:
    def __init__ (self, color: Color, roughness: float, glossiness: float):
        """
        A simple material that reflects light based on its color, roughness, and glossiness.
        - color: The base color of the material.
        - roughness: A value between 0.0 and 1.0 that determines
            how rough the surface is. 0.0 = perfectly smooth, 1.0 = very rough.
        - glossiness: A value between 0.0 and 1.0 that determines
            how glossy the surface is. 0.0 = matte, 1.0 = perfectly glossy.
        """
        self.color = color
        self.roughness = roughness
        self.glossiness = clamp(glossiness)

    def ReflectColor(self, light_color: Color):
        return Color(
            self.color.r * light_color.r,
            self.color.g * light_color.g,
            self.color.b * light_color.b,
            self.color.alpha * light_color.alpha
        )

    def ReflectRay(self, ray: Ray, normal: Vector, origin: Vector):
        reflected_direction = ray.direction - 2 * (ray.direction.Dot(normal)) * normal

        # Glossiness controls the spread of the random vector
        spread = (1.0 - self.glossiness) * self.roughness
        if spread > 0:
            random_vec = Vector(
                random.uniform(-spread, spread),
                random.uniform(-spread, spread),
                random.uniform(-spread, spread)
            ).Normalize()
            reflected_direction += random_vec

        return Ray(origin, reflected_direction.Normalize())

    def ReflectRayColor(self, rayCol: RayColor, normal: Vector, origin: Vector):
        return RayColor(origin, self.ReflectRay(Ray(rayCol.origin, rayCol.direction), normal), self.ReflectColor(rayCol.color))
    
    def __repr__(self):
        return f"SimpleMaterial(color={self.color}, roughness={self.roughness}, glossiness={self.glossiness})"