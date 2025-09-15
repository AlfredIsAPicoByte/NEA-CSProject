from src.Basic import Ray
import numpy as np
import random

class Color:
    def __init__(self, r: float, g: float, b: float, alpha: float = 1.0):
        """
        Color values are clamped between 0.0 and 1.0
        0.0 = no intensity, 1.0 = full intensity
        Alpha is the opacity of the color, 0.0 = fully transparent, 1.0 = fully opaque
        """
        self.rgba = np.array([clamp(r), clamp(g), clamp(b), clamp(alpha)])

    @property
    def red(self):
        return self.rgba[0]
    
    @property
    def green(self):
        return self.rgba[1]

    @property
    def blue(self):
        return self.rgba[2]

    @property
    def alpha(self):
        return self.rgba[3]

    def __mul__(self, other):
        if isinstance(other, Color):
            return Color(*(self.rgba * other.rgba))
        elif isinstance(other, (int, float)):
            return Color(*(self.rgba[:3] * other))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Color):
            return Color(*(self.rgba + other.rgba))
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Color):
            return Color(*(self.rgba - other.rgba))
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, Color):
            return Color(*(self.rgba / other.rgba))
        elif isinstance(other, (int, float)):
            return Color(*(self.rgba / other))
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Color):
            return np.array_equal(self.rgba, other.rgba)
        return NotImplemented

    def __repr__(self):
        return f"Color({self.r}, {self.g}, {self.b})"

def clamp(value, min_value: float|int = 0.0, max_value: float|int = 1.0):
    return max(min_value, min(value, max_value))


class LightRay(Color, Ray):
    def __init__(self, origin: np.ndarray, direction: np.ndarray, color: Color, intensity: float = 1.0):
        Ray.__init__(self, origin, direction / np.linalg.norm(direction))
        Color.__init__(self, color.r, color.g, color.b, color.alpha)
        self.intensity = intensity

    def __repr__(self):
        return f"LightRay(origin={self.origin}, direction={self.direction}, color={self.color}, intensity={self.intensity})"

class SimpleMaterial:
    def __init__ (self, color: Color, roughness: float, glossiness: float):
        """
        A simple material that reflects LightRay based on its color, roughness, and glossiness.
        
        Attribute:
            color: The base color of the material.
            roughness: A value between 0.0 and 1.0 that determines
                how rough the surface is. 0.0 = perfectly smooth, 1.0 = very rough.
            glossiness: A value between 0.0 and 1.0 that determines
                how glossy the surface is. 0.0 = matte, 1.0 = perfectly glossy.
        """
        self.color = color
        self.roughness = roughness
        self.glossiness = clamp(glossiness)

    def ReflectColor(self, LightRay_color: Color):
        return self.color * LightRay_color

    def ReflectRay(self, ray: Ray, normal: np.ndarray, origin: np.ndarray, seed: int|None = None):
        if seed is not None:
            random.seed(seed)
        
        reflected_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal

        spread = (1.0 - self.glossiness) * self.roughness
        if spread > 0:
            random_vec = np.array([
                random.uniform(-spread, spread),
                random.uniform(-spread, spread),
                random.uniform(-spread, spread)
            ])
            reflected_direction += random_vec

        # Normalize the final direction
        return Ray(origin, reflected_direction / np.linalg.norm(reflected_direction))

    def ReflectLightRay(self, lightRay: LightRay, normal: np.ndarray, origin: np.ndarray):
        reflected_ray = self.ReflectRay(Ray(lightRay.origin, lightRay.direction), normal, origin)
        reflected_color = self.ReflectColor(lightRay.color)
        return LightRay(origin, reflected_ray.direction, reflected_color)
    
    def __repr__(self):
        return f"SimpleMaterial(color={self.color}, roughness={self.roughness}, glossiness={self.glossiness})"