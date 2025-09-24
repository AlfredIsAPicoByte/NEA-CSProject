from src.Basic import Ray
from src.Lighting import LightRay
from src. Projections import *
import numpy as np

def Raycast(ray: Ray, scene, max_distance: float, epsilon: float, steps: int):
    """
    Perform ray marching to find the intersection of a ray with the scene.
    Returns the hit information (hit point, normal, material) or None if no hit.
    """
    distance_traveled = 0.0
    for step in range(steps):
        point = ray.origin + ray.direction * distance_traveled
        distance_to_closest = 0 # scene.distance_estimator(point)
        if distance_to_closest < epsilon:
            normal = scene.estimate_normal(point)
            material = scene.get_material(point)
            print("Hit at distance:", distance_traveled)
            print("After:", step, "steps")
            return point, normal, material
        distance_traveled += distance_to_closest

        if distance_traveled >= max_distance:
            print("Max distance reached without hit")
            print("After:", step, "steps")
            break
    return None

class Simple:
    max_bounces: int = 5
    max_distance: float = 1000.0
    epsilon: float = 0.001 # Small offset to avoid self-intersection
    raycast_steps: int = 1000 # Number of steps for ray marching

    def __init__(self):
        pass

    def TraceRay(self, ray: Ray, scene) -> Color:
        light = LightRay(ray.origin, ray.direction, Color(1, 1, 1), intensity=1.0)

        for bounce in range(self.max_bounces):
            hit_info = Raycast(ray, scene, self.max_distance, self.epsilon, self.raycast_steps)
            if hit_info is None:
                color += attenuation * scene.background_color
                break
            hit_point, normal, material = hit_info
            color += attenuation * material.emission
            for light in scene.lights:
                light_dir = (light.position - hit_point).normalized()
                light_distance = (light.position - hit_point).length()
                shadow_ray = Ray(hit_point + normal * self.epsilon, light_dir)
                shadow_hit = Raycast(shadow_ray, scene, light_distance, self.epsilon, self.raycast_steps)
                if shadow_hit is None:
                    lambert = max(normal.dot(light_dir), 0)
                    color += attenuation * material.color * light.color * lambert
            attenuation *= material.color
            if material.reflectivity <= 0:
                break
            reflect_dir = ray.direction - 2 * ray.direction.dot(normal) * normal
            ray = Ray(hit_point + normal * self.epsilon, reflect_dir.normalized())
        return color
