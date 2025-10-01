from src.Basic import Ray
from src.Lighting import Color, LightRay
from src. Projections import *
import numpy as np

class Raycaster:
    max_bounces: int = 5
    max_distance: float = 1000.0
    epsilon: float = 0.001 # Small offset to avoid self-intersection
    raycast_steps: int = 1000 # Number of steps for ray marching

    def __init__(self):
        pass

    def Raycast(self, scene, rays_per_pixle: int) -> tuple[np.ndarray, np.ndarray, object] | None:
        cam = scene.camera
        width, height = cam.width, cam.height

        rays = []
        for y in range(height):
            for x in range(width):
                for r in range(rays_per_pixle):
                    # Jitter for anti-aliasing
                    jitter_u = np.random.uniform(-0.5, 0.5) / width
                    jitter_v = np.random.uniform(-0.5, 0.5) / height
                    u = (x + 0.5 + jitter_u) / width
                    v = (y + 0.5 + jitter_v) / height
                    direction = cam.transfrom.forward + (u - 0.5) * cam.transfrom.right + (v - 0.5) * cam.transform.up
                    direction = direction.normalized()
                    ray = Ray(cam.transform.position, direction)
                    rays.append(ray)

    def RayIntersect(self, ray: Ray, scene): 
        """
        Perform ray marching to find the intersection of a ray with the scene.
        Returns the hit information (hit point, normal, material) or None if no hit.
        """
        distance_traveled = 0.0
        for step in range(self.raycast_steps):
            point = ray.origin + ray.direction * distance_traveled

            distance_to_closest, closest_object = scene.distance_estimator(point) #

            if distance_to_closest < self.epsilon:
                print("Hit at distance:", distance_traveled)
                print("After:", step, "steps")

                return closest_object
            
            distance_traveled += distance_to_closest

            if distance_traveled >= self.max_distance:
                print("Max distance reached without hit")
                print("After:", step, "steps")
                break
        return None
    
    def RayCasting(self, ray: Ray, scene) -> Color:
        light = LightRay(ray.origin, ray.direction, Color(1, 1, 1), intensity=1.0)
        attenuation = 0.1
        
        for bounce in range(self.max_bounces):
            hit_info = self.Raycast(ray, scene, self.max_distance, self.epsilon, self.raycast_steps)
            if hit_info is None:

                color += attenuation * scene.background_color #

                break
            hit_point, normal, material = hit_info
            color += attenuation * material.emission
            
            for light in scene.lights: #
                light_dir = (light.position - hit_point).normalized()
                light_distance = (light.position - hit_point).length()
                shadow_ray = Ray(hit_point + normal * self.epsilon, light_dir)
                shadow_hit = self.Raycast(shadow_ray, scene, light_distance, self.epsilon, self.raycast_steps)
                if shadow_hit is None:
                    lambert = max(normal.dot(light_dir), 0)
                    color += attenuation * material.color * light.color * lambert
            attenuation *= material.color
            if material.reflectivity <= 0:
                break
            reflect_dir = ray.direction - 2 * ray.direction.dot(normal) * normal
            ray = Ray(hit_point + normal * self.epsilon, reflect_dir.normalized())
        return color
    
    def __repr__(self):
        return f"Raycaster(max_bounces={self.max_bounces}, max_distance={self.max_distance}, epsilon={self.epsilon}, raycast_steps={self.raycast_steps})"
