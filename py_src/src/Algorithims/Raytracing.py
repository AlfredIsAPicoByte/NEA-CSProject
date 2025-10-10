from src.Camera import CameraObject
from src.PrimaryStructures import Ray
from src.Geometry import Shape
from src.Luminance import ColorData, LightRay
import numpy as np
import random

"""

"""

class Raytracer:
    max_bounces: int = 5
    max_distance: float = 1000.0
    epsilon: float = 0.001 # Small offset to avoid self-intersection
    raycast_steps: int = 1000 # Number of steps for ray marching

    def __init__(self):
        self.rays: list[Ray|LightRay] = []
        self.bounces: list[int] = [] # bounces
        self.distance: list[float] = []

    def CamaraCast(self, camara: CameraObject, rays_per_pixle: int, seed: int|None = None):
        if seed is not None:
            random.seed(seed)
        
        width, height = camara.width, camara.height

        rays_casted = []
        for y in range(height):
            for x in range(width):
                for r in range(rays_per_pixle):
                    # Jitter for anti-aliasing
                    jitter_u = random.uniform(-0.5, 0.5) / width
                    jitter_v = random.uniform(-0.5, 0.5) / height

                    # Set the pixle positions for the rays
                    u = (x + 0.5 + jitter_u) / width
                    v = (y + 0.5 + jitter_v) / height

                    # Align the rays to the portion of the scron from the 
                    direction = camara.transfrom.forward + (u - 0.5) * camara.transfrom.right + (v - 0.5) * camara.transform.up

                    ray = Ray(camara.transform.position, np.linalg.norm(direction), name=f"{str(x)}_{str(y)}_{str(r)}")
                    rays_casted.append(ray)

                    self.bounces.append(0)
                    self.distance.append(0)
        
        self.rays = rays_casted

    def IntractCast(self, index: int, object: Shape):
        normal = object.GetNormal(self.rays[index].point_at(self.distance[index]))
        pass

    def RayIntersection(self, index: int, scene) -> Shape | None: 
        """
        Perform ray marching to find the intersection of a ray with the scene.
        Returns the hit information (hit point, normal, material) or None if no hit.
        """
        distance_traveled = 0.0

        for step in range(self.raycast_steps):
            point = self.ray[index].origin + self.ray[index].direction * distance_traveled

            distance_to_closest, closest_object = scene.distance_estimator(point) #

            if distance_to_closest <= self.epsilon:
                print("Hit at distance:", distance_traveled)
                print("After:", step, "steps")

                self.distance[index] = distance_to_closest
                return closest_object
            
            distance_traveled += distance_to_closest

            if distance_traveled >= self.max_distance:
                print("Max distance reached without hit")
                return None
        print("Max steps taken for this ray")
        return None
    
    def RayInteraction(self, index: int, hit_object: Shape):
        if hit_object is None:
            pass
        
        pass
    
    def Raycast(self, index: int, scene) -> ColorData:
        lightRay = LightRay(ray.origin, ray.direction, ColorData(1, 1, 1), intensity=1.0)

        for bounce in range(self.max_bounces):
            hit_info = self.Raycast(Ray(), scene, self.max_distance, self.epsilon, self.raycast_steps)
            
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
