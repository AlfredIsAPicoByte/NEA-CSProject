import numpy as np
import math
from enum import Enum
from dataclasses import dataclass, field

from src.Data.Color import Color
from src.Utilities.Common import unit, attenuate_inv_sqr_distance_arguments

class LightType(Enum):
    POINT = 1        # Light bulb: Radiates in all directions from a position
    DIRECTIONAL = 2  # Sun: Parallel rays, infinite distance, no falloff
    SPOT = 3         # Flashlight: Cone of light from a position

@dataclass
class Light:
    """
    A unified light class that handles Point, Directional, and Spot behaviors.
    Ensure direction vectors are normalized.
    """
    type: LightType = LightType.POINT

    color: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))
    intensity: float = 100.0
    radius: float = 1
    
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))  # For Directional and Spot lights

    # Angles in degrees
    spot_inner_angle: float = 15.0 # Full brightness inside this angle
    spot_outer_angle: float = 45.0 # Falls off to zero at this angle

    def __post_init__(self):
        if self.radius <= 1e-16:
            self.radius = 1e-16

    def get_direction(self, light_pos: np.ndarray, hit_point: np.ndarray) -> np.ndarray:
        """
        Returns the unit vector POINTING TO the light from the hit_point.
        """
        if self.type == LightType.DIRECTIONAL:
            # Directional lights are infinitely far away.
            # The direction is constant everywhere in the scene.
            return -self.direction
            
        else: # POINT or SPOT
            # Vector from Surface -> Light
            to_light = light_pos - hit_point
            dist = np.linalg.norm(to_light)
            
            # Avoid division by zero
            if dist < 1e-6:
                return np.array([0.0, 1.0, 0.0])
                
            return to_light / dist
    
    def get_distance(self, light_pos: np.ndarray, hit_point: np.ndarray, bias: float = 1e-6) -> float:
        """
        Returns a tuple of (unit vector POINTING TO the light, distance to light) from the hit_point.
        The distance is infinite for Directional lights.
        """
        if self.type == LightType.DIRECTIONAL:
            # Directional lights are infinitely far away.
            # The direction is constant everywhere in the scene.
            # We negate self.direction because we want the vector pointing TO the light source.
            return float('inf')
            
        else: # POINT or SPOT
            # Vector from Surface -> Light
            to_light = light_pos - hit_point
            dist = np.linalg.norm(to_light)
            
            # Avoid division by zero
            if dist < bias:
                return float(dist)
                
            return float(dist)

    def get_radiance(self, light_pos: np.ndarray, hit_point: np.ndarray) -> Color:
        """
        Calculates the actual light energy (Radiance) arriving at the hit_point.
        Handles Inverse Square Law (Point/Spot) and Cone Attenuation (Spot).
        """
        # 1. Base Intensity
        radiance = self.color * self.intensity
        
        # 2. Distance Attenuation
        if self.type == LightType.DIRECTIONAL:
            # The Sun doesn't get dimmer if you walk 10 meters.
            # No distance attenuation.
            return radiance
        
        # Calculate distance
        dist = np.linalg.norm(light_pos - hit_point)
        
        # Apply Inverse Square Law (1 / distance ^ 2 + radius ^ 2)
        radiance *= attenuate_inv_sqr_distance_arguments(1e-8, dist, self.radius)

        # 3. Spot Cone Attenuation
        if self.type == LightType.SPOT:
            # Check angle between "Light Facing Direction" and "Vector to Pixel"
            # surface_to_light = unit(light_pos - hit_point)
            # light_dir = direction
            
            # We need the vector pointing FROM light TO surface
            light_to_surface = unit(hit_point - light_pos)
            
            # Dot product: 1.0 = dead center, 0.0 = 90 degrees away
            theta = np.dot(light_to_surface, self.direction)
            
            # Convert angles to cosines for fast comparison
            cos_inner = math.cos(math.radians(self.spot_inner_angle))
            cos_outer = math.cos(math.radians(self.spot_outer_angle))
            
            if theta > cos_inner:
                # Inside inner cone: Full brightness
                spot_factor = 1.0
            elif theta < cos_outer:
                # Outside outer cone: Dark
                spot_factor = 0.0
            else:
                # In the penumbra (fading edge): Interpolate
                # Map theta from [cos_outer, cos_inner] to [0, 1]
                t = (theta - cos_outer) / (cos_inner - cos_outer)
                # Smoothstep for softer edges
                spot_factor = t * t * (3.0 - 2.0 * t)
                
            radiance *= spot_factor

        return radiance