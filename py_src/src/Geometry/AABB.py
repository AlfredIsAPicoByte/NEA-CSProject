from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from src.Data.Transform import Transform
from src.Data.Ray import Ray

if TYPE_CHECKING:
    from src.Geometry.Core import Shape

class AABB:
    """
    Axis-Aligned Bounding Box.
    Used for quick rejection tests before checking complex geometry.
    """
    __slots__ = ['min_point', 'max_point']

    def __init__(self, min_point: np.ndarray, max_point: np.ndarray):
        self.min_point = min_point
        self.max_point = max_point

    def intersect(self, ray: Ray, max_t: float =  1e30, bias: float = 1e-9) -> float:
        """
        Slab Method for Ray/AABB intersection.
        Returns distance to entry, or infinity if miss.
        """
        # We use the inverse direction to replace division with multiplication
        # This handles division by zero gracefully (results in +/- inf)
        with np.errstate(divide='ignore'):
            inv_dir = 1.0 / ray.orientation
        
        t0 = (self.min_point - ray.origin) * inv_dir
        t1 = (self.max_point - ray.origin) * inv_dir

        tmin = np.maximum(np.minimum(t0, t1), 0.0)
        tmax = np.minimum(np.maximum(t0, t1), max_t)

        # Find largest entry time and smallest exit time across all axes
        t_enter = np.max(tmin)
        t_exit = np.min(tmax)

        if t_exit >= t_enter and t_exit > 0 and t_enter < max_t:
            # If we are inside the box (t_enter < 0), return 0 or t_exit?
            # Usually for BVH culling, returning t_enter is fine.
            return max(t_enter, 0.0)
        
        return float('inf')

    @staticmethod
    def from_transform_shape(world_Transform: Transform, shape: "Shape", padding: float = 1e-2) -> 'AABB':
        """
        Calculates the world-space AABB for a given object.
        """
        if shape is None:
            return AABB(np.zeros(3), np.zeros(3))
        
        # 1. Get Transform Matrix
        matrix = world_Transform.get_global_matrix()
        
        # 2. Define the 8 corners of a cube localy
        local_corners = None
        
        if shape is not None:
            # 1. Handle Spheres (Look for radius)
            if hasattr(shape, "radius"):
                # Create a box that fully encloses the sphere
                r = float(shape.radius)
                local_corners = np.array([
                    [-r, -r, -r], [r, -r, -r], [-r, r, -r], [r, r, -r],
                    [-r, -r, r],  [r, -r, r],  [-r, r, r],  [r, r, r]
                ])

            # 2. Handle Cube/Cuboids/Prisims (Look for size)
            if hasattr(shape, "size"):
                half_x = shape.size[0] * 0.5
                half_y = shape.size[1] * 0.5
                half_z = shape.size[2] * 0.5
                local_corners = np.array([
                    [-half_x, -half_y, -half_z], [half_x, -half_y, -half_z], 
                    [-half_x, half_y, -half_z], [half_x, half_y, -half_z],
                    [-half_x, -half_y, half_z],  [half_x, -half_y, half_z],  
                    [-half_x, half_y, half_z],  [half_x, half_y, half_z]
                ])

            # 3. Handle Meshes (Anything with corners)
            if hasattr(shape, "convex_hull"):
                local_corners = np.array(shape.convex_hull())

        # C. Fallback: Unit Cube (-0.5 to 0.5)
        if local_corners is None:
            local_corners = np.array([
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], 
                [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],  [0.5, -0.5, 0.5],  
                [-0.5, 0.5, 0.5],  [0.5, 0.5, 0.5]
            ])

        # 2. Transform to World Space
        # Convert to homogeneous coordinates (N, 4)
        ones = np.ones((len(local_corners), 1))
        corners_4d = np.hstack([local_corners, ones]) 
        
        # Apply Matrix (Scale, Rotate, Translate)
        world_corners = (matrix @ corners_4d.T).T[:, :3]

        # 4. Find min/max of transformed corners
        min_p = np.min(world_corners, axis=0) - padding # Small padding
        max_p = np.max(world_corners, axis=0) + padding
        
        return AABB(min_p, max_p)

    @staticmethod
    def union(box_a: 'AABB', box_b: 'AABB') -> 'AABB':
        return AABB(
            np.minimum(box_a.min_point, box_b.min_point),
            np.maximum(box_a.max_point, box_b.max_point)
        )
    
    @property
    def center(self) -> np.ndarray:
        return (self.min_point + self.max_point) * 0.5