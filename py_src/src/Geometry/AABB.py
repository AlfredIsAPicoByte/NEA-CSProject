from __future__ import annotations
from turtle import Shape
import numpy as np
from typing import Any

from src.Data.Transform import Transform
from src.Data.Ray import Ray
from .Operations import *

class AABB:
    """
    Axis-Aligned Bounding Box.
    Used for quick rejection tests before checking complex geometry.
    """
    __slots__ = ['min_point', 'max_point']

    def __init__(self, min_point: np.ndarray, max_point: np.ndarray):
        self.min_point = np.minimum(min_point, max_point)
        self.max_point = np.maximum(max_point, min_point)

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
    def transform_local_bounds_to_world_bounds(transformation_matrix: np.ndarray, local_bounds: np.ndarray) -> np.ndarray:
        ones = np.ones((len(local_bounds), 1))
        corners_4d = np.hstack([local_bounds, ones])

        return (transformation_matrix @ corners_4d.T).T[:, :3]

    @staticmethod
    def combine(box_a: 'AABB', box_b: 'AABB', operation: str) -> 'AABB':
        """
        Combines two AABBs using the specified operation.
        Supported operations: 'union', 'intersect'
        """
        if operation == 'union':
            min_point = np.maximum(box_a.min_point, box_b.min_point)
            max_point = np.maximum(box_a.max_point, box_b.max_point)
        
        elif operation == 'intersect':
            min_point = np.maximum(box_a.min_point, box_b.min_point)
            max_point = np.minimum(box_a.max_point, box_b.max_point)

            # Ensure valid AABB
            if np.any(min_point > max_point):
                return AABB(np.zeros(3), np.zeros(3))  # Empty AABB
        else:
            raise ValueError(f"Unsupported operation '{operation}' for AABB combination.")
        
        return AABB(min_point, max_point)
    
    @property
    def center(self) -> np.ndarray:
        return (self.min_point + self.max_point) * 0.5
    
    @property
    def size(self) -> np.ndarray:
        return self.max_point - self.min_point
    
    @staticmethod
    def empty() -> 'AABB':
        """Returns an empty AABB."""
        return AABB(np.full(3, np.inf), np.full(3, -np.inf))
    
    @staticmethod
    def infinite() -> 'AABB':
        """Returns an infinite AABB."""
        return AABB(np.full(3, -np.inf), np.full(3, np.inf))
    
    @staticmethod
    def unit_cube() -> 'AABB':
        """Returns a unit cube AABB from (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5)."""
        return AABB(np.full(3, -0.5), np.full(3, 0.5))
    
    def __repr__(self) -> str:
        return f"AABB(min={self.min_point}, max={self.max_point})"