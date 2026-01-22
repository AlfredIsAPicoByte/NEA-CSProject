import numpy as np
from enum import Enum
from typing import List, Optional, Tuple

from .AABB import AABB
from .Primitive import Primitive

class BVHSplitMode(Enum):
    LONGEST_AXIS = "longest_axis"
    BALANCED = "balanced"

class BVHNode:
    def __init__(self, objects: Optional[List[Primitive]] = None):
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None
        self.box: Optional[AABB] = None
        self.objects: Optional[List[Primitive]] = objects

def build_bvh_tree(
        objects: List[Primitive],
        split_mode: BVHSplitMode = BVHSplitMode.LONGEST_AXIS,
        enable_limit: bool = False,
        max_depth: int = 0,
        build_depth: int = 0,
    ) -> BVHNode:
    """
    Public entry point: Calculates AABBs once and starts recursion.
    """
    # 1. Pre-calculate AABBs for all objects once. 
    # This prevents O(N * Depth) re-calculations.
    item_cache = []
    for obj in objects:
        # Use the existing static method to generate the box
        box = AABB.from_transform_shape(
            getattr(obj, 'world_transform', obj.transform), 
            obj.shape
        )
        item_cache.append((obj, box))

    return _build_bvh_recursive(
        item_cache, 
        split_mode, 
        enable_limit, 
        max_depth, 
        build_depth
    )

def _build_bvh_recursive(
        items: List[Tuple[Primitive, AABB]], 
        split_mode: BVHSplitMode,
        enable_limit: bool,
        max_depth: int,
        build_depth: int
    ) -> BVHNode:
    """
    Internal recursive worker that uses cached (Object, Box) tuples.
    """
    node = BVHNode()

    # 1. Handle Empty Case
    if not items:
        node.box = AABB(np.zeros(3), np.zeros(3))
        node.objects = []
        return node

    # 2. Calculate Union Box for this specific node
    # Initialize with the first box's bounds
    node_min = items[0][1].min_point.copy()
    node_max = items[0][1].max_point.copy()
    
    for _, box in items:
        node_min = np.minimum(node_min, box.min_point)
        node_max = np.maximum(node_max, box.max_point)
    
    node.box = AABB(node_min, node_max)

    # 3. Leaf Condition
    # Stop if we hit depth limit OR if we have few objects
    is_max_depth = enable_limit and build_depth >= max_depth
    if is_max_depth or len(items) <= 2:
        node.objects = [obj for obj, box in items] # Unpack tuples back to objects
        return node

    # 4. Split Strategy
    
    # Helper lambda to get center of a cached box
    # box is at index 1 of the tuple
    def get_center(item, axis_idx):
        return (item[1].min_point[axis_idx] + item[1].max_point[axis_idx]) * 0.5

    # Determine sorting based on mode
    if split_mode == BVHSplitMode.LONGEST_AXIS:
        # Find longest axis of the *Node's* box
        extent = node_max - node_min
        axis = np.argmax(extent) # 0=x, 1=y, 2=z
        
        # Sort cached items by center along that axis
        items.sort(key=lambda item: get_center(item, axis))

    elif split_mode == BVHSplitMode.BALANCED:
        # Sort by the average of all axes (scalar sort)
        # This keeps spatially coherent clusters somewhat together
        items.sort(key=lambda item: np.mean(item[1].min_point + item[1].max_point))

    # 5. Partition
    mid = len(items) // 2

    # SAFETY CHECK: If geometry overlaps perfectly, sorting might not separate them.
    # If mid is 0 or len(items), we would recurse infinitely on the same list.
    if mid == 0 or mid == len(items):
        node.objects = [obj for obj, box in items]
        return node

    # Recurse
    # Note: We pass the cached list down, no re-calculation needed
    node.left = _build_bvh_recursive(items[:mid], split_mode, enable_limit, max_depth, build_depth + 1)
    node.right = _build_bvh_recursive(items[mid:], split_mode, enable_limit, max_depth, build_depth + 1)

    return node