from __future__ import annotations
import numpy as np
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

from .AABB import AABB

if TYPE_CHECKING:
    from src.Data.Scene import SceneNode

class BVHSplitMode(Enum):
    LONGEST_AXIS = "longest_axis"
    BALANCED = "balanced"

class BVHNode:
    """
    A node in the Bounding Volume Hierarchy (BVH) tree.
    Each node may have left and right children, an AABB bounding box,
    and a list of SceneNode objects if it's a leaf.
    """
    def __init__(self, objects: Optional[List["SceneNode"]] = None):
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None
        self.box: Optional[AABB] = None
        self.objects: Optional[List["SceneNode"]] = objects

def build_bvh_tree(
        objects: List["SceneNode"],
        split_mode: BVHSplitMode = BVHSplitMode.LONGEST_AXIS,
        enable_limit: bool = False,
        max_depth: int = 0,
        build_depth: int = 0,
    ) -> BVHNode:
    """
    Public entry point: Calculates AABBs once and starts recursion.
    """
    
    # Pre-calculate AABBs for all objects once. 
    # This prevents O(N * Depth) re-calculations.
    item_cache = []
    for obj in objects:
        if obj is None: # Empty nodes are discarded
            continue
        
        # Get local bounds for objects within BoundingSceneNode
        box = obj.get_bounds() # Handles None internally
        item_cache.append((obj, box))

    return _build_bvh_recursive(
        item_cache, 
        split_mode, 
        enable_limit, 
        max_depth, 
        build_depth
    )

def _build_bvh_recursive(
        items: List[Tuple["SceneNode", AABB]], 
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
        if box is None: # Handle missing boxes, by giving large bounds to avoid issues with culled nodes exept empty nodes
            box = AABB(np.full(3, np.inf), np.full(3, -np.inf))
            continue

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
    def get_center(item, axis_idx):
        return (item[1].center[axis_idx] if item[1] is not None else 0.0)

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
        items.sort(key=lambda item: np.mean(item[1].center) if item[1] is not None else 0.0)

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