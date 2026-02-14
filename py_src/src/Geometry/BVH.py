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
        # Convenience: if caller passed a list to the constructor, build
        # a BVH from those items so `BVHNode(shapes)` behaves like tests expect.
        if objects is not None:
            try:
                # Lazy import to avoid circular import at module load
                from src.Data.Scene import SceneNode
            except Exception:
                SceneNode = None

            # Wrap raw shape/context objects into SceneNode when needed
            prepared = []
            for o in objects:
                if SceneNode is not None and isinstance(o, SceneNode):
                    prepared.append(o)
                else:
                    # Create a SceneNode wrapper for plain shape/context
                    from src.Data.Scene import SceneNode as _SN
                    prepared.append(_SN(context=o))

            # Build the BVH tree from the prepared SceneNode list and copy
            # the resulting root node into this instance.
            root = build_bvh_tree(prepared)
            # Copy structure
            self.left = root.left
            self.right = root.right
            self.box = root.box
            self.objects = root.objects

    @property
    def is_leaf(self) -> bool:
        return self.objects is not None
    
    @property
    def is_empty(self) -> bool:
        return self.objects is not None and len(self.objects) == 0
    
    @property
    def is_internal(self) -> bool:
        return self.objects is None and (self.left is not None or self.right is not None)
    
    @property
    def is_valid(self) -> bool:
        return self.box is not None and (self.is_leaf or self.is_internal)
    
    @property
    def leaf_count(self) -> int:
        if self.is_leaf:
            return len(self.objects)
        count = 0
        if self.left is not None:
            count += self.left.leaf_count
        if self.right is not None:
            count += self.right.leaf_count
        return count
    
    @property
    def node_count(self) -> int:
        count = 1  # Count self
        if self.left is not None:
            count += self.left.node_count
        if self.right is not None:
            count += self.right.node_count
        return count
    
    @property
    def depth(self) -> int:
        if self.is_leaf:
            return 1
        left_depth = self.left.depth if self.left is not None else 0
        right_depth = self.right.depth if self.right is not None else 0
        return 1 + max(left_depth, right_depth)
    
    @property
    def is_balanced(self) -> bool:
        if self.is_leaf:
            return True
        left_depth = self.left.depth if self.left is not None else 0
        right_depth = self.right.depth if self.right is not None else 0
        return abs(left_depth - right_depth) <= 1 and (self.left.is_balanced if self.left is not None else True) and (self.right.is_balanced if self.right is not None else True)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"BVHNode(Leaf, Box={self.box}, Objects={len(self.objects)})"
        else:
            return f"BVHNode(Internal, Box={self.box})"
        
    def intersect(self, origin: np.ndarray, direction: np.ndarray):
        """
        Traverse BVH: if a leaf's objects hit, return first hit t (float) else None.
        Uses shape.ray_intersect if available (after transforming ray into local space).
        Optimized with:
        - Early termination when best_t is updated
        - Dynamic MAX_DISTANCE based on current best hit
        - Ordered child traversal (visit nearer child first for better pruning)
        """
        from src.Data.Ray import Ray
        root = self
        if root is None:
            return None

        query_ray = Ray(np.array(origin, dtype=float), np.array(direction, dtype=float))
        best_t = None

        def get_distance_to_box_center(node):
            """Helper to determine which child to visit first."""
            if node is None or node.box is None:
                return float('inf')
            center = node.box.center
            ray_origin = query_ray.origin
            return np.linalg.norm(center - ray_origin)

        def recurse(node):
            nonlocal best_t
            if node is None or node.box is None:
                return
            
            # Quick aabb-ray test with early pruning
            t = node.box.intersect(query_ray)
            if t is None or (best_t is not None and t >= best_t):
                return
            
            # Leaf node: test all objects
            if node.left is None and node.right is None:
                for obj in (node.objects or []):
                    shape = getattr(obj, "context", obj)
                    
                    # Transform ray into object's local space if obj is SceneNode
                    inv_mat = getattr(obj, "get_local_matrix", lambda: None)()
                    if inv_mat is not None:
                        from src.Data.Transform import Transform as _T
                        inv_t = _T.from_matrix(inv_mat)
                        local_ray = inv_t.apply(query_ray)
                    else:
                        local_ray = query_ray

                    # Prefer ray_intersect
                    if hasattr(shape, "ray_intersect"):
                        hits = shape.ray_intersect(local_ray)
                        if hits:
                            for ht in hits:
                                if ht > 0 and (best_t is None or ht < best_t):
                                    best_t = ht
                    else:
                        # Fallback: sphere-trace using get_distance
                        if hasattr(shape, "get_distance"):
                            t0 = 0.0
                            MAX_STEPS = 128
                            EPS = 1e-4
                            # Dynamic max distance: limit search to nearest hit found so far
                            max_distance = best_t if best_t is not None else 1e30
                            
                            for _ in range(MAX_STEPS):
                                # Early exit if we've exceeded best found hit
                                if t0 >= max_distance:
                                    break
                                
                                p = local_ray.origin + local_ray.direction * t0
                                d = shape.get_distance(p)
                                
                                if d < EPS:
                                    if best_t is None or t0 < best_t:
                                        best_t = t0
                                        max_distance = best_t  # Update limit for next objects
                                    break
                                
                                t0 += d
                return
            
            # Internal node: recurse on children with ordered traversal
            # Visit nearer child first for better pruning of far branch
            left_dist = get_distance_to_box_center(node.left)
            right_dist = get_distance_to_box_center(node.right)
            
            if left_dist < right_dist:
                recurse(node.left)
                recurse(node.right)
            else:
                recurse(node.right)
                recurse(node.left)

        recurse(root)
        return best_t

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
        box = obj.get_transformed_aabb()
        if box is None:
            continue
        
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
        node.box = AABB.empty()
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