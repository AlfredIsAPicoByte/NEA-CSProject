from typing import List, Optional

from src.Data.AABB import AABB
from .Primitive import Primitive

class BVHNode:
    def __init__(self, objects: List[Primitive]):
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None
        self.box: Optional[AABB] = None
        self.objects: List[Primitive] = objects