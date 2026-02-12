from tests.bench_scenes import get_minimal_scene
from src.Rendering.Intersections import BVHIntersection

s = get_minimal_scene(64,64)
for obj in s.objects:
    obj.update_matrices()

bvh = BVHIntersection()
root = bvh._build_bvh(list(s.objects))
print('Root box min, max:', root.box.min_point, root.box.max_point)
for node in [root.left, root.right]:
    if node is not None and node.box is not None:
        print('Child box min, max:', node.box.min_point, node.box.max_point)
