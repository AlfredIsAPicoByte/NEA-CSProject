from main import save_image
import numpy as np, os
arr = np.zeros((10,10,3), dtype=np.float32)
for y in range(10):
    for x in range(10):
        arr[y,x]=[x/9.0,y/9.0,(x+y)/18.0]
os.makedirs("benchmark/simple_scene", exist_ok=True)
save_image(arr, "benchmark/simple_scene/test_out.png")
print("Wrote benchmark/simple_scene/test_out.png")