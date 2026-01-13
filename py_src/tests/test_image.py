import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

sys.path.insert(0, current_dir)

from main import save_image
import numpy as np, os

def test_image_creation():
    width, height = 256, 256

    # Create a simple gradient image for testing
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = [x % 256, y % 256, (x + y) % 256]
    
    output_path = os.path.join(os.path.dirname(__file__), "test_output.png")
    save_image(image, output_path)
    print(f"Test image saved to {output_path}")

if __name__ == '__main__':
    sys.exit(pytest.main(["-v", __file__]))