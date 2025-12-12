import numpy as np
from PIL import Image
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.Algorithims import Algorithm
from src.Raytracing import Raytracer
from src.Camera import VCamera, CameraType
from src.Scene import Scene
from src.Geometry import Sphere, Cube, VObject
from src.Luminance import LightSource, Color, ColorGradient, Material
from src.PrimaryStructures import Transform
from src.Sampling import Sampler, RandomSampler

def render_and_save(scene, algorithim: Algorithm, sampler: Sampler, out_path="render_out_strat.png"):
    """Render the scene using the given algorithm and return a PIL Image."""
    pixel_colors = algorithim.render(scene, scene.camera, sampler=sampler)
    
    W, H = scene.camera.width, scene.camera.height
    
    # Convert flat pixel_colors list (row-major, likely bottom-up) to 2D array
    img_array = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, color in enumerate(pixel_colors):
        
        # 1. Calculate the raw row index (0 is bottom, H-1 is top)
        raw_y = idx // W
        x = idx % W
        
        # 2. <<< FIX APPLIED HERE: FLIP THE Y-COORDINATE >>>
        # Maps the first rendered row (raw_y=0) to the last row of the array (H-1)
        # Maps the last rendered row (raw_y=H-1) to the first row of the array (0)
        y = H - 1 - raw_y
        
        # Extract RGB from Color object (rest of the logic is fine)
        if hasattr(color, "rgba"):
            rgb = color.rgba[:3]
        elif hasattr(color, "to_array"):
            rgb = color.to_array()[:3]
        else:
            rgb = [0.0, 0.0, 0.0]
        
        # Clamp to [0, 1] and convert to [0, 255]
        rgb = np.clip(np.asarray(rgb, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
        
        # Assign the color to the correctly flipped position
        img_array[y, x, :] = rgb
    
    im = Image.fromarray(img_array, mode="RGB")
    print(f"Rendered image: {W}x{H}, {len(pixel_colors)} pixel colors")
    
    # Convert to numpy to inspect pixel range (works for PIL Image or array)
    try:
        arr = np.array(im)
        print(f"Render pixel range: min={arr.min()}, max={arr.max()}, "
              f"shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print("Could not convert render to numpy array:", e)
    
    # Save output
    im.save(out_path)
    print(f"Saved render to {out_path}")

if __name__ == "__main__":    # For a quick test, use small resolution; change to 320 X 180 for final
    width, height = 64, 64

    # Camera: positioned at z = -3 looking toward +z (default Transform rotation => forward = +z)
    cam_transform = Transform(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)

    # Scene
    scene = Scene(name="SimpleScene", camera=cam, background_color=ColorGradient([Color.from_hex("#87CEEB"), Color.from_hex("#6678DF")], [0, 1]))  # Light blue background

    # Geometry: Sphere at origin
    sphere_shape = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5, name="Ball")
    # Material (optional) - Luminance.Material expects Color, roughness, etc.
    mat = Material(color=Color.from_hex("#4E70E0"), emissive=Color(0.5, 0.5, 1), roughness=0.5, glossiness=0.2, metallic=0.0)
    sphere_shape.material = mat

    # Ground approximated with a large sphere (simple trick)
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="Ground")
    mat = Material(color=Color.from_hex("#8FBF7F"), emissive=Color.from_hex("#8FBF7F"), roughness=1.0, glossiness=0.0, metallic=0.0)
    ground.material = mat

    # Light source
    light = LightSource(position=np.array([2.0, 3.0, -1.0]), color=Color.from_hex("#FFFFFF"), intensity=1.5, name="Sun")
    scene.add_light(light)

    # Additional object: rotated cube
    box = Cube(center=np.array([-1, 4, 5]), side_length=2, name="Box")
    box.Rotate(1, [0, 1, 0])
    mat = Material(color=Color.from_hex("#A38A5A"), emissive=Color(0, 0, 0), roughness=0, glossiness=0.0, metallic=0.5)
    box.material = mat

    vobj = VObject(shape=sphere_shape, name="SphereObject")
    ground_obj = VObject(shape=ground, name="GroundObject")
    box_obj = VObject(shape=box, name="BoxObject")

    scene.add_object(vobj)
    scene.add_object(ground_obj)
    scene.add_object(box_obj)

    rpp = 1  # Rays per pixel
    spp = 1  # Samples per pixel for sampling
    raytracer = Raytracer(rays_per_pixel=rpp)
    sampler = RandomSampler(samples_per_pixel=spp)
    render_and_save(scene, raytracer, sampler, out_path="render_out.png")