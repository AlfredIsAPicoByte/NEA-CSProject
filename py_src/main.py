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

def get_test_scene() -> tuple[int, int, Scene]:
    # --- GLOBAL SETTINGS ---
    width, height = 50, 50  # Increased resolution for better viewing

    # --- 1. CAMERA & SCENE SETUP (Enhanced Depth) ---
    # Camera: Moved back to Z = -6.0 for a wider, more composed view
    cam_transform = Transform(np.array([0.0, 1.5, -6.0]), np.array([-10.0, 0.0, 0.0]), np.ones(3)) # Tilted down slightly
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)

    # Background Gradient: Richer sunset/dusk sky for dramatic lighting
    sky_colors = [
        Color.from_hex("#000033"),  # Deep navy at the bottom
        Color.from_hex("#14408A"),  # Dark blue in the middle
        Color.from_hex("#C13584"),  # Magenta/pink at the top (zenith)
    ]
    sky_positions = [0.0, 0.5, 1.0] 
    scene = Scene(name="InterestingScene", camera=cam, background_color=ColorGradient(sky_colors, sky_positions)) 

    # --- 2. LIGHTING (Dramatic Side Light + Ambient) ---
    
    # Primary Key Light (Sharp, slightly yellow, placed high and to the left for side lighting)
    key_light = LightSource(position=np.array([4.0, 5.0, 0.0]), color=Color.from_hex("#FFEDC7"), intensity=2.5, name="Key Light")
    scene.add_light(key_light)
    
    # Soft Fill Light (Simulates general ambient light or bounce light)
    fill_light = LightSource(position=np.array([-5.0, 2.0, -5.0]), color=Color.from_hex("#C7E5FF"), intensity=0.3, name="Fill Light")
    scene.add_light(fill_light)
    
    # --- 3. GEOMETRY AND MATERIALS (Contrast and Realism) ---

    # Main Sphere (Mid-Ground): Highly Reflective Metal
    sphere_shape_1 = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.5, name="MainReflectiveBall")
    mat_metal = Material(color=Color.from_hex("#E0E0E0"), emissive=Color(0, 0, 0), roughness=0.05, glossiness=0.9, metallic=1.0) # Highly reflective metal
    sphere_shape_1.material = mat_metal
    scene.add_object(VObject(shape=sphere_shape_1, name="ReflectiveSphere"))

    # Ground: Darker, slightly reflective floor for showing reflections
    ground = Sphere(center=np.array([0.0, -100.0, 0.0]), radius=100.0, name="FloorPlane")
    mat_floor = Material(color=Color.from_hex("#4B5320"), emissive=Color(0.01, 0.01, 0.01), roughness=0.3, glossiness=0.6, metallic=0.0)
    ground.material = mat_floor
    scene.add_object(VObject(shape=ground, name="GroundObject"))

    # Additional Object 1: Cube (Background/Visual Anchor) - Matte and Rough
    box_shape = Cube(center=np.array([-2.5, 1.0, 4.0]), side_length=2.5, name="BackgroundBox")
    box_shape.transform.rotate(15, np.array([0, 1, 0])) # Simple rotation for visual interest
    mat_matte = Material(color=Color.from_hex("#C27A23"), emissive=Color(0, 0, 0), roughness=0.8, glossiness=0.1, metallic=0.0) # Rough, terracotta-like
    box_shape.material = mat_matte
    scene.add_object(VObject(shape=box_shape, name="MatteBoxObject"))
    
    # Additional Object 2: Small Emissive Sphere (Light Source Helper) - Floating in air
    sphere_shape_2 = Sphere(center=np.array([2.0, 2.5, 2.0]), radius=0.3, name="EmissiveOrb")
    mat_glow = Material(color=Color(0, 0, 0), emissive=Color(0.8, 0.1, 0.8), roughness=0.0, glossiness=0.0, metallic=0.0) # Pure glow
    sphere_shape_2.material = mat_glow
    scene.add_object(VObject(shape=sphere_shape_2, name="EmissiveOrbObject"))

    return width, height, scene

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