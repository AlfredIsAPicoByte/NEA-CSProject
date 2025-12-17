import numpy as np
from PIL import Image
from typing import Optional, Callable, Any
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.Algorithims import Algorithm
from src.Postprocessing import PostProcessingPipeline
from src.Raytracing import Raytracer, RayMarchingIntersection, BasicLambertShading
from src.Camera import VCamera, CameraType
from src.Scene import Scene
from src.Geometry import Sphere, Cube, VObject
from src.Luminance import LightSource, Color, ColorGradient, Material
from src.PrimaryStructures import Transform
from src.Sampling import Sampler, RandomSampler

def render_process(scene: Scene, algorithim: Algorithm, sampler: Sampler, post_processing: Optional[Callable[[Color], Color]] = None):
    # Use named argument for sampler - do not pass camera as a positional value (legacy behavior)
    pixel_colors = algorithim.render(scene, sampler=sampler)        

def render_image_cleanup_and_save(pixel_colors: Scene, out_path="render_out_strat.png", flip_verticaly: bool = False):
    """Render the scene using the given algorithm and return a PIL Image."""
    # Ensure parent directory exists for out_path
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    W, H = scene.camera.width, scene.camera.height
    
    # Convert flat pixel_colors list (row-major, likely bottom-up) to 2D array
    img_array = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, color in enumerate(pixel_colors):
        raw_y = idx // W
        x = idx % W
        # Assuming your camera fills top-to-bottom or bottom-to-top. 
        # If image is upside down, change this to: y = (H - 1) - raw_y
        y = (H - 1) - raw_y if flip_verticaly else raw_y

        if not post_processing is None:
            corrected_color = post_processing(color)
        else:
            corrected_color = color

        # C. Quantize to 0-255
        # We need to extract the raw float components first (r, g, b)
        cr, cg, cb = corrected_color.red, corrected_color.green, corrected_color.blue

        # Explicitly multiply floats by 255 here to be safe and clear
        r = int(np.clip(cr * 255.0, 0, 255))
        g = int(np.clip(cg * 255.0, 0, 255))
        b = int(np.clip(cb * 255.0, 0, 255))

        # Assign
        img_array[y, x, :] = [r, g, b]
    
    im = Image.fromarray(img_array, mode="RGB")
    print(f"Rendered image: {W}x{H}")

    # Save output
    im.save(out_path)
    print(f"Saved render to {out_path}")

def get_gradient_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.5, -6.0]), np.array([0, 0.2, 0]), np.ones(3))
    cam = VCamera(cam_transform, fov=450.0, near=0.1, far=86.0, width=width, height=height, camType=CameraType.PERSPECTIVE)

    # Background Gradient: Richer sunset/dusk sky for dramatic lighting
    sky_colors = [
        Color.from_hex("#000033"),  # Deep navy at the bottom
        Color.from_hex("#14408A"),  # Dark blue in the middle
        Color.from_hex("#C13584"),  # Magenta/pink at the top (zenith)
    ]
    sky_positions = [0.0, 0.5, 1.0] 

    # Primary Key Light (Sharp, slightly yellow, placed high and to the left for side lighting)
    key_light = LightSource(position=np.array([4.0, 5.0, 0.0]), color=Color.from_hex("#FFEDC7"), intensity=15.0, radius=0.5, name="Key Light")
    
    # Soft Fill Light (Simulates general ambient light or bounce light)
    fill_light = LightSource(position=np.array([-5.0, 2.0, -5.0]), color=Color.from_hex("#C7E5FF"), intensity=3.0, radius=4, name="Fill Light")

    # Main Sphere (Mid-Ground): Highly Reflective Metal
    sphere_shape_1 = Sphere(center=np.array([13.0, 5.0, 22.0]), radius=0.5, name="MainReflectiveBall")
    mat_metal = Material(color=Color.from_hex("#E0E0E0"), emissive=Color(0, 0, 0), roughness=0.05, glossiness=0.9, metallic=1.0) # Highly reflective metal
    sphere_shape_1.material = mat_metal

    # Ground: Darker, slightly reflective floor for showing reflections
    ground = Sphere(center=np.array([0.0, -100.0, 0.0]), radius=100.0, name="FloorPlane")
    mat_floor = Material(color=Color.from_hex("#4B5320"), emissive=Color(0.01, 0.01, 0.01), roughness=0.3, glossiness=0.6, metallic=0.0)
    ground.material = mat_floor

    # Additional Object 1: Cube (Background/Visual Anchor) - Matte and Rough
    box_shape = Cube(center=np.array([2.5, 3.0, 4.0]), side_length=2.5, name="BackgroundBox")
    box_shape.transform.rotate(15, np.array([0, 1, 0])) # Simple rotation for visual interest
    mat_matte = Material(color=Color.from_hex("#C27A23"), emissive=Color(0, 0, 0), roughness=0.8, glossiness=0.1, metallic=0.0) # Rough, terracotta-like
    box_shape.material = mat_matte
    
    # Additional Object 2: Small Emissive Sphere (Light Source Helper) - Floating in air
    sphere_shape_2 = Sphere(center=np.array([-2.0, 2.5, 2.0]), radius=0.3, name="EmissiveOrb")
    mat_glow = Material(color=Color(0, 0, 0), emissive=Color(0.5, 0.3, 0.1), roughness=0.0, glossiness=0.0, metallic=0.0) # Pure glow
    sphere_shape_2.material = mat_glow

    cam.transform.look_at(sphere_shape_1.transform.position)

    scene = Scene(name="gradient_scene", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    scene.add_light(key_light)
    scene.add_light(fill_light)
    scene.add_object(VObject(shape=sphere_shape_1, name="ReflectiveSphere"))
    scene.add_object(VObject(shape=ground, name="GroundObject"))
    scene.add_object(VObject(shape=box_shape, name="MatteBoxObject"))
    scene.add_object(VObject(shape=sphere_shape_2, name="EmissiveOrbObject"))

    return scene

# New: minimal scene - single sphere on ground with a single directional light
def get_minimal_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    scene = Scene(name="minimal_scene", camera=cam, background_color=Color.from_hex("#a0c8ff"))

    # Sphere at origin
    sphere_shape = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5, name="BallMin")
    mat = Material(color=Color.from_hex("#44A1FF"), emissive=Color(0, 0, 0), roughness=0.5, glossiness=0.1, metallic=0.0)
    sphere_shape.material = mat
    scene.add_object(VObject(shape=sphere_shape, name="SphereMin"))

    # Ground
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="GroundMin")
    matg = Material(color=Color.from_hex("#808080"), emissive=Color(0, 0, 0), roughness=1.0, glossiness=0.0, metallic=0.0)
    ground.material = matg
    scene.add_object(VObject(shape=ground, name="GroundMin"))

    # Single light
    light = LightSource(position=np.array([2.0, 3.0, -1.0]), color=Color.from_hex("#FFFFFF"), intensity=15.0, name="SunMin")
    scene.add_light(light)

    return scene

# New: Emissive scene - emission as lighting (emissive sphere and simple fill light)
def get_emissive_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -3.5]), np.zeros(3), np.ones(3))
    cam = VCamera(cam_transform, fov=75.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    scene = Scene(name="emissive_scene", camera=cam, background_color=Color.from_hex("#402050"))

    # Emissive sphere
    emissive = Sphere(center=np.array([0.8, 1.0, 0.0]), radius=0.3, name="EmissiveOrb")
    mat_glow = Material(color=Color(0,0,0), emissive=Color(0.3, 0.3, 0.9), roughness=0.0, glossiness=0.0, metallic=0.0, emissive_intensity=1.5)
    emissive.material = mat_glow
    scene.add_object(VObject(shape=emissive, name="GlowingSphere"))

    # Reflective sphere
    mirror = Sphere(center=np.array([-0.5, 0.5, 0.0]), radius=0.5, name="Mirror")
    mat_reflect = Material(color=Color.from_hex("#EDEDED"), emissive=Color(0, 0, 0), roughness=0.05, glossiness=0.9, metallic=0.9)
    mirror.material = mat_reflect
    scene.add_object(VObject(shape=mirror, name="MirrorSphere"))

    # Ground
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="GroundEmissive")
    matg = Material(color=Color.from_hex("#202020"), emissive=Color(0,0,0), roughness=0.8, glossiness=0.0, metallic=0.0)
    ground.material = matg
    scene.add_object(VObject(shape=ground, name="GroundEmissive"))

    # Small ambient fill light
    fill = LightSource(position=np.array([-4.0, 2.0, -3.0]), color=Color.from_hex("#AAAACC"), intensity=25.0, radius=10.0, name="FillEmiss")
    scene.add_light(fill)

    return scene

# New: studio / many lights scene to show shadows and complex shading
def get_lit_studio_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -4.0]), np.array([-0.15, 0.0, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=50.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    scene = Scene(name="lit_studio", camera=cam, background_color=Color.from_hex("#dfe7ff"))

    # Objects: two spheres and box as background
    s1 = Sphere(center=np.array([-0.6, 0.4, 0.5]), radius=0.4, name="StudioBallA")
    mat1 = Material(color=Color.from_hex("#FFB86B"), emissive=Color(0,0,0), roughness=0.3, glossiness=0.2, metallic=0.0)
    s1.material = mat1
    scene.add_object(VObject(shape=s1, name="StudioBallA"))

    s2 = Sphere(center=np.array([0.8, 0.45, 0.2]), radius=0.45, name="StudioBallB")
    mat2 = Material(color=Color.from_hex("#6B9BFF"), emissive=Color(0,0,0), roughness=0.15, glossiness=0.6, metallic=0.2)
    s2.material = mat2
    scene.add_object(VObject(shape=s2, name="StudioBallB"))

    # Background box
    box_shape = Cube(center=np.array([0.0, 0.5, 3.0]), side_length=4.0, name="StudioBack")
    mat_box = Material(color=Color.from_hex("#C2C6C9"), emissive=Color(0,0,0), roughness=1.0, glossiness=0.0, metallic=0.0)
    box_shape.material = mat_box
    scene.add_object(VObject(shape=box_shape, name="StudioBox"))

    # Lights
    key = LightSource(position=np.array([2.5, 3.5, -1.0]), color=Color.from_hex("#EEE0BA"), intensity=150.0, radius=1.5, name="StudioKey")
    key.radius = 0.3  # area light radius (for soft shadows)
    scene.add_light(key)
    rim = LightSource(position=np.array([-3.0, 2.0, 1.0]), color=Color.from_hex("#DC97C5"), intensity=20.0, radius=0.75, name="StudioRim")
    rim.radius = 0.2
    scene.add_light(rim)
    fill = LightSource(position=np.array([0.0, -2.5, -2.0]), color=Color.from_hex("#C7DBD8"), intensity=15.0, radius=2, name="StudioFill")
    scene.add_light(fill)

    return scene

def get_rgb_room_with_objects_scene(width: int = 126, height: int = 126) -> Scene:
    pass

# Add orchestrator to render and save a set of scenes
if __name__ == "__main__":
    # where to save outputs (repo root / benchmark / simple_scene)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "simple_scene")
    os.makedirs(OUT_DIR, exist_ok=True)

    # list of scenes to render
    all_scenes = [
        get_minimal_scene(),
        get_gradient_scene(),
        get_emissive_scene(),
        get_lit_studio_scene(),
    ]

    rpp = 1  # rays per pixel
    spp = 1  # samples per pixel
    intersector = RayMarchingIntersection(
        max_distance=1000,
        max_steps=256
    )
    shader = BasicLambertShading(
        ambient_enabled=True, ambient_color=Color.from_hex("#0B0B0C"), ambient_intensity=0.1,
        enable_shadows=True, shadow_samples=8, shadow_bias=1e-3
    )
    raytracer = Raytracer(rays_per_pixel=rpp, intersection_strategy=intersector, shading_strategy=shader)
    sampler = RandomSampler(samples_per_pixel=spp)

    for scene in all_scenes:
        sanitized_name = scene.name.replace(" ", "_").lower()
        out_path = os.path.join(OUT_DIR, f"{sanitized_name}.png")
        print(f"Rendering '{scene.name}' -> {out_path} ({scene.camera.width}x{scene.camera.height})")
        try:
            render_and_save(scene, raytracer, sampler, out_path=out_path)
        except Exception as e:
            print(f"Failed to render '{scene.name}': {e}")