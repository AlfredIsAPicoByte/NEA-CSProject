import numpy as np
from PIL import Image
from typing import Optional, Callable, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.Algorithims import Algorithm
from src.Postprocessing import PostProcessingPipeline
from src.Raytracing import Raytracer
from src.Camera import VCamera, CameraType
from src.Scene import Scene
from src.Geometry import Sphere, Cube, VObject
from src.Luminance import LightSource, Color, ColorGradient, Material
from src.PrimaryStructures import Transform
from src.Sampling import Sampler, RandomSampler

def render_process(scene: Scene, algorithim: Algorithm, sampler: Sampler) -> List[Color]:
    """
    Renders the scene and returns a NumPy float32 array (H, W, 3) 
    ready for post-processing.
    """
    # 1. Get raw list of Color objects
    pixel_colors: List[Color] = algorithim.render(scene)
    
    W, H = scene.camera.width, scene.camera.height
    
    # 2. Convert to NumPy Float Array (for efficient Post-Processing)
    # We initialize with float32 to handle HDR values (> 1.0)
    raw_buffer = np.zeros((H, W, 3), dtype=np.float32)
    
    for idx, color in enumerate(pixel_colors):
        raw_y = idx // W
        x = idx % W
        y = raw_y # (y = (H - 1) - raw_y) to flip the image
        
        r, g, b = color.red, color.green, color.blue
        
        raw_buffer[y, x] = [r, g, b]
        
    return raw_buffer

def save_image(img_data: np.ndarray, out_path="render_out.png"):
    """
    Saves the image.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Quantize to 0-255 uint8
    final_pixels = np.clip(img_data * 255.0, 0, 255).astype(np.uint8)
    
    im = Image.fromarray(final_pixels, mode="RGB")
    im.save(out_path)
    print(f"  Saved to {out_path}")

# --- SCENE DEFINITIONS ---

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
    # 1. Camera Setup (Wide FOV to see the whole room)
    # Positioned slightly back to view the open box
    cam_transform = Transform(
        position=np.array([0.0, 2.5, -7.5]), 
        rotation=np.array([0.0, 0.0, 0.0]), 
        scale=np.ones(3)
    )
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    
    # 2. Materials
    # Walls (Matte)
    mat_white = Material(color=Color.from_hex("#E0E0E0"), emissive=Color(), roughness=1.0, glossiness=0.0, metallic=0.0)
    mat_red   = Material(color=Color.from_hex("#B03030"), emissive=Color(), roughness=1.0, glossiness=0.0, metallic=0.0)
    mat_green = Material(color=Color.from_hex("#30B030"), emissive=Color(), roughness=1.0, glossiness=0.0, metallic=0.0)
    
    # Objects (Shiny/Transmissive)
    mat_mirror = Material(color=Color.from_hex("#FFFFFF"), emissive=Color(), roughness=0.02, glossiness=0.98, metallic=1.0)
    mat_glass  = Material(color=Color.from_hex("#FFFFFF"), emissive=Color(), roughness=0.0, glossiness=1.0, metallic=0.0, ior=1.5)
    # Note: Enable refraction flag on material if your engine supports it
    mat_glass.is_transparent = True
    mat_glass.can_refract = True

    # 3. Room Geometry (The Box)
    room_objects = []
    
    # Floor
    floor = Cube(center=np.array([0.0, -0.5, 0.0]), side_length=10.0, name="Floor")
    floor.transform.scale = np.array([2.0, 0.1, 2.0]) # Flatten into plane
    floor.material = mat_white
    room_objects.append(floor)
    
    # Ceiling
    ceiling = Cube(center=np.array([0.0, 5.5, 0.0]), side_length=10.0, name="Ceiling")
    ceiling.transform.scale = np.array([2.0, 0.1, 2.0])
    ceiling.material = mat_white
    room_objects.append(ceiling)

    # Back Wall
    back = Cube(center=np.array([0.0, 2.5, 5.5]), side_length=10.0, name="BackWall")
    back.transform.scale = np.array([2.0, 2.0, 0.1])
    back.material = mat_white
    room_objects.append(back)
    
    # Left Wall (Red)
    left = Cube(center=np.array([-5.5, 2.5, 0.0]), side_length=10.0, name="LeftWall")
    left.transform.scale = np.array([0.1, 2.0, 2.0])
    left.material = mat_red
    room_objects.append(left)

    # Right Wall (Green)
    right = Cube(center=np.array([5.5, 2.5, 0.0]), side_length=10.0, name="RightWall")
    right.transform.scale = np.array([0.1, 2.0, 2.0])
    right.material = mat_green
    room_objects.append(right)
    
    # 4. Content Objects
    
    # Tall Box (Rotated)
    tall_box = Cube(center=np.array([-2.0, 1.5, 2.0]), side_length=3.0, name="TallBox")
    tall_box.transform.scale = np.array([0.6, 1.0, 0.6]) # Make it a pillar
    tall_box.transform.rotate(20.0, np.array([0.0, 1.0, 0.0])) # Rotate Y
    tall_box.material = mat_white # Standard white box for diffusal
    room_objects.append(tall_box)
    
    # Sphere (Mirror)
    mirror_sphere = Sphere(center=np.array([2.0, 1.25, 1.0]), radius=1.25, name="MirrorBall")
    mirror_sphere.material = mat_mirror
    room_objects.append(mirror_sphere)
    
    # Small Cube (Glass/Crystal in front)
    glass_cube = Cube(center=np.array([0.0, 0.75, -2.0]), side_length=1.5, name="GlassCube")
    glass_cube.transform.rotate(-15.0, np.array([0.0, 1.0, 0.0]))
    glass_cube.material = mat_glass
    room_objects.append(glass_cube)

    # 5. Lighting
    # A single strong area light on the ceiling (simulating the Cornell Box light patch)
    ceiling_light = LightSource(
        position=np.array([0.0, 4.8, 0.0]), 
        color=Color.from_hex("#FFF0E0"), 
        intensity=25.0, 
        radius=1.5, 
        name="CeilingLight"
    )

    # Assemble Scene
    scene = Scene(name="rgb_cornell_box", camera=cam, background_color=Color(0,0,0)) # Pitch black void outside
    
    scene.add_light(ceiling_light)
    for obj in room_objects:
        scene.add_object(VObject(shape=obj, name=obj.name))

    return scene

# Add orchestrator to render and save a set of scenes
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "simple_scene")
    os.makedirs(OUT_DIR, exist_ok=True)

    all_scenes = [
        get_minimal_scene(100, 100),
        get_gradient_scene(100, 100),
        get_emissive_scene(128, 128),
        get_lit_studio_scene(144, 108),
        get_rgb_room_with_objects_scene(108, 144),
    ]

    rpp = 1
    spp = 1
    raytracer = Raytracer(rays_per_pixel=rpp)
    sampler = RandomSampler(samples_per_pixel=spp)

    for scene in all_scenes:
        sanitized_name = scene.name.replace(" ", "_").lower()
        out_path = os.path.join(OUT_DIR, f"{sanitized_name}.png")
        print(f"Rendering '{scene.name}' -> {out_path} ({scene.camera.width}x{scene.camera.height})")
        
        try:
            # 1. Render to Float Array
            raw_img_data = render_process(scene, raytracer, sampler)
            # 2. Post-Process and Save
            save_image(raw_img_data, out_path=out_path)
            
        except Exception as e:
            print(f"Failed to render '{scene.name}': {e}")
            import traceback
            traceback.print_exc()