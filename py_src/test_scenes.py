import numpy as np
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.PrimaryStructures import Transform
from src.Geometry import Sphere, Cube, VObject
from src.Scene import Scene
from src.Camera import VCamera, CameraType
from src.Luminance import LightSource, Color, ColorGradient, PBRMaterial
from src.Refractions import REFRACTIVE_INDICES

def get_gradient_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.5, -4.0]), np.array([0, 0.2, 0]), np.ones(3))
    cam = VCamera(cam_transform, fov=70.0, near=0.1, far=86.0, width=width, height=height, camType=CameraType.PERSPECTIVE)

    # Background Gradient
    sky_colors = [
        Color.from_hex("#2D2515"),
        Color.from_hex("#42424E"),
        Color.from_hex("#5B6791"),
        Color.from_hex("#87BFC6"),
    ]
    sky_positions = [0.0, 0.4, 0.45, 1.0] 

    # Primary Key Light (Sharp, slightly yellow, placed high and to the left for side lighting)
    key_light = LightSource(position=np.array([4.0, 5.0, 0.0]), color=Color.from_hex("#FFEDC7"), intensity=15.0, radius=0.5, name="Key Light")
    
    # Soft Fill Light (Simulates general ambient light or bounce light)
    fill_light = LightSource(position=np.array([-5.0, 2.0, -5.0]), color=Color.from_hex("#C7E5FF"), intensity=3.0, radius=4, name="Fill Light")

    # Main Sphere (Mid-Ground): Highly Reflective Metal
    sphere_shape_1 = Sphere(center=np.array([0.0, 2.25, 5.0]), radius=2.5, name="MainReflectiveBall")
    mat_metal = PBRMaterial.create_specular(Color.from_hex("#47505C"), 0.2, 0.9, 1.0, 1.0) # Highly reflective semi metal
    sphere_shape_1.PBRMaterial = mat_metal

    # Additional Object 1: Cube (Background/Visual Anchor) - Matte and Rough
    box_shape = Cube(center=np.array([2.0, 3.0, 4.0]), side_length=2.5, name="BackgroundBox")
    box_shape.rotate(np.radians(15), np.array([0, 1, 0])) # Simple rotation for visual interest
    mat_matte = PBRMaterial.create_diffuse(Color.from_hex("#C27A23"), roughness=0.8) # Rough, terracotta-like
    box_shape.PBRMaterial = mat_matte
    
    # Additional Object 2: Small Emissive Sphere (Light Source Helper) - Floating in air
    sphere_shape_2 = Sphere(center=np.array([-0.5, 2.5, 1.5]), radius=1, name="EmissiveOrb")
    mat_glow = PBRMaterial.create_emissive(Color.from_hex("#EE1717"), 2) # Pure glow
    sphere_shape_2.PBRMaterial = mat_glow

    cam.transform.look_at(sphere_shape_1.transform.position, [0, 1, 0])

    scene = Scene(name="gradient_scene", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    scene.add_light(key_light)
    scene.add_light(fill_light)
    scene.add_object(VObject(shape=sphere_shape_1, name="ReflectiveSphere"))
    scene.add_object(VObject(shape=box_shape, name="MatteBoxObject"))
    scene.add_object(VObject(shape=sphere_shape_2, name="EmissiveOrbObject"))

    return scene

def get_minimal_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    scene = Scene(name="minimal_scene", camera=cam, background_color=Color.from_hex("#403A43"))

    # Sphere at origin
    sphere_shape = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5, name="BallMin")
    mat = PBRMaterial.create_diffuse(Color.from_hex("#227DD7"), 0.2)
    sphere_shape.PBRMaterial = mat
    scene.add_object(VObject(shape=sphere_shape, name="SphereMin"))

    # Ground
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="GroundMin")
    matg = PBRMaterial.create_diffuse(Color.from_hex("#3F3F3F"), 0.9)
    ground.PBRMaterial = matg
    scene.add_object(VObject(shape=ground, name="GroundMin"))

    # Single light
    light = LightSource(position=np.array([2.0, 3.0, -1.0]), color=Color.from_hex("#FFFFFF"), intensity=15.0, radius=2, name="SunMin")
    scene.add_light(light)

    return scene

def get_emissive_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -3.5]), np.zeros(3), np.ones(3))
    cam = VCamera(cam_transform, fov=75.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    scene = Scene(name="emissive_scene", camera=cam, background_color=Color.from_hex("#000000"))

    # Emissive sphere
    emissive = Sphere(center=np.array([0.8, 1.0, 0.0]), radius=0.3, name="EmissiveOrb")
    mat_glow = PBRMaterial.create_emissive(Color.from_hex("#FFEA62"), 1.2)
    emissive.PBRMaterial = mat_glow
    scene.add_object(VObject(shape=emissive, name="GlowingSphere"))

    # Reflective sphere
    mirror = Sphere(center=np.array([-0.5, 0.5, 0.0]), radius=0.5, name="Mirror")
    mat_reflect = PBRMaterial.create_specular(Color.from_hex("#6B6666"), roughness=0.1, metallicness=0.5, specular_intensity=1.0, specular_tint_amount=1.0)
    mirror.PBRMaterial = mat_reflect
    scene.add_object(VObject(shape=mirror, name="MirrorSphere"))

    # Ground
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="GroundEmissive")
    matg = PBRMaterial.create_diffuse(Color.from_hex("#202020"), roughness=0.8)
    ground.PBRMaterial = matg
    scene.add_object(VObject(shape=ground, name="GroundEmissive"))

    # Small ambient fill light
    fill = LightSource(position=np.array([-4.0, 2.0, -3.0]), color=Color.from_hex("#AAAACC"), intensity=25.0, radius=10.0, name="FillEmiss")
    scene.add_light(fill)

    return scene

def get_lit_studio_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -4.0]), np.array([-0.15, 0.0, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=50.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    scene = Scene(name="lit_studio", camera=cam, background_color=Color.from_hex("#BEC2CF"))

    # Objects: two spheres and box as background
    s1 = Sphere(center=np.array([-0.6, 0.4, 0.5]), radius=0.4, name="StudioBallA")
    mat1 = PBRMaterial.create_specular(Color.from_hex("#FFB86B"), 0.2, 0.1, 0.9, 0)
    s1.PBRMaterial = mat1
    scene.add_object(VObject(shape=s1, name="StudioBallA"))

    s2 = Sphere(center=np.array([0.8, 0.45, 0.2]), radius=0.45, name="StudioBallB")
    mat2 = PBRMaterial.create_specular(Color.from_hex("#6B9BFF"), 0.2, 0.4, 0.9, 0)
    s2.PBRMaterial = mat2
    scene.add_object(VObject(shape=s2, name="StudioBallB"))

    # Background 
    box_shape = Cube(center=np.array([0.0, 0.5, 2.0]), side_length=6.0, name="StudioBack")
    box_shape.enlarge(np.array([1.0, 1.0, 0.1]))
    mat_box = PBRMaterial.create_diffuse(Color.from_hex("#C1CBD0"), roughness=1.0)
    box_shape.PBRMaterial = mat_box
    scene.add_object(VObject(shape=box_shape, name="StudioBox"))

    # Lights
    key = LightSource(position=np.array([2.5, 3.5, -1.0]), color=Color.from_hex("#EEE0BA"), intensity=25.0, radius=100, name="StudioKey")
    key.radius = 0.3  # area light radius (for soft shadows)
    scene.add_light(key)
    rim = LightSource(position=np.array([-3.0, 2.0, 1.0]), color=Color.from_hex("#DC97C5"), intensity=10.0, radius=0.75, name="StudioRim")
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
    mat_white = PBRMaterial.create_diffuse(Color.from_hex("#E0E0E0"), 1.0)
    mat_red   = PBRMaterial.create_diffuse(Color.from_hex("#B03030"), 1.0)
    mat_green = PBRMaterial.create_diffuse(Color.from_hex("#30B030"), 1.0)
    mat_blue = PBRMaterial.create_diffuse(Color.from_hex("#3036B0"), 1.0)
    
    # Objects (Shiny/Transmissive)
    mat_mirror = PBRMaterial.create_specular(Color.from_hex("#FFFFFF"), 0.1, 1.0, 0)
    mat_glass  = PBRMaterial.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["glass"], 0)

    # 3. Room Geometry (The Box)
    room_objects = []
    
    # Floor
    floor = Cube(center=np.array([0.0, -0.5, 0.0]), side_length=10.0, name="Floor")
    floor.transform.scale = np.array([2.0, 0.1, 2.0]) # Flatten into plane
    floor.PBRMaterial = mat_white
    room_objects.append(floor)
    
    # Ceiling
    ceiling = Cube(center=np.array([0.0, 6.5, 0.0]), side_length=10.0, name="Ceiling")
    ceiling.transform.scale = np.array([2.0, 0.1, 2.0])
    ceiling.PBRMaterial = mat_white
    room_objects.append(ceiling)

    # Back Wall
    back = Cube(center=np.array([0.0, 3.0, 5.5]), side_length=10.0, name="BackWall")
    back.transform.scale = np.array([2.0, 3.0, 0.1])
    back.PBRMaterial = mat_blue
    room_objects.append(back)
    
    # Left Wall (Red)
    left = Cube(center=np.array([-5.5, 3.0, 0.0]), side_length=10.0, name="LeftWall")
    left.transform.scale = np.array([0.1, 3.0, 2.0])
    left.PBRMaterial = mat_red
    room_objects.append(left)

    # Right Wall (Green)
    right = Cube(center=np.array([5.5, 3.0, 0.0]), side_length=10.0, name="RightWall")
    right.transform.scale = np.array([0.1, 3.0, 2.0])
    right.PBRMaterial = mat_green
    room_objects.append(right)
    
    # 4. Content Objects
    # Tall Box (Rotated)
    tall_box = Cube(center=np.array([-2.0, 1.5, 2.0]), side_length=3.0, name="TallBox")
    tall_box.transform.scale = np.array([0.6, 1.0, 0.6]) # Make it a pillar
    tall_box.transform.rotate(20.0, np.array([0.0, 1.0, 0.0])) # Rotate Y
    tall_box.PBRMaterial = mat_white # Standard white box for diffusal
    room_objects.append(tall_box)
    
    # Sphere (Mirror)
    mirror_sphere = Sphere(center=np.array([2.0, 1.25, 3.0]), radius=1.25, name="MirrorBall")
    mirror_sphere.PBRMaterial = mat_mirror
    room_objects.append(mirror_sphere)
    
    # Small Cube (Glass/Crystal in front)
    glass_cube = Cube(center=np.array([0.0, 0.75, -2.0]), side_length=1.5, name="GlassCube")
    glass_cube.transform.rotate(-15.0, np.array([0.0, 1.0, 0.0]))
    glass_cube.PBRMaterial = mat_glass
    room_objects.append(glass_cube)

    # 5. Lighting
    # A single strong area light on the ceiling (simulating the Cornell Box light patch)
    ceiling_light = LightSource(
        position=np.array([0.0, 5.8, 0.0]), 
        color=Color.from_hex("#FFECDE"), 
        intensity=25.0, 
        radius=1.5, 
        name="CeilingLight"
    )

    # 6. Point camera towards the room center
    cam.transform.look_at(np.array([0.0, 2.5, 0.0]))

    # Assemble Scene
    scene = Scene(name="rgb_cornell_box", camera=cam, background_color=Color(0,0,0)) # Pitch black void outside
    
    scene.add_light(ceiling_light)
    for obj in room_objects:
        scene.add_object(VObject(shape=obj, name=obj.name))

    return scene

def get_cyberpunk_scene(width: int = 120, height: int = 120) -> Scene:
    # Low angle camera, looking up slightly
    cam_transform = Transform(np.array([0.0, 0.5, -4.0]), np.array([-0.1, 0.0, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    
    # Dark Night Sky
    # Background Gradient
    sky_colors = [
        Color.from_hex("#050008"),
        Color.from_hex("#0B1333"),
    ]
    sky_positions = [0.0, 1.0] 
    scene = Scene(name="cyberpunk_street", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    # 1. The "Wet" Asphalt Road (Dark, slightly reflective, uneven roughness)
    road = Cube(center=np.array([0.0, -1.0, 0.0]), side_length=20.0, name="WetRoad")
    road.transform.scale = np.array([1.0, 0.1, 2.0]) 
    mat_wet = PBRMaterial.create_diffuse(Color.from_hex("#151515"), roughness=0.2) # Low roughness = wet look
    road.PBRMaterial = mat_wet
    scene.add_object(VObject(shape=road, name="Road"))

    # 2. Hero Object: Chrome Sphere (reflects the neon lights)
    hero = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.8, name="ChromeHero")
    mat_chrome = PBRMaterial.create_specular(Color.from_hex("#313238"), roughness=0.2, metallicness=1.0)
    hero.PBRMaterial = mat_chrome
    scene.add_object(VObject(shape=hero, name="HeroSphere"))

    # 3. Background Buildings (Silhouettes)
    bldg_left = Cube(center=np.array([-2.5, 2.0, 2.0]), side_length=4.0, name="BuildingLeft")
    bldg_left.transform.scale = np.array([0.5, 2.0, 0.5])
    bldg_left.PBRMaterial = PBRMaterial.create_diffuse(Color.from_hex("#4DBC3E"), roughness=0.9)
    scene.add_object(VObject(shape=bldg_left, name="BldgLeft"))

    bldg_right = Cube(center=np.array([2.5, 1.3, 2.2]), side_length=4.0, name="BuildingRight")
    bldg_right.transform.scale = np.array([0.65, 1.3, 0.65])
    bldg_right.PBRMaterial = PBRMaterial.create_diffuse(Color.from_hex("#E28335"), roughness=0.9)
    scene.add_object(VObject(shape=bldg_right, name="BldgRight"))

    # 4. Lighting (The "Cyberpunk" look relies on split toning)
    # Light A: Hot Pink (Left)
    light_pink = LightSource(
        position=np.array([-3.0, 2.0, -2.0]), 
        color=Color.from_hex("#FF0099"), 
        intensity=15.0, 
        radius=0.2, 
        name="NeonPink"
    )
    
    # Light B: Cyan/Electric Blue (Right)
    light_cyan = LightSource(
        position=np.array([-2.5, 1.5, 2.0]), 
        color=Color.from_hex("#00F0FF"), 
        intensity=12.0, 
        radius=0.2, 
        name="NeonCyan"
    )

    # Light C: Neon Blue (Between buildings)
    light_blue = LightSource(
        position=np.array([3.0, 1.0, -1.0]), 
        color=Color.from_hex("#3700FF"), 
        intensity=10.0, 
        radius=0.2, 
        name="NeonBlue"
    )

    # Light D: Street light rim (Back)
    light_rim = LightSource(
        position=np.array([0.0, 3.0, 4.0]),
        color=Color.from_hex("#FFFFFF"),
        intensity=5.0,
        radius=0.5,
        name="StreetLight"
    )

    scene.add_light(light_pink)
    scene.add_light(light_cyan)
    scene.add_light(light_blue)
    scene.add_light(light_rim)
    return scene

def get_material_deck_scene(width: int = 160, height: int = 80) -> Scene:
    # Wide aspect ratio to see the line of spheres
    cam_transform = Transform(np.array([0.0, 1.5, -5.0]), np.array([0.2, 0.0, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=50.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    
    scene = Scene(name="material_deck", camera=cam, background_color=Color.from_hex("#000000"))

    # Floor
    floor = Cube(center=np.array([0.0, -1.0, 0.0]), side_length=15.0, name="Floor")
    floor.transform.scale = np.array([2.0, 0.1, 1.0])
    floor.PBRMaterial = PBRMaterial.create_diffuse(Color.from_hex("#CCCCCC"), roughness=1.0)
    scene.add_object(VObject(shape=floor, name="Floor"))

    # --- The Spheres ---
    # We will create a Gold PBRMaterial and vary its roughness from left to right.
    
    base_col = Color.from_hex("#D4AF37")
    
    # 1. Perfectly Polished Gold (Mirror)
    s1 = Sphere(center=np.array([-3.0, 0.5, 0.0]), radius=0.6, name="Gold_0.0")
    s1.PBRMaterial = PBRMaterial.create_specular(base_col, roughness=0.0)
    scene.add_object(VObject(shape=s1, name="S_Mirror"))

    # 2. Slightly Brushed Gold
    s2 = Sphere(center=np.array([-1.5, 0.5, 0.0]), radius=0.6, name="Gold_0.25")
    s2.PBRMaterial = PBRMaterial.create_specular(base_col, roughness=0.25)
    scene.add_object(VObject(shape=s2, name="S_Brushed"))

    # 3. Rough Gold
    s3 = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.6, name="Gold_0.5")
    s3.PBRMaterial = PBRMaterial.create_specular(base_col, roughness=0.5)
    scene.add_object(VObject(shape=s3, name="S_Rough"))

    # 4. Matte/Dusty Gold
    s4 = Sphere(center=np.array([1.5, 0.5, 0.0]), radius=0.6, name="Gold_0.75")
    s4.PBRMaterial = PBRMaterial.create_specular(base_col, roughness=0.75)
    scene.add_object(VObject(shape=s4, name="S_Matte"))
    
    # 5. Control: Red Plastic (Dielectric)
    s5 = Sphere(center=np.array([3.0, 0.5, 0.0]), radius=0.6, name="Plastic_Red")
    s5.PBRMaterial = PBRMaterial.create_diffuse(Color.from_hex("#FF4A4A"), roughness=0.1)
    scene.add_object(VObject(shape=s5, name="S_Plastic"))

    # Lighting: Standard Studio Setup
    scene.add_light(LightSource(np.array([0.0, 5.0, -5.0]), Color(1,1,1), 20.0, name="Main"))
    scene.add_light(LightSource(np.array([5.0, 2.0, -2.0]), Color(0.8, 0.8, 1.0), 10.0, name="Fill"))

    cam.transform.look_at(s3.transform.position)
    return scene

def get_refraction_lab_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.0, -4.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=45.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)
    
    scene = Scene(name="refraction_lab", camera=cam, background_color=Color(0.05, 0.05, 0.05))

    # Striped Background Wall (To visualize the distortion/refraction clearly)
    wall = Cube(center=np.array([0.0, 2.0, 4.0]), side_length=8.0, name="StripedWall")
    wall.transform.scale = np.array([1.5, 1.0, 0.1])
    # Give it a bright texture or color so we can see it warp
    wall.PBRMaterial = PBRMaterial.create_emissive(Color(1.0, 1.0, 1.0), 1.0) # Bright white wall
    scene.add_object(VObject(shape=wall, name="BackWall"))

    # Blocker bars (Black cubes to create stripes on the white wall)
    for i in range(-6, 7):
        bar = Cube(center=np.array([i, 2.0, 3.5]), side_length=5.0, name=f"Bar_{i}")
        bar.scale(np.array([0.1, 4.0, 0.1]))
        bar.PBRMaterial = PBRMaterial.create_diffuse(Color(0, 0, 0), 1.0)
        scene.add_object(VObject(shape=bar, name=f"Bar_{i}"))

    # 1. Glass Sphere (IOR 1.5)
    s_glass = Sphere(center=np.array([-1.2, 0.5, 0.0]), radius=0.6, name="Acrylic")
    s_glass.PBRMaterial  = PBRMaterial.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["acrylic"], 0)
    scene.add_object(VObject(shape=s_glass, name="AcrylicSphere"))

    # 2. Diamond Sphere (IOR 2.4 - Should bend light heavily)
    s_diamond = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.6, name="Diamond")
    s_diamond.PBRMaterial = PBRMaterial.create_glass(Color.from_hex("#B9D3E3"), Color(0.9, 0.9, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["diamond"], 0.2)
    scene.add_object(VObject(shape=s_diamond, name="DiamondSphere"))

    # 3. Water Sphere / Bubble (IOR 1.33 - Subtle bending)
    s_water = Sphere(center=np.array([1.2, 0.5, 0.0]), radius=0.6, name="Water")
    s_water.PBRMaterial  = PBRMaterial.create_glass(Color.from_hex("#A6ADD5"), Color.from_hex("#1F1FFF"), 0.0, 0.0, REFRACTIVE_INDICES["water"], 0.1)
    scene.add_object(VObject(shape=s_water, name="WaterSphere"))

    # Lighting
    # We need front lighting to see the specular highlights (shininess)
    scene.add_light(LightSource(np.array([2.0, 3.0, -3.0]), Color(1,1,1), 15.0, name="FrontLight"))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))

    return scene