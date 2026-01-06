import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.PrimaryStructures import Transform
from src.Geometry import Sphere, Cube, VObject
from src.Scene import Scene
from src.Camera import VCamera, CameraType
from src.Luminance import LightSource, Color, ColorGradient, PBRMaterial
from src.Refractions import REFRACTIVE_INDICES

def get_gradient_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.5, -4.0]), np.array([0, 0.2, 0]), np.ones(3))
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )

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
    mat_metal = PBRMaterial.create_specular(Color.from_hex("#47505C"), 0.2, 0.9, 1.0, 1.0)
    sphere_shape_1.material = mat_metal

    # Additional Object 1: Cube (Background/Visual Anchor) - Matte and Rough
    box_shape = Cube(center=np.array([2.0, 3.0, 4.0]), side_length=2.5, name="BackgroundBox")
    box_shape.rotate(np.radians(15), np.array([0, 1, 0]))
    mat_matte = PBRMaterial.create_diffuse(Color.from_hex("#C27A23"), roughness=0.8)
    box_shape.material = mat_matte
    
    # Additional Object 2: Small Emissive Sphere (Light Source Helper) - Floating in air
    sphere_shape_2 = Sphere(center=np.array([-0.5, 2.5, 1.5]), radius=1, name="EmissiveOrb")
    mat_glow = PBRMaterial.create_emissive(Color.from_hex("#EE1717"), 2)
    sphere_shape_2.material = mat_glow

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
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene(name="minimal_scene", camera=cam, background_color=Color.from_hex("#403A43"))

    # Sphere at origin
    sphere_shape = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5, name="BallMin")
    mat = PBRMaterial.create_diffuse(Color.from_hex("#227DD7"), 0.2)
    sphere_shape.material = mat
    scene.add_object(VObject(shape=sphere_shape, name="SphereMin"))

    # Ground
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="GroundMin")
    matg = PBRMaterial.create_diffuse(Color.from_hex("#3F3F3F"), 0.9)
    ground.material = matg
    scene.add_object(VObject(shape=ground, name="GroundMin"))

    # Single light
    light = LightSource(position=np.array([2.0, 3.0, -1.0]), color=Color.from_hex("#FFFFFF"), intensity=15.0, radius=2, name="SunMin")
    scene.add_light(light)

    return scene

def get_emissive_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -3.5]), np.zeros(3), np.ones(3))
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene(name="emissive_scene", camera=cam, background_color=Color.from_hex("#000000"))

    # Emissive sphere
    emissive = Sphere(center=np.array([0.8, 1.0, 0.0]), radius=0.3, name="EmissiveOrb")
    mat_glow = PBRMaterial.create_emissive(Color.from_hex("#FFEA62"), 1.2)
    emissive.material = mat_glow
    scene.add_object(VObject(shape=emissive, name="GlowingSphere"))

    # Reflective sphere
    mirror = Sphere(center=np.array([-0.5, 0.5, 0.0]), radius=0.5, name="Mirror")
    mat_reflect = PBRMaterial.create_specular(Color.from_hex("#6B6666"), roughness=0.1, metallicness=0.5, specular_intensity=1.0, specular_tint_amount=1.0)
    mirror.material = mat_reflect
    scene.add_object(VObject(shape=mirror, name="MirrorSphere"))

    # Ground
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="GroundEmissive")
    matg = PBRMaterial.create_diffuse(Color.from_hex("#202020"), roughness=0.8)
    ground.material = matg
    scene.add_object(VObject(shape=ground, name="GroundEmissive"))

    # Small ambient fill light
    fill = LightSource(position=np.array([-4.0, 2.0, -3.0]), color=Color.from_hex("#AAAACC"), intensity=25.0, radius=10.0, name="FillEmiss")
    scene.add_light(fill)

    return scene

def get_lit_studio_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -4.0]), np.array([-0.15, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene(name="lit_studio", camera=cam, background_color=Color.from_hex("#BEC2CF"))

    # Objects: two spheres and box as background
    s1 = Sphere(center=np.array([-0.6, 0.4, 0.5]), radius=0.4, name="StudioBallA")
    mat1 = PBRMaterial.create_specular(Color.from_hex("#FFB86B"), 0.2, 0.1, 0.9, 0)
    s1.material = mat1
    scene.add_object(VObject(shape=s1, name="StudioBallA"))

    s2 = Sphere(center=np.array([0.8, 0.45, 0.2]), radius=0.45, name="StudioBallB")
    mat2 = PBRMaterial.create_specular(Color.from_hex("#6B9BFF"), 0.2, 0.4, 0.9, 0)
    s2.material = mat2
    scene.add_object(VObject(shape=s2, name="StudioBallB"))

    # Background 
    box_shape = Cube(center=np.array([0.0, 0.5, 2.0]), side_length=6.0, name="StudioBack")
    box_shape.scale(np.array([1.0, 1.0, 0.1]))
    mat_box = PBRMaterial.create_diffuse(Color.from_hex("#C1CBD0"), roughness=1.0)
    box_shape.material = mat_box
    scene.add_object(VObject(shape=box_shape, name="StudioBox"))

    # Lights
    key = LightSource(position=np.array([2.5, 3.5, -1.0]), color=Color.from_hex("#EEE0BA"), intensity=25.0, radius=100, name="StudioKey")
    key.radius = 0.3
    scene.add_light(key)
    rim = LightSource(position=np.array([-3.0, 2.0, 1.0]), color=Color.from_hex("#DC97C5"), intensity=10.0, radius=0.75, name="StudioRim")
    rim.radius = 0.2
    scene.add_light(rim)
    fill = LightSource(position=np.array([0.0, -2.5, -2.0]), color=Color.from_hex("#C7DBD8"), intensity=15.0, radius=2, name="StudioFill")
    scene.add_light(fill)

    return scene

def get_rgb_room_with_objects_scene(width: int = 126, height: int = 126) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.5, -7.5]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    # Materials
    mat_white = PBRMaterial.create_diffuse(Color.from_hex("#E0E0E0"), 1.0)
    mat_red   = PBRMaterial.create_diffuse(Color.from_hex("#B03030"), 1.0)
    mat_green = PBRMaterial.create_diffuse(Color.from_hex("#30B030"), 1.0)
    mat_blue = PBRMaterial.create_diffuse(Color.from_hex("#3036B0"), 1.0)
    
    mat_mirror = PBRMaterial.create_specular(Color.from_hex("#FFFFFF"), 0.1, 1.0, 0)
    mat_glass  = PBRMaterial.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["glass"], 0)

    room_objects = []
    
    # Floor
    floor = Cube(center=np.array([0.0, -0.5, 0.0]), side_length=10.0, name="Floor")
    floor.transform.scale = np.array([2.0, 0.1, 2.0])
    floor.material = mat_white
    room_objects.append(floor)
    
    # Ceiling
    ceiling = Cube(center=np.array([0.0, 6.5, 0.0]), side_length=10.0, name="Ceiling")
    ceiling.transform.scale = np.array([2.0, 0.1, 2.0])
    ceiling.material = mat_white
    room_objects.append(ceiling)

    # Back Wall
    back = Cube(center=np.array([0.0, 3.0, 5.5]), side_length=10.0, name="BackWall")
    back.transform.scale = np.array([2.0, 3.0, 0.1])
    back.material = mat_blue
    room_objects.append(back)
    
    # Left Wall (Red)
    left = Cube(center=np.array([-5.5, 3.0, 0.0]), side_length=10.0, name="LeftWall")
    left.transform.scale = np.array([0.1, 3.0, 2.0])
    left.material = mat_red
    room_objects.append(left)

    # Right Wall (Green)
    right = Cube(center=np.array([5.5, 3.0, 0.0]), side_length=10.0, name="RightWall")
    right.transform.scale = np.array([0.1, 3.0, 2.0])
    right.material = mat_green
    room_objects.append(right)
    
    # Tall Box (Rotated)
    tall_box = Cube(center=np.array([-2.0, 1.5, 2.0]), side_length=3.0, name="TallBox")
    tall_box.transform.scale = np.array([0.6, 1.0, 0.6])
    tall_box.transform.rotate(20.0, np.array([0.0, 1.0, 0.0]))
    tall_box.material = mat_white
    room_objects.append(tall_box)
    
    # Sphere (Mirror)
    mirror_sphere = Sphere(center=np.array([2.0, 1.25, 3.0]), radius=1.25, name="MirrorBall")
    mirror_sphere.material = mat_mirror
    room_objects.append(mirror_sphere)
    
    # Small Cube (Glass/Crystal in front)
    glass_cube = Cube(center=np.array([0.0, 0.75, -2.0]), side_length=1.5, name="GlassCube")
    glass_cube.transform.rotate(-15.0, np.array([0.0, 1.0, 0.0]))
    glass_cube.material = mat_glass
    room_objects.append(glass_cube)

    # Lighting
    ceiling_light = LightSource(
        position=np.array([0.0, 5.8, 0.0]), 
        color=Color.from_hex("#FFECDE"), 
        intensity=25.0, 
        radius=1.5, 
        name="CeilingLight"
    )

    cam.transform.look_at(np.array([0.0, 2.5, 0.0]))

    scene = Scene(name="rgb_cornell_box", camera=cam, background_color=Color(0.0, 0.0, 0.0))
    
    scene.add_light(ceiling_light)
    for obj in room_objects:
        scene.add_object(VObject(shape=obj, name=obj.name))

    return scene

def get_cyberpunk_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -4.0]), np.array([-0.1, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    sky_colors = [
        Color.from_hex("#050008"),
        Color.from_hex("#0B1333"),
    ]
    sky_positions = [0.0, 1.0] 
    scene = Scene(name="cyberpunk_street", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    # Road
    road = Cube(center=np.array([0.0, -1.0, 0.0]), side_length=20.0, name="WetRoad")
    road.transform.scale = np.array([1.0, 0.1, 2.0]) 
    mat_wet = PBRMaterial.create_diffuse(Color.from_hex("#151515"), roughness=0.2)
    road.material = mat_wet
    scene.add_object(VObject(shape=road, name="Road"))

    # Hero Object: Chrome Sphere
    hero = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.8, name="ChromeHero")
    mat_chrome = PBRMaterial.create_specular(Color.from_hex("#313238"), roughness=0.2, metallicness=1.0)
    hero.material = mat_chrome
    scene.add_object(VObject(shape=hero, name="HeroSphere"))

    # Background Buildings
    bldg_left = Cube(center=np.array([-2.5, 2.0, 2.0]), side_length=4.0, name="BuildingLeft")
    bldg_left.transform.scale = np.array([0.5, 2.0, 0.5])
    bldg_left.material = PBRMaterial.create_diffuse(Color.from_hex("#4DBC3E"), roughness=0.9)
    scene.add_object(VObject(shape=bldg_left, name="BldgLeft"))

    bldg_right = Cube(center=np.array([2.5, 1.3, 2.2]), side_length=4.0, name="BuildingRight")
    bldg_right.transform.scale = np.array([0.65, 1.3, 0.65])
    bldg_right.material = PBRMaterial.create_diffuse(Color.from_hex("#E28335"), roughness=0.9)
    scene.add_object(VObject(shape=bldg_right, name="BldgRight"))

    # Lighting
    light_pink = LightSource(
        position=np.array([-3.0, 2.0, -2.0]), 
        color=Color.from_hex("#FF0099"), 
        intensity=15.0, 
        radius=0.2, 
        name="NeonPink"
    )
    
    light_cyan = LightSource(
        position=np.array([-2.5, 1.5, 2.0]), 
        color=Color.from_hex("#00F0FF"), 
        intensity=12.0, 
        radius=0.2, 
        name="NeonCyan"
    )

    light_blue = LightSource(
        position=np.array([3.0, 1.0, -1.0]), 
        color=Color.from_hex("#3700FF"), 
        intensity=10.0, 
        radius=0.2, 
        name="NeonBlue"
    )

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
    cam_transform = Transform(np.array([0.0, 1.5, -5.0]), np.array([0.2, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="material_deck", camera=cam, background_color=Color.from_hex("#000000"))

    # Floor
    floor = Cube(center=np.array([0.0, -1.0, 0.0]), side_length=15.0, name="Floor")
    floor.transform.scale = np.array([2.0, 0.1, 1.0])
    floor.material = PBRMaterial.create_diffuse(Color.from_hex("#CCCCCC"), roughness=1.0)
    scene.add_object(VObject(shape=floor, name="Floor"))

    base_col = Color.from_hex("#D4AF37")
    
    # Spheres with varying roughness
    s1 = Sphere(center=np.array([-3.0, 0.5, 0.0]), radius=0.6, name="Gold_0.0")
    s1.material = PBRMaterial.create_specular(base_col, roughness=0.0)
    scene.add_object(VObject(shape=s1, name="S_Mirror"))

    s2 = Sphere(center=np.array([-1.5, 0.5, 0.0]), radius=0.6, name="Gold_0.25")
    s2.material = PBRMaterial.create_specular(base_col, roughness=0.25)
    scene.add_object(VObject(shape=s2, name="S_Brushed"))

    s3 = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.6, name="Gold_0.5")
    s3.material = PBRMaterial.create_specular(base_col, roughness=0.5)
    scene.add_object(VObject(shape=s3, name="S_Rough"))

    s4 = Sphere(center=np.array([1.5, 0.5, 0.0]), radius=0.6, name="Gold_0.75")
    s4.material = PBRMaterial.create_specular(base_col, roughness=0.75)
    scene.add_object(VObject(shape=s4, name="S_Matte"))
    
    s5 = Sphere(center=np.array([3.0, 0.5, 0.0]), radius=0.6, name="Plastic_Red")
    s5.material = PBRMaterial.create_diffuse(Color.from_hex("#FF0000"), roughness=0.1)
    scene.add_object(VObject(shape=s5, name="S_Plastic"))

    # Lighting
    scene.add_light(LightSource(np.array([0.0, 5.0, -5.0]), Color(1.0, 1.0, 1.0), 15.0, name="Main"))
    scene.add_light(LightSource(np.array([5.0, 2.0, -2.0]), Color(0.8, 0.8, 1.0), 10.0, name="Fill"))

    cam.transform.look_at(s3.transform.position)
    return scene

def get_refraction_lab_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.0, -4.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="refraction_lab", camera=cam, background_color=Color(0.05, 0.05, 0.05))

    # Striped Background Wall
    wall = Cube(center=np.array([0.0, 2.0, 4.0]), side_length=8.0, name="StripedWall")
    wall.transform.scale = np.array([1.5, 1.0, 0.1])
    wall.material = PBRMaterial.create_emissive(Color(1.0, 1.0, 1.0), 1.0)
    scene.add_object(VObject(shape=wall, name="BackWall"))

    # Blocker bars
    for i in range(-6, 7):
        bar = Cube(center=np.array([i, 2.0, 3.5]), side_length=5.0, name=f"Bar_{i}")
        bar.scale(np.array([0.1, 4.0, 0.1]))
        bar.material = PBRMaterial.create_diffuse(Color(0.0, 0.0, 0.0), 1.0)
        scene.add_object(VObject(shape=bar, name=f"Bar_{6 + i}"))

    # Glass Sphere (IOR 1.5)
    s_glass = Sphere(center=np.array([-1.2, 0.5, 0.0]), radius=0.6, name="Acrylic")
    s_glass.material = PBRMaterial.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["acrylic"], 0)
    scene.add_object(VObject(shape=s_glass, name="AcrylicSphere"))

    # Diamond Sphere (IOR 2.4)
    s_diamond = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.6, name="Diamond")
    s_diamond.material = PBRMaterial.create_glass(Color.from_hex("#B9D3E3"), Color(0.9, 0.9, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["diamond"], 0.2)
    scene.add_object(VObject(shape=s_diamond, name="DiamondSphere"))

    # Water Sphere / Bubble (IOR 1.33)
    s_water = Sphere(center=np.array([1.2, 0.5, 0.0]), radius=0.6, name="Water")
    s_water.material = PBRMaterial.create_glass(Color.from_hex("#A6ADD5"), Color.from_hex("#1F1FFF"), 0.0, 0.0, REFRACTIVE_INDICES["water"], 0.1)
    scene.add_object(VObject(shape=s_water, name="WaterSphere"))

    # Lighting
    scene.add_light(LightSource(np.array([2.0, 3.0, -3.0]), Color(1.0, 1.0, 1.0), 15.0, name="FrontLight"))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))
    return scene

def get_scifi_corridor_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A high-contrast scene featuring repetitive metallic geometry and emissive lighting.
    Focuses on reflections of light sources on rough metal.
    """
    cam_transform = Transform(np.array([0.0, 1.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform, fov=80.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="scifi_corridor", camera=cam, background_color=Color.from_hex("#020205"))

    # Materials
    mat_floor = PBRMaterial.create_specular(Color.from_hex("#2A2A2A"), roughness=0.3, metallicness=0.8)
    mat_pillar = PBRMaterial.create_specular(Color.from_hex("#111111"), roughness=0.5, metallicness=0.9)
    mat_light_strip = PBRMaterial.create_emissive(Color.from_hex("#00FFFF"), 5.0)

    # Floor
    floor = Cube(center=np.array([0.0, -1.0, -10.0]), side_length=20.0, name="CorridorFloor")
    floor.transform.scale = np.array([0.5, 0.1, 4.0])
    floor.material = mat_floor
    scene.add_object(VObject(shape=floor, name="Floor"))

    # Ceiling
    ceiling = Cube(center=np.array([0.0, 3.0, -10.0]), side_length=20.0, name="CorridorCeiling")
    ceiling.transform.scale = np.array([0.5, 0.1, 4.0])
    ceiling.material = mat_floor
    scene.add_object(VObject(shape=ceiling, name="Ceiling"))

    # Repetitive Pillars and Lights
    for z in range(0, -20, -4):
        # Left Pillar
        p_left = Cube(center=np.array([-2.5, 1.0, z]), side_length=2.0, name=f"PillarL_{z}")
        p_left.transform.scale = np.array([0.5, 2.0, 0.5])
        p_left.material = mat_pillar
        scene.add_object(VObject(shape=p_left, name=f"PillarLeft_{z}"))

        # Right Pillar
        p_right = Cube(center=np.array([2.5, 1.0, z]), side_length=2.0, name=f"PillarR_{z}")
        p_right.transform.scale = np.array([0.5, 2.0, 0.5])
        p_right.material = mat_pillar
        scene.add_object(VObject(shape=p_right, name=f"PillarRight_{z}"))

        # Emissive Light Strips on floor edges
        l_strip = Cube(center=np.array([0.0, -0.9, z]), side_length=0.2, name=f"Light_{z}")
        l_strip.transform.scale = np.array([8.0, 0.1, 0.5])
        l_strip.material = mat_light_strip
        scene.add_object(VObject(shape=l_strip, name=f"Strip_{z}"))

        # Actual Light Sources corresponding to strips
        light = LightSource(position=np.array([0.0, 0.5, z]), color=Color.from_hex("#00AAAA"), intensity=5.0, radius=2.0, name=f"PointLight_{z}")
        scene.add_light(light)

    # End focal point
    sphere_end = Sphere(center=np.array([0.0, 1.0, -18.0]), radius=1.5, name="EndSphere")
    sphere_end.material = PBRMaterial.create_specular(Color.from_hex("#FF0000"), roughness=0.1, metallicness=1.0)
    scene.add_object(VObject(shape=sphere_end, name="EndSphere"))

    cam.transform.look_at(np.array([0, 1, -20]))
    return scene

def get_sunset_monolith_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A scene focusing on warm lighting, long shadows, and the contrast between
    a matte organic ground and a sharp, reflective geometric object.
    """
    cam_transform = Transform(np.array([3.0, 1.5, -4.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform, fov=65.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )

    # Sunset Gradient Background
    sky_colors = [
        Color.from_hex("#38160D"), # Dark ground horizon
        Color.from_hex("#E85635"), # Deep Orange
        Color.from_hex("#F29B36"), # Gold
        Color.from_hex("#685888"), # Purple zenith
    ]
    sky_positions = [0.0, 0.3, 0.5, 1.0]
    
    scene = Scene(name="sunset_monolith", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    # The Monolith (Highly Specular Black Metal)
    monolith = Cube(center=np.array([0.0, 2.0, 0.0]), side_length=4.0, name="Monolith")
    monolith.transform.scale = np.array([0.4, 2.0, 0.4])
    monolith.transform.rotate(np.radians(25), np.array([0, 1, 0]))
    mat_mono = PBRMaterial.create_specular(Color.from_hex("#050505"), roughness=0.05, metallicness=1.0)
    monolith.material = mat_mono
    scene.add_object(VObject(shape=monolith, name="MonolithObj"))

    # Sand Dunes (Matte, rough)
    floor = Sphere(center=np.array([0.0, -51.0, 0.0]), radius=50.0, name="Sand")
    mat_sand = PBRMaterial.create_diffuse(Color.from_hex("#D6783B"), roughness=1.0)
    floor.material = mat_sand
    scene.add_object(VObject(shape=floor, name="SandGround"))

    # Floating particles/smaller rocks
    rock1 = Sphere(center=np.array([-1.5, 0.3, 1.5]), radius=0.3, name="Rock1")
    rock1.material = PBRMaterial.create_diffuse(Color.from_hex("#554433"), roughness=0.9)
    scene.add_object(VObject(shape=rock1, name="Rock1"))

    # Lighting
    # Sun (Low angle, very bright, sharp shadows)
    sun = LightSource(position=np.array([-8.0, 2.0, 10.0]), color=Color.from_hex("#FF9944"), intensity=30.0, radius=100.0, name="Sun")
    sun.radius = 0.5 # Make it physically small for sharp shadows
    scene.add_light(sun)

    # Skylight fill (Purple/Blue ambient)
    fill = LightSource(position=np.array([5.0, 10.0, -5.0]), color=Color.from_hex("#5544AA"), intensity=4.0, radius=20.0, name="SkyFill")
    scene.add_light(fill)

    cam.transform.look_at(np.array([0, 1.5, 0]))
    return scene

def get_pastel_blocks_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A 'toy' scene with soft, bright lighting and materials that look like plastic or chalk.
    No metallic or glass surfaces.
    """
    cam_transform = Transform(np.array([0.0, 3.0, -5.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform, fov=60.0, near=0.1, far=50.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )

    scene = Scene(name="pastel_blocks", camera=cam, background_color=Color.from_hex("#F0F4F8"))

    # Materials (Plastic/Chalky: Diffuse with very low specular or high roughness)
    mat_pink = PBRMaterial.create_diffuse(Color.from_hex("#FFB7B2"), roughness=0.6)
    mat_mint = PBRMaterial.create_diffuse(Color.from_hex("#B5EAD7"), roughness=0.6)
    mat_purple = PBRMaterial.create_diffuse(Color.from_hex("#E2F0CB"), roughness=0.6) # Actually yellowish-green
    mat_white = PBRMaterial.create_diffuse(Color.from_hex("#FFFFFF"), roughness=0.9)

    # Floor
    floor = Cube(center=np.array([0.0, -1.0, 0.0]), side_length=10.0, name="WhiteFloor")
    floor.transform.scale = np.array([2.0, 0.1, 2.0])
    floor.material = mat_white
    scene.add_object(VObject(shape=floor, name="Floor"))

    # Stacked Objects
    # Base Cube
    base = Cube(center=np.array([0.0, 0.0, 0.0]), side_length=2.0, name="BaseCube")
    base.transform.rotate(np.radians(15), np.array([0, 1, 0]))
    base.material = mat_mint
    scene.add_object(VObject(shape=base, name="BaseObj"))

    # Middle Cylinder (Simulated by stretched sphere or cube? using Sphere for variety)
    mid = Sphere(center=np.array([0.0, 1.6, 0.0]), radius=0.8, name="MidSphere")
    mid.material = mat_pink
    scene.add_object(VObject(shape=mid, name="MidObj"))

    # Top floating cube
    top = Cube(center=np.array([0.2, 2.8, 0.2]), side_length=1.0, name="TopCube")
    top.transform.rotate(np.radians(45), np.array([1, 1, 0]))
    top.material = mat_purple
    scene.add_object(VObject(shape=top, name="TopObj"))

    # Lighting (Soft Studio setup)
    # Main soft light
    key = LightSource(position=np.array([3.0, 5.0, -5.0]), color=Color.from_hex("#FFFBEB"), intensity=18.0, radius=5.0, name="Key")
    scene.add_light(key)

    # Fill light
    fill = LightSource(position=np.array([-4.0, 2.0, -2.0]), color=Color.from_hex("#E6E6FA"), intensity=8.0, radius=5.0, name="Fill")
    scene.add_light(fill)

    cam.transform.look_at(np.array([0, 1.2, 0]))
    return scene

def get_glass_prism_scene(width: int = 120, height: int = 120) -> Scene:
    """
    Features multiple glass objects with different Refractive Indices (IOR) 
    to demonstrate varying levels of light bending.
    """
    cam_transform = Transform(np.array([0.0, 2.0, -6.0]), np.array([-0.2, 0.0, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=70.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="glass_prism_row", camera=cam, background_color=Color.from_hex("#101015"))

    # 1. Diamond Sphere (High IOR: 2.42) - Center
    # High dispersion and internal reflection
    sphere_diamond = Sphere(center=np.array([0.0, 0.5, 0.0]), radius=0.6, name="Diamond")
    mat_diamond = PBRMaterial.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["diamond"], 
        transmission=1.0
    )
    sphere_diamond.material = mat_diamond
    scene.add_object(VObject(shape=sphere_diamond, name="CenterDiamond"))

    # 2. Water Sphere (Low IOR: 1.33) - Left
    # Subtle bending, looks more transparent
    sphere_water = Sphere(center=np.array([-1.5, 0.5, 0.0]), radius=0.6, name="Water")
    mat_water = PBRMaterial.create_glass(
        Color(0.9, 0.9, 1.0), 
        Color(0.8, 0.9, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=1.33, 
        transmission=1.0
    )
    sphere_water.material = mat_water
    scene.add_object(VObject(shape=sphere_water, name="LeftWater"))

    # 3. Heavy Flint Glass Cube (Medium-High IOR: 1.65) - Right
    cube_glass = Cube(center=np.array([1.5, 0.5, 0.0]), side_length=1.2, name="FlintGlass")
    # Rotate to show refraction through edges
    cube_glass.transform.rotate(np.radians(30), np.array([0, 1, 0]))
    cube_glass.transform.rotate(np.radians(10), np.array([1, 0, 0]))
    mat_flint = PBRMaterial.create_glass(
        Color(1.0, 0.9, 0.9), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.01, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["glass_flint_heavy"], 
        transmission=1.0
    )
    cube_glass.material = mat_flint
    scene.add_object(VObject(shape=cube_glass, name="RightFlint"))

    # Checkerboard Floor (to make refraction obvious)
    # Note: Since we don't have texture support in this snippet, we use a highly patterned background object
    floor = Cube(center=np.array([0.0, -10.5, 5.0]), side_length=20.0, name="Floor")
    floor.transform.scale = np.array([1.0, 1.0, 0.5]) # Flatten
    # Using a striped emissive material to create lines visible THROUGH the glass
    mat_floor = PBRMaterial.create_diffuse(Color.from_hex("#888888"), roughness=0.8)
    floor.material = mat_floor
    scene.add_object(VObject(shape=floor, name="FloorBase"))

    # Striped Wall behind objects
    for i in range(-5, 6):
        bar = Cube(center=np.array([i, 2.0, 4.0]), side_length=0.5, name=f"Strip_{i}")
        bar.transform.scale = np.array([0.5, 8.0, 0.1])
        # Alternating colors
        col = Color.from_hex("#FF0000") if i % 2 == 0 else Color.from_hex("#FFFFFF")
        bar.material = PBRMaterial.create_emissive(col, 2.0)
        scene.add_object(VObject(shape=bar, name=f"Bar_{i}"))

    # Light
    scene.add_light(LightSource(np.array([0.0, 5.0, -3.0]), Color(1.0, 1.0, 1.0), 20.0, name="TopLight"))

    return scene

def get_glass_sculpture_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A complex arrangement of overlapping glass plates and spheres.
    Good for testing recursion depth and transmission color absorption.
    """
    cam_transform = Transform(np.array([3.0, 2.5, -3.0]), np.array([-0.3, 0.7, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    cam.transform.look_at(np.array([0, 0.5, 0]))
    
    scene = Scene(name="glass_sculpture", camera=cam, background_color=Color.from_hex("#200505"))

    # Central Red Glass Sphere
    center_sphere = Sphere(center=np.array([0.0, 0.8, 0.0]), radius=0.8, name="RedOrb")
    # Red transmission color: Light passing through will turn red
    mat_red_glass = PBRMaterial.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 0.0, 0.2), # Transmission Color
        roughness=0.02, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["glass"], 
        transmission=1.0
    )
    center_sphere.material = mat_red_glass
    scene.add_object(VObject(shape=center_sphere, name="RedOrb"))

    # Encasing Glass Cube (Clear)
    outer_box = Cube(center=np.array([0.0, 0.8, 0.0]), side_length=2.0, name="ClearBox")
    mat_clear = PBRMaterial.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=1.1, # Low IOR to look like thin plastic or aerogel
        transmission=0.9
    )
    outer_box.material = mat_clear
    scene.add_object(VObject(shape=outer_box, name="ClearBox"))

    # Back Mirror to reflect the back of the glass objects
    mirror = Cube(center=np.array([0.0, 2.0, 3.0]), side_length=6.0, name="MirrorBack")
    mirror.transform.scale = np.array([1.0, 1.0, 0.1])
    mirror.material = PBRMaterial.create_specular(Color(1.0, 1.0, 1.0), roughness=0.0, metallicness=1.0)
    scene.add_object(VObject(shape=mirror, name="MirrorBack"))

    # Lights
    # Cyan light to contrast with red glass
    scene.add_light(LightSource(np.array([4.0, 4.0, -4.0]), Color.from_hex("#00FFFF"), 15.0, name="CyanKey"))
    # White rim
    scene.add_light(LightSource(np.array([-4.0, 1.0, 0.0]), Color.from_hex("#FFFFFF"), 5.0, name="Rim"))

    return scene

def get_100_spheres_grid_scene(width: int = 128, height: int = 128) -> Scene:
    """
    Stress test scene: Generates a 10x10 grid of spheres with varying materials.
    Total objects: 100 spheres + 1 floor = 101 objects.
    """
    cam_transform = Transform(np.array([-8.0, 8.0, -8.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=200.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="100_spheres_grid", camera=cam, background_color=Color.from_hex("#1A1A1A"))

    # Grid settings
    rows = 10
    cols = 10
    spacing = 1.5
    offset_x = -((rows - 1) * spacing) / 2
    offset_z = -((cols - 1) * spacing) / 2

    # Loop to create 100 spheres
    for r in range(rows):
        for c in range(cols):
            x = offset_x + (r * spacing)
            z = offset_z + (c * spacing)
            
            # Create a wave pattern for height (y) using sine/cosine
            y = 0.5 + 0.5 * np.sin(r * 0.5) * np.cos(c * 0.5)
            
            pos = np.array([x, y, z])
            
            # Vary material based on position
            # Even rows: Metallic, Odd rows: Matte
            # Color shifts across the grid
            
            # Calculate color factor (0.0 to 1.0)
            factor = (r * cols + c) / 100.0
            
            color = Color(
                r / rows,      # Red increases with Row
                0.5,           # Green constant
                c / cols       # Blue increases with Col
            )
            
            name = f"S_{r}_{c}"
            shape = Sphere(center=pos, radius=0.4, name=name)
            
            if (r + c) % 2 == 0:
                # Shiny Metal
                mat = PBRMaterial.create_specular(color, roughness=0.1, metallicness=0.9)
            else:
                # Rough Diffuse
                mat = PBRMaterial.create_diffuse(color, roughness=0.8)
                
            shape.material = mat
            scene.add_object(VObject(shape=shape, name=name))

    # Floor
    floor = Cube(center=np.array([0.0, -2.0, 0.0]), side_length=50.0, name="Floor")
    floor.transform.scale = np.array([1.0, 0.1, 1.0])
    floor.material = PBRMaterial.create_diffuse(Color.from_hex("#333333"), roughness=0.5)
    scene.add_object(VObject(shape=floor, name="FloorObj"))

    # Single Bright Sun Light to cast many shadows
    sun = LightSource(position=np.array([10.0, 20.0, -10.0]), color=Color(1.0, 1.0, 0.9), intensity=50.0, radius=100.0, name="Sun")
    scene.add_light(sun)
    
    cam.transform.look_at(np.array([0, 0, 0]))

    return scene

def get_low_ior_scene(width: int = 120, height: int = 120) -> Scene:
    """
    Features a sphere with an IOR < 1.0 (0.8).
    This acts like an 'air bubble in glass' but inverted, pushing light rays 
    away from the normal rather than towards it upon entry.
    """
    cam_transform = Transform(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = VCamera(
        cam_transform, fov=60.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="low_ior_anomaly", camera=cam, background_color=Color.from_hex("#000000"))

    # 1. The Low IOR Sphere (IOR 0.8)
    # This will bend light 'outward', making the center look magnified/distorted 
    # differently than standard glass.
    anomaly = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=1.2, name="AnomalySphere")
    mat_low_ior = PBRMaterial.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(0.8, 1.0, 0.9), # Slight cyan tint to internal transmission
        roughness=0.0, 
        metallicness=0.0, 
        ior=0.8,              # < 1.0 creates the unique divergence effect
        transmission=1.0
    )
    anomaly.material = mat_low_ior
    scene.add_object(VObject(shape=anomaly, name="AnomalyObj"))

    # 2. Background Grid (To visualize the distortion)
    # We place a checkerboard of cubes behind the sphere.
    for x in range(-3, 4):
        for y in range(-3, 4):
            if (x + y) % 2 == 0:
                bg_cube = Cube(center=np.array([x * 1.5, y * 1.5, 4.0]), side_length=1.0, name=f"Bg_{x}_{y}#a")
                bg_cube.material = PBRMaterial.create_emissive(Color.from_hex("#FF4444"), 2.0)
            else:
                bg_cube = Cube(center=np.array([x * 1.5, y * 1.5, 4.0]), side_length=1.0, name=f"Bg_{x}_{y}#b")
                bg_cube.material = PBRMaterial.create_emissive(Color.from_hex("#4444FF"), 2.0)
            
            # Scale them to be flat tiles
            bg_cube.transform.scale = np.array([1.0, 1.0, 0.1])
            scene.add_object(VObject(shape=bg_cube, name=f"Tile_{x}_{y}"))

    # Lighting (Standard setup, though the emissive background does most of the work)
    scene.add_light(LightSource(np.array([2.0, 2.0, -3.0]), Color(1.0, 1.0, 1.0), 10.0, name="Front"))

    return scene