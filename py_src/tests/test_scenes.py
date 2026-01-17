import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

sys.path.insert(0, current_dir)

from src.Data.Transform import Transform
from src.Data.Color import Color, ColorGradient
from src.Geometry.Core import *
from src.Geometry.Primitive import *
from src.Geometry.Mesh import *
from src.Data.Scene import Scene
from src.Data.Camera import Camera, CameraType
from src.Lighting.Core import LightSource, LightType
from src.Lighting.Optics import REFRACTIVE_INDICES
from src.Material.Factory import MaterialFactory

def get_gradient_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.5, -4.0]), np.array([0, 0.2, 0]), np.ones(3))
    cam = Camera(
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
    sky_positions = np.array([0.0, 0.4, 0.45, 1.0])

    scene = Scene("gradient_scene", cam, background_color=ColorGradient(sky_colors, sky_positions))

    # Primary Key Light (Sharp, slightly yellow, placed high and to the left for side lighting)
    key_light = LightSource(position=np.array([4.0, 5.0, 0.0]), color=Color.from_hex("#FFEDC7"), intensity=15.0, radius=0.5, name="Key Light")
    scene.add_light(key_light)

    # Soft Fill Light (Simulates general ambient light or bounce light)
    fill_light = LightSource(position=np.array([-5.0, 2.0, -5.0]), color=Color.from_hex("#C7E5FF"), intensity=3.0, radius=4, name="Fill Light")
    scene.add_light(fill_light)

    # Main Sphere (Mid-Ground): Highly Reflective Metal
    mat_metal = MaterialFactory.create_specular(Color.from_hex("#47505C"), 0.2, 0.9, 1.0, 1.0)
    sph_1 = Primitive("ReflectiveSphere", Transform(np.array([0.0, 2.25, 5.0])), Sphere(), mat_metal)
    cam.transform.look_at(sph_1.transform.position, np.array([0, 1, 0]))
    scene.add_object(sph_1)

    # Additional Object 1: Cube (Background/Visual Anchor) - Matte and Rough
    mat_matte = MaterialFactory.create_diffuse(Color.from_hex("#C27A23"), 0.8)
    bx_1 = Primitive("MatteBoxObject", shape=Cube(size=2.5), material=mat_matte)
    bx_1.transform.rotate(np.deg2rad(15), np.array([0, 1, 0]))
    scene.add_object(bx_1)
    
    # Additional Object 2: Small Emissive Sphere (Light Source Helper) - Floating in air
    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#EE1717"), 2)
    sph_2 = Primitive("EmissiveOrbObject", Transform(np.array([-0.5, 2.5, 1.5])), Sphere(), mat_glow)
    scene.add_object(sph_2)

    return scene

def get_minimal_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.0, -5.0]), np.zeros(3), np.ones(3))
    cam = Camera(
        cam_transform,
        fov=60.0, near=0.1, far=1000.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("minimal_scene", cam, background_color=Color.from_hex("#3A4655"))

    # Sphere at origin
    mat = MaterialFactory.create_diffuse(Color.from_hex("#227DD7"), 0.2)
    v_sphere = Primitive("SphereMin", shape=Sphere(), material=mat)
    scene.add_object(v_sphere)

    # Ground
    matg = MaterialFactory.create_diffuse(Color.from_hex("#3F3F3F"), 0.9)
    v_ground = Primitive("GroundMin", Transform(np.array([0.0, -100.5, 0.0])), Sphere(radius=100), matg)
    scene.add_object(v_ground)

    # Single light
    light = LightSource(position=np.array([2.0, 3.0, -1.0]), color=Color.from_hex("#FFFFFF"), intensity=300.0, radius=2, name="SunMin")
    scene.add_light(light)

    return scene

def get_emissive_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -3.5]), np.zeros(3), np.ones(3))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene(name="emissive_scene", camera=cam, background_color=Color.from_hex("#000000"))

    # Emissive sphere
    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#FFEA62"), 1.2)
    v_emissive = Primitive("GlowingSphere", Transform(np.array([0.8, 1.0, 0.0])), Sphere(radius=0.3), mat_glow)
    scene.add_object(v_emissive)

    # Reflective sphere
    mat_reflect = MaterialFactory.create_specular(Color.from_hex("#6B6666"), roughness=0.1, metallicness=0.5, specular_intensity=1.0, specular_tint_amount=1.0)
    v_mirror = Primitive("MirrorSphere", Transform(np.array([-0.5, 0.5, 0.0])), Sphere(radius=0.5), mat_reflect)
    scene.add_object(v_mirror)

    # Ground
    matg = MaterialFactory.create_diffuse(Color.from_hex("#202020"), roughness=0.8)
    v_ground = Primitive("Ground", Transform(np.array([0.0, -100.5, 0.0])), Sphere(radius=100), matg)
    scene.add_object(v_ground)

    # Small ambient fill light
    fill = LightSource(position=np.array([-4.0, 2.0, -3.0]), color=Color.from_hex("#AAAACC"), intensity=25.0, radius=10.0, name="FillEmiss")
    scene.add_light(fill)

    return scene

def get_lit_studio_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -4.0]), np.array([-0.15, 0.0, 0.0]), np.ones(3))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene(name="lit_studio", camera=cam, background_color=Color.from_hex("#BEC2CF"))

    # Objects: two spheres and box as background
    mat1 = MaterialFactory.create_specular(Color.from_hex("#FFB86B"), 0.2, 0.1, 0.9, 0)
    v_s1 = Primitive("StudioBallA", Transform(np.array([-0.6, 0.4, 0.5]), scale=np.ones(3) * 0.4), Sphere(), mat1)
    scene.add_object(v_s1)

    mat2 = MaterialFactory.create_specular(Color.from_hex("#6B9BFF"), 0.2, 0.4, 0.9, 0)
    v_s2 = Primitive("StudioBallB", Transform(np.array([0.8, 0.45, 0.2]), scale=np.ones(3) * 0.45), Sphere(), mat2)
    scene.add_object(v_s2)

    # Background
    mat_plane = MaterialFactory.create_diffuse(Color.from_hex("#C1CBD0"), roughness=1.0)
    v_plane = Primitive("StudioBack", Transform(np.array([0.0, 0.5, 2.0])), Plane(), mat_plane)
    v_plane.transform.rotate(np.deg2rad(90), np.array([1.0, 0.0, 0.0]))
    scene.add_object(v_plane)

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
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    # Materials
    mat_white = MaterialFactory.create_diffuse(Color.from_hex("#E0E0E0"), 1.0)
    mat_red   = MaterialFactory.create_diffuse(Color.from_hex("#B03030"), 1.0)
    mat_green = MaterialFactory.create_diffuse(Color.from_hex("#30B030"), 1.0)
    mat_blue = MaterialFactory.create_diffuse(Color.from_hex("#3036B0"), 1.0)
    mat_mirror = MaterialFactory.create_specular(Color.from_hex("#FFFFFF"), 0.1, 1.0, 0)
    mat_glass  = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["glass"], 0)

    scene = Scene(name="rgb_cornell_box", camera=cam, background_color=Color(0.0, 0.0, 0.0))

    # Floor
    floor_shape = Cube(side_length=10.0, name="Floor")
    v_floor = Primitive(shape=floor_shape, name="Floor")
    v_floor.material = mat_white
    v_floor.transform.translate(np.array([0.0, -0.5, 0.0]))
    v_floor.transform.enlarge(np.array([2.0, 0.1, 2.0]))
    scene.add_object(v_floor)
    
    # Ceiling
    ceiling_shape = Cube(side_length=10.0, name="Ceiling")
    v_ceiling = Primitive(shape=ceiling_shape, name="Ceiling")
    v_ceiling.material = mat_white
    v_ceiling.transform.translate(np.array([0.0, 6.5, 0.0]))
    v_ceiling.transform.enlarge(np.array([2.0, 0.1, 2.0]))
    scene.add_object(v_ceiling)

    # Back Wall
    back_shape = Cube(side_length=10.0, name="BackWall")
    v_back = Primitive(shape=back_shape, name="BackWall")
    v_back.material = mat_blue
    v_back.transform.translate(np.array([0.0, 3.0, 5.5]))
    v_back.transform.enlarge(np.array([2.0, 3.0, 0.1]))
    scene.add_object(v_back)
    
    # Left Wall (Red)
    left_shape = Cube(side_length=10.0, name="LeftWall")
    v_left = Primitive(shape=left_shape, name="LeftWall")
    v_left.material = mat_red
    v_left.transform.translate(np.array([-5.5, 3.0, 0.0]))
    v_left.transform.enlarge(np.array([0.1, 3.0, 2.0]))
    scene.add_object(v_left)

    # Right Wall (Green)
    right_shape = Cube(side_length=10.0, name="RightWall")
    v_right = Primitive(shape=right_shape, name="RightWall")
    v_right.material = mat_green
    v_right.transform.translate(np.array([5.5, 3.0, 0.0]))
    v_right.transform.enlarge(np.array([0.1, 3.0, 2.0]))
    scene.add_object(v_right)
    
    # Tall Box (Rotated)
    tall_box_shape = Cube(side_length=3.0, name="TallBox")
    v_tall_box = Primitive(shape=tall_box_shape, name="TallBox")
    v_tall_box.material = mat_white
    v_tall_box.transform.translate(np.array([-2.0, 1.5, 2.0]))
    v_tall_box.transform.enlarge(np.array([0.6, 1.0, 0.6]))
    v_tall_box.transform.rotate(20.0, np.array([0.0, 1.0, 0.0]))
    scene.add_object(v_tall_box)
    
    # Sphere (Mirror)
    mirror_sphere_shape = Sphere(radius=1.25, name="MirrorBall")
    v_mirror_sphere = Primitive(shape=mirror_sphere_shape, name="MirrorBall")
    v_mirror_sphere.material = mat_mirror
    v_mirror_sphere.transform.translate(np.array([2.0, 1.25, 3.0]))
    scene.add_object(v_mirror_sphere)
    
    # Small Cube (Glass/Crystal in front)
    glass_cube_shape = Cube(side_length=1.5, name="GlassCube")
    v_glass_cube = Primitive(shape=glass_cube_shape, name="GlassCube")
    v_glass_cube.material = mat_glass
    v_glass_cube.transform.translate(np.array([0.0, 0.75, -2.0]))
    v_glass_cube.transform.rotate(-15.0, np.array([0.0, 1.0, 0.0]))
    scene.add_object(v_glass_cube)

    # Lighting
    ceiling_light = LightSource(
        position=np.array([0.0, 5.8, 0.0]), 
        color=Color.from_hex("#FFECDE"), 
        intensity=25.0, 
        radius=5, 
        name="CeilingLight"
    )
    scene.add_light(ceiling_light)

    cam.transform.look_at(np.array([0.0, 2.5, 0.0]))

    return scene

def get_cyberpunk_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -4.0]), np.array([-0.1, 0.0, 0.0]), np.ones(3))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    sky_colors = [
        Color.from_hex("#050008"),
        Color.from_hex("#0B1333"),
    ]
    sky_positions = np.array([0.0, 1.0])
    scene = Scene(name="cyberpunk_street", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    # Road
    road_shape = Cube(side_length=20.0, name="WetRoad")
    mat_wet = MaterialFactory.create_diffuse(Color.from_hex("#151515"), roughness=0.2)
    v_road = Primitive(shape=road_shape, name="Road")
    v_road.material = mat_wet
    v_road.transform.translate(np.array([0.0, -1.0, 0.0]))
    v_road.transform.enlarge(np.array([1.0, 0.1, 2.0]))
    scene.add_object(v_road)

    # Hero Object: Chrome Sphere
    hero_shape = Sphere(radius=0.8, name="ChromeHero")
    mat_chrome = MaterialFactory.create_specular(Color.from_hex("#313238"), roughness=0.2, metallicness=1.0)
    v_hero = Primitive(shape=hero_shape, name="HeroSphere")
    v_hero.material = mat_chrome
    v_hero.transform.translate(np.array([0.0, 0.5, 0.0]))
    scene.add_object(v_hero)

    # Background Buildings
    bldg_left_shape = Cube(side_length=4.0, name="BuildingLeft")
    v_bldg_left = Primitive(shape=bldg_left_shape, name="BldgLeft")
    v_bldg_left.material = MaterialFactory.create_diffuse(Color.from_hex("#4DBC3E"), roughness=0.9)
    v_bldg_left.transform.translate(np.array([-2.5, 2.0, 2.0]))
    v_bldg_left.transform.enlarge(np.array([0.5, 2.0, 0.5]))
    scene.add_object(v_bldg_left)

    bldg_right_shape = Cube(side_length=4.0, name="BuildingRight")
    v_bldg_right = Primitive(shape=bldg_right_shape, name="BldgRight")
    v_bldg_right.material = MaterialFactory.create_diffuse(Color.from_hex("#E28335"), roughness=0.9)
    v_bldg_right.transform.translate(np.array([2.5, 1.3, 2.2]))
    v_bldg_right.transform.enlarge(np.array([0.65, 1.3, 0.65]))
    scene.add_object(v_bldg_right)

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
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="material_deck", camera=cam, background_color=Color.from_hex("#000000"))

    # Floor
    floor_shape = Cube(side_length=15.0, name="Floor")
    v_floor = Primitive(shape=floor_shape, name="Floor")
    v_floor.material = MaterialFactory.create_diffuse(Color.from_hex("#CCCCCC"), roughness=1.0)
    v_floor.transform.translate(np.array([0.0, -1.0, 0.0]))
    v_floor.transform.enlarge(np.array([2.0, 0.1, 1.0]))
    scene.add_object(v_floor)

    base_col = Color.from_hex("#D4AF37")
    
    # Spheres with varying roughness
    s1_shape = Sphere(radius=0.6, name="Gold_0.0")
    v_s1 = Primitive(shape=s1_shape, name="S_Mirror")
    v_s1.material = MaterialFactory.create_specular(base_col, roughness=0.0)
    v_s1.transform.translate(np.array([-3.0, 0.5, 0.0]))
    scene.add_object(v_s1)

    s2_shape = Sphere(radius=0.6, name="Gold_0.25")
    v_s2 = Primitive(shape=s2_shape, name="S_Brushed")
    v_s2.material = MaterialFactory.create_specular(base_col, roughness=0.25)
    v_s2.transform.translate(np.array([-1.5, 0.5, 0.0]))
    scene.add_object(v_s2)

    s3_shape = Sphere(radius=0.6, name="Gold_0.5")
    v_s3 = Primitive(shape=s3_shape, name="S_Rough")
    v_s3.material = MaterialFactory.create_specular(base_col, roughness=0.5)
    v_s3.transform.translate(np.array([0.0, 0.5, 0.0]))
    scene.add_object(v_s3)

    s4_shape = Sphere(radius=0.6, name="Gold_0.75")
    v_s4 = Primitive(shape=s4_shape, name="S_Matte")
    v_s4.material = MaterialFactory.create_specular(base_col, roughness=0.75)
    v_s4.transform.translate(np.array([1.5, 0.5, 0.0]))
    scene.add_object(v_s4)
    
    s5_shape = Sphere(radius=0.6, name="Plastic_Red")
    v_s5 = Primitive(shape=s5_shape, name="S_Plastic")
    v_s5.material = MaterialFactory.create_diffuse(Color.from_hex("#FF0000"), roughness=0.1)
    v_s5.transform.translate(np.array([3.0, 0.5, 0.0]))
    scene.add_object(v_s5)

    # Lighting
    scene.add_light(LightSource(position=np.array([0.0, 5.0, -5.0]), color=Color(1.0, 1.0, 1.0), intensity=15.0, name="Main"))
    scene.add_light(LightSource(position=np.array([5.0, 2.0, -2.0]), color=Color(0.8, 0.8, 1.0), intensity=10.0, name="Fill"))

    cam.transform.look_at(v_s3.transform.position)
    return scene

def get_refraction_lab_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.0, -4.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="refraction_lab", camera=cam, background_color=Color(0.05, 0.05, 0.05))

    # Striped Background Wall
    wall_shape = Cube(side_length=8.0, name="StripedWall")
    v_wall = Primitive(shape=wall_shape, name="BackWall")
    v_wall.material = MaterialFactory.create_emissive(Color(1.0, 1.0, 1.0), 1.0)
    v_wall.transform.translate(np.array([0.0, 2.0, 4.0]))
    v_wall.transform.enlarge(np.array([1.5, 1.0, 0.1]))
    scene.add_object(v_wall)

    # Blocker bars
    for i in range(-6, 7):
        bar_shape = Cube(side_length=5.0, name=f"Bar_{i}")
        v_bar = Primitive(shape=bar_shape, name=f"Bar_{6 + i}")
        v_bar.material = MaterialFactory.create_diffuse(Color(0.0, 0.0, 0.0), 1.0)
        v_bar.transform.translate(np.array([i, 2.0, 3.5]))
        v_bar.transform.enlarge(np.array([0.1, 4.0, 0.1]))
        scene.add_object(v_bar)

    # Glass Sphere (IOR 1.5)
    s_glass_shape = Sphere(radius=0.6, name="Acrylic")
    v_s_glass = Primitive(shape=s_glass_shape, name="AcrylicSphere")
    v_s_glass.material = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["acrylic"], 0)
    v_s_glass.transform.translate(np.array([-1.2, 0.5, 0.0]))
    scene.add_object(v_s_glass)

    # Diamond Sphere (IOR 2.4)
    s_diamond_shape = Sphere(radius=0.6, name="Diamond")
    v_s_diamond = Primitive(shape=s_diamond_shape, name="DiamondSphere")
    v_s_diamond.material = MaterialFactory.create_glass(Color.from_hex("#B9D3E3"), Color(0.9, 0.9, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["diamond"], 0.2)
    v_s_diamond.transform.translate(np.array([0.0, 0.5, 0.0]))
    scene.add_object(v_s_diamond)

    # Water Sphere / Bubble (IOR 1.33)
    s_water_shape = Sphere(radius=0.6, name="Water")
    v_s_water = Primitive(shape=s_water_shape, name="WaterSphere")
    v_s_water.material = MaterialFactory.create_glass(Color.from_hex("#A6ADD5"), Color.from_hex("#1F1FFF"), 0.0, 0.0, REFRACTIVE_INDICES["water"], 0.1)
    v_s_water.transform.translate(np.array([1.2, 0.5, 0.0]))
    scene.add_object(v_s_water)

    # Lighting
    scene.add_light(LightSource(position=np.array([2.0, 3.0, -3.0]), color=Color(1.0, 1.0, 1.0), intensity=15.0, name="FrontLight"))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))
    return scene

def get_scifi_corridor_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A high-contrast scene featuring repetitive metallic geometry and emissive lighting.
    Focuses on reflections of light sources on rough metal.
    """
    cam_transform = Transform(np.array([0.0, 1.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = Camera(
        cam_transform, fov=80.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="scifi_corridor", camera=cam, background_color=Color.from_hex("#020205"))

    # Materials
    mat_floor = MaterialFactory.create_specular(Color.from_hex("#2A2A2A"), roughness=0.3, metallicness=0.8)
    mat_pillar = MaterialFactory.create_specular(Color.from_hex("#111111"), roughness=0.5, metallicness=0.9)
    mat_light_strip = MaterialFactory.create_emissive(Color.from_hex("#00FFFF"), 5.0)

    # Floor
    floor_shape = Cube(side_length=20.0, name="CorridorFloor")
    v_floor = Primitive(shape=floor_shape, name="Floor")
    v_floor.material = mat_floor
    v_floor.transform.translate(np.array([0.0, -1.0, -10.0]))
    v_floor.transform.enlarge(np.array([0.5, 0.1, 4.0]))
    scene.add_object(v_floor)

    # Ceiling
    ceiling_shape = Cube(side_length=20.0, name="CorridorCeiling")
    v_ceiling = Primitive(shape=ceiling_shape, name="Ceiling")
    v_ceiling.material = mat_floor
    v_ceiling.transform.translate(np.array([0.0, 3.0, -10.0]))
    v_ceiling.transform.enlarge(np.array([0.5, 0.1, 4.0]))
    scene.add_object(v_ceiling)

    # Repetitive Pillars and Lights
    for z in range(0, -20, -4):
        # Left Pillar
        p_left_shape = Cube(side_length=2.0, name=f"PillarL_{z}")
        v_p_left = Primitive(shape=p_left_shape, name=f"PillarLeft_{z}")
        v_p_left.material = mat_pillar
        v_p_left.transform.translate(np.array([-2.5, 1.0, z]))
        v_p_left.transform.enlarge(np.array([0.5, 2.0, 0.5]))
        scene.add_object(v_p_left)

        # Right Pillar
        p_right_shape = Cube(side_length=2.0, name=f"PillarR_{z}")
        v_p_right = Primitive(shape=p_right_shape, name=f"PillarRight_{z}")
        v_p_right.material = mat_pillar
        v_p_right.transform.translate(np.array([2.5, 1.0, z]))
        v_p_right.transform.enlarge(np.array([0.5, 2.0, 0.5]))
        scene.add_object(v_p_right)

        # Emissive Light Strips on floor edges
        l_strip_shape = Cube(side_length=0.2, name=f"Light_{z}")
        v_l_strip = Primitive(shape=l_strip_shape, name=f"Strip_{z}")
        v_l_strip.material = mat_light_strip
        v_l_strip.transform.translate(np.array([0.0, -0.9, z]))
        v_l_strip.transform.enlarge(np.array([8.0, 0.1, 0.5]))
        scene.add_object(v_l_strip)

        # Actual Light Sources corresponding to strips
        light = LightSource(position=np.array([0.0, 0.5, z]), color=Color.from_hex("#00AAAA"), intensity=5.0, radius=2.0, name=f"PointLight_{z}")
        scene.add_light(light)

    # End focal point
    sphere_end_shape = Sphere(radius=1.5, name="EndSphere")
    v_sphere_end = Primitive(shape=sphere_end_shape, name="EndSphere")
    v_sphere_end.material = MaterialFactory.create_specular(Color.from_hex("#FF0000"), roughness=0.1, metallicness=1.0)
    v_sphere_end.transform.translate(np.array([0.0, 1.0, -18.0]))
    scene.add_object(v_sphere_end)

    cam.transform.look_at(np.array([0, 1, -20]))
    return scene

def get_sunset_monolith_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A scene focusing on warm lighting, long shadows, and the contrast between
    a matte organic ground and a sharp, reflective geometric object.
    """
    cam_transform = Transform(np.array([3.0, 1.5, -4.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = Camera(
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
    sky_positions = np.array([0.0, 0.3, 0.5, 1.0])
    
    scene = Scene(name="sunset_monolith", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    # The Monolith (Highly Specular Black Metal)
    monolith_shape = Cube(side_length=4.0, name="Monolith")
    mat_mono = MaterialFactory.create_specular(Color.from_hex("#050505"), roughness=0.05, metallicness=1.0)
    v_monolith = Primitive(shape=monolith_shape, name="MonolithObj")
    v_monolith.material = mat_mono
    v_monolith.transform.translate(np.array([0.0, 2.0, 0.0]))
    v_monolith.transform.enlarge(np.array([0.4, 2.0, 0.4]))
    v_monolith.transform.rotate(np.deg2rad(25), np.array([0, 1, 0]))
    scene.add_object(v_monolith)

    # Sand Dunes (Matte, rough)
    floor_shape = Sphere(radius=50.0, name="Sand")
    mat_sand = MaterialFactory.create_diffuse(Color.from_hex("#D6783B"), roughness=1.0)
    v_floor = Primitive(shape=floor_shape, name="SandGround")
    v_floor.material = mat_sand
    v_floor.transform.translate(np.array([0.0, -51.0, 0.0]))
    scene.add_object(v_floor)

    # Floating particles/smaller rocks
    rock1_shape = Sphere(radius=0.3, name="Rock1")
    v_rock1 = Primitive(shape=rock1_shape, name="Rock1")
    v_rock1.material = MaterialFactory.create_diffuse(Color.from_hex("#554433"), roughness=0.9)
    v_rock1.transform.translate(np.array([-1.5, 0.3, 1.5]))
    scene.add_object(v_rock1)

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
    cam = Camera(
        cam_transform, fov=60.0, near=0.1, far=50.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )

    scene = Scene(name="pastel_blocks", camera=cam, background_color=Color.from_hex("#F0F4F8"))

    # Materials (Plastic/Chalky: Diffuse with very low specular or high roughness)
    mat_pink = MaterialFactory.create_diffuse(Color.from_hex("#FFB7B2"), roughness=0.6)
    mat_mint = MaterialFactory.create_diffuse(Color.from_hex("#B5EAD7"), roughness=0.6)
    mat_purple = MaterialFactory.create_diffuse(Color.from_hex("#E2F0CB"), roughness=0.6) # Actually yellowish-green
    mat_white = MaterialFactory.create_diffuse(Color.from_hex("#FFFFFF"), roughness=0.9)

    # Floor
    floor_shape = Cube(side_length=10.0, name="WhiteFloor")
    v_floor = Primitive(shape=floor_shape, name="Floor")
    v_floor.material = mat_white
    v_floor.transform.translate(np.array([0.0, -1.0, 0.0]))
    v_floor.transform.enlarge(np.array([2.0, 0.1, 2.0]))
    scene.add_object(v_floor)

    # Stacked Objects
    # Base Cube
    base_shape = Cube(side_length=2.0, name="BaseCube")
    v_base = Primitive(shape=base_shape, name="BaseObj")
    v_base.material = mat_mint
    v_base.transform.translate(np.array([0.0, 0.0, 0.0]))
    v_base.transform.rotate(np.deg2rad(15), np.array([0, 1, 0]))
    scene.add_object(v_base)

    # Middle Cylinder (Simulated by stretched sphere or cube? using Sphere for variety)
    mid_shape = Sphere(radius=0.8, name="MidSphere")
    v_mid = Primitive(shape=mid_shape, name="MidObj")
    v_mid.material = mat_pink
    v_mid.transform.translate(np.array([0.0, 1.6, 0.0]))
    scene.add_object(v_mid)

    # Top floating cube
    top_shape = Cube(side_length=1.0, name="TopCube")
    v_top = Primitive(shape=top_shape, name="TopObj")
    v_top.material = mat_purple
    v_top.transform.translate(np.array([0.2, 2.8, 0.2]))
    v_top.transform.rotate(np.deg2rad(45), np.array([1, 1, 0]))
    scene.add_object(v_top)

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
    cam = Camera(cam_transform, fov=70.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="glass_prism_row", camera=cam, background_color=Color.from_hex("#101015"))

    # 1. Diamond Sphere (High IOR: 2.42) - Center
    # High dispersion and internal reflection
    sphere_diamond_shape = Sphere(radius=0.6, name="Diamond")
    mat_diamond = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["diamond"], 
        transmission=1.0
    )
    v_sphere_diamond = Primitive(shape=sphere_diamond_shape, name="CenterDiamond")
    v_sphere_diamond.material = mat_diamond
    v_sphere_diamond.transform.translate(np.array([0.0, 0.5, 0.0]))
    scene.add_object(v_sphere_diamond)

    # 2. Water Sphere (Low IOR: 1.33) - Left
    # Subtle bending, looks more transparent
    sphere_water_shape = Sphere(radius=0.6, name="Water")
    mat_water = MaterialFactory.create_glass(
        Color(0.9, 0.9, 1.0), 
        Color(0.8, 0.9, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=1.33, 
        transmission=1.0
    )
    v_sphere_water = Primitive(shape=sphere_water_shape, name="LeftWater")
    v_sphere_water.material = mat_water
    v_sphere_water.transform.translate(np.array([-1.5, 0.5, 0.0]))
    scene.add_object(v_sphere_water)

    # 3. Heavy Flint Glass Cube (Medium-High IOR: 1.65) - Right
    cube_glass_shape = Cube(side_length=1.2, name="FlintGlass")
    mat_flint = MaterialFactory.create_glass(
        Color(1.0, 0.9, 0.9), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.01, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["glass_flint_heavy"], 
        transmission=1.0
    )
    v_cube_glass = Primitive(shape=cube_glass_shape, name="RightFlint")
    v_cube_glass.material = mat_flint
    v_cube_glass.transform.translate(np.array([1.5, 0.5, 0.0]))
    # Rotate to show refraction through edges
    v_cube_glass.transform.rotate(np.deg2rad(30), np.array([0, 1, 0]))
    v_cube_glass.transform.rotate(np.deg2rad(10), np.array([1, 0, 0]))
    scene.add_object(v_cube_glass)

    # Checkerboard Floor (to make refraction obvious)
    floor_shape = Cube(side_length=20.0, name="Floor")
    # Using a striped emissive material to create lines visible THROUGH the glass
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#888888"), roughness=0.8)
    v_floor = Primitive(shape=floor_shape, name="FloorBase")
    v_floor.material = mat_floor
    v_floor.transform.translate(np.array([0.0, -10.5, 5.0]))
    v_floor.transform.enlarge(np.array([1.0, 1.0, 0.5])) # Flatten
    scene.add_object(v_floor)

    # Striped Wall behind objects
    for i in range(-5, 6):
        bar_shape = Cube(side_length=0.5, name=f"Strip_{i}")
        # Alternating colors
        col = Color.from_hex("#FF0000") if i % 2 == 0 else Color.from_hex("#FFFFFF")
        v_bar = Primitive(shape=bar_shape, name=f"Bar_{i}")
        v_bar.material = MaterialFactory.create_emissive(col, 2.0)
        v_bar.transform.translate(np.array([i, 2.0, 4.0]))
        v_bar.transform.enlarge(np.array([0.5, 8.0, 0.1]))
        scene.add_object(v_bar)

    # Light
    scene.add_light(LightSource(position=np.array([0.0, 5.0, -3.0]), color=Color(1.0, 1.0, 1.0), intensity=20.0, name="TopLight"))

    return scene

def get_glass_sculpture_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A complex arrangement of overlapping glass plates and spheres.
    Good for testing recursion depth and transmission color absorption.
    """
    cam_transform = Transform(np.array([3.0, 2.5, -3.0]), np.array([-0.3, 0.7, 0.0]), np.ones(3))
    cam = Camera(cam_transform, fov=60.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    cam.transform.look_at(np.array([0, 0.5, 0]))
    
    scene = Scene(name="glass_sculpture", camera=cam, background_color=Color.from_hex("#200505"))

    # Central Red Glass Sphere
    center_sphere_shape = Sphere(radius=0.8, name="RedOrb")
    # Red transmission color: Light passing through will turn red
    mat_red_glass = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 0.0, 0.2), # Transmission Color
        roughness=0.02, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["glass"], 
        transmission=1.0
    )
    v_center_sphere = Primitive(shape=center_sphere_shape, name="RedOrb")
    v_center_sphere.material = mat_red_glass
    v_center_sphere.transform.translate(np.array([0.0, 0.8, 0.0]))
    scene.add_object(v_center_sphere)

    # Encasing Glass Cube (Clear)
    outer_box_shape = Cube(side_length=2.0, name="ClearBox")
    mat_clear = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=1.1, # Low IOR to look like thin plastic or aerogel
        transmission=0.9
    )
    v_outer_box = Primitive(shape=outer_box_shape, name="ClearBox")
    v_outer_box.material = mat_clear
    v_outer_box.transform.translate(np.array([0.0, 0.8, 0.0]))
    scene.add_object(v_outer_box)

    # Back Mirror to reflect the back of the glass objects
    mirror_shape = Cube(side_length=6.0, name="MirrorBack")
    v_mirror = Primitive(shape=mirror_shape, name="MirrorBack")
    v_mirror.material = MaterialFactory.create_specular(Color(1.0, 1.0, 1.0), roughness=0.0, metallicness=1.0)
    v_mirror.transform.translate(np.array([0.0, 2.0, 3.0]))
    v_mirror.transform.enlarge(np.array([1.0, 1.0, 0.1]))
    scene.add_object(v_mirror)

    # Lights
    # Cyan light to contrast with red glass
    scene.add_light(LightSource(position=np.array([4.0, 4.0, -4.0]), color=Color.from_hex("#00FFFF"), intensity=15.0, name="CyanKey"))
    # White rim
    scene.add_light(LightSource(position=np.array([-4.0, 1.0, 0.0]), color=Color.from_hex("#FFFFFF"), intensity=5.0, name="Rim"))

    return scene

def get_100_spheres_grid_scene(width: int = 128, height: int = 128) -> Scene:
    """
    Stress test scene: Generates a 10x10 grid of spheres with varying materials.
    Total objects: 100 spheres + 1 floor = 101 objects.
    """
    cam_transform = Transform(np.array([-8.0, 8.0, -8.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = Camera(cam_transform, fov=60.0, resolution_width=width, resolution_height=height)
    
    sky_colors = [
        Color.from_hex("#2D2515"),
        Color.from_hex("#42424E"),
        Color.from_hex("#5B6791"),
        Color.from_hex("#87BFC6"),
    ]
    sky_positions = np.array([0.0, 0.4, 0.45, 1.0])
    
    scene = Scene(name="100_spheres_grid", camera=cam, background_color=ColorGradient(sky_colors, sky_positions))

    # Optimization: Create ONE shape instance and reuse it 100 times
    # The geometry is identical; only the position (Transform) and Material change.
    shared_sphere_shape = Sphere(radius=0.2)

    rows = 10
    cols = 10
    spacing = 1.5
    offset_x = -((rows - 1) * spacing) / 2
    offset_z = -((cols - 1) * spacing) / 2

    for r in range(rows):
        for c in range(cols):
            x = offset_x + (r * spacing)
            z = offset_z + (c * spacing)
            
            # Wave pattern height
            y = 0.5 + 0.5 * np.sin(r * 0.5) * np.cos(c * 0.5)
            
            # 1. Transform: Defines the unique position
            t_sphere = Transform(position=np.array([x, y, z]), rotation=np.zeros(3), scale=np.ones(3))
            
            # 2. Material: Varies per object
            color = Color(r / rows, 0.5, c / cols)
            
            if (r + c) % 2 == 0:
                mat = MaterialFactory.create_specular(color, roughness=0.1, metallicness=0.9)
            else:
                mat = MaterialFactory.create_diffuse(color, roughness=0.8)
            
            # 3. Primitive: Links the shared shape, unique transform, and unique material
            scene.add_object(Primitive(
                shape=shared_sphere_shape, 
                transform=t_sphere, 
                material=mat, 
                name=f"S_{r}_{c}"
            ))

    # Floor
    # We use a unit cube and scale it up. 
    # Scaled to (50, 0.1, 50) creates a large flat floor.
    floor_shape = Cube(size=1.0) 
    
    t_floor = Transform(position=np.array([0.0, -2.0, 0.0]), rotation=np.zeros(3), scale=np.ones(3))
            
    t_floor.enlarge(np.array([50.0, 0.1, 50.0]))
    
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#333333"), roughness=0.5)
    
    scene.add_object(Primitive("Floor", t_floor, floor_shape, mat_floor))

    # Light
    scene.add_light(LightSource(position=np.array([10.0, 20.0, -10.0]), color=Color(1.0, 1.0, 0.9), intensity=50.0, name="Sun"))
    
    # Ensure camera looks below the origin
    cam.transform.look_at(np.array([0, -1, 0]))

    return scene

def get_low_ior_scene(width: int = 120, height: int = 120) -> Scene:
    """
    Features a sphere with an IOR < 1.0 (0.8).
    This acts like an 'air bubble in glass' but inverted.
    """
    cam_transform = Transform(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 0.0]), np.ones(3))
    cam = Camera(cam_transform, fov=60.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="low_ior_anomaly", camera=cam, background_color=Color.from_hex("#000000"))

    # 1. The Low IOR Sphere
    anomaly_shape = Sphere(radius=1.2)
    mat_low_ior = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(0.8, 1.0, 0.9),
        roughness=0.0, 
        ior=0.8
    )
    t_anomaly = Transform(position=np.array([0.0, 0.0, 0.0]), rotation=np.zeros(3), scale=np.ones(3))
            
    
    scene.add_object(Primitive("AnomalyObj", t_anomaly, anomaly_shape, mat_low_ior))

    # 2. Background Grid
    # Reuse a single cube shape for all tiles
    tile_shape = Cube(size=1.0)
    
    mat_red = MaterialFactory.create_emissive(Color.from_hex("#FF4444"), 2.0)
    mat_blue = MaterialFactory.create_emissive(Color.from_hex("#4444FF"), 2.0)

    for x in range(-3, 4):
        for y in range(-3, 4):
            # Calculate position
            pos = np.array([x * 1.5, y * 1.5, 4.0])
            
            # Create Transform with Position AND Scale (flattening the cube)
            t_tile = Transform(position=pos, rotation=np.zeros(3), scale=np.ones(3))
            t_tile.enlarge(np.array([1.0, 1.0, 0.1]))
            
            # Select material
            mat = mat_red if (x + y) % 2 == 0 else mat_blue
            
            scene.add_object(Primitive(
                shape=tile_shape, 
                transform=t_tile, 
                material=mat, 
                name=f"Tile_{x}_{y}"
            ))

    # Light
    scene.add_light(LightSource(position=np.array([2.0, 2.0, -3.0]), color=Color(1.0, 1.0, 1.0), intensity=10.0, name="Front"))

    return scene