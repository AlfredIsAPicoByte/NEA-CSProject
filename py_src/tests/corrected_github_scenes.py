import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, current_dir)

from src.Data.Transform import Transform
from src.Data.Color import Color, ColorGradient
from src.Data.Scene import Scene
from src.Data.Context import Mesh_Material, SDF_Material
from src.Data.Camera import Camera, CameraType
from src.Geometry.SDF import *
from src.Geometry.Operations import *
from src.Geometry.Mesh import *
from src.Lighting.Core import Light
from src.Lighting.Optics import REFRACTIVE_INDICES
from src.Material.Factory import MaterialFactory

def get_minimal_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.0, -5.0]), np.zeros(3))
    cam = Camera(
        cam_transform,
        fov=60.0, near=0.1, far=1000.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("minimal_scene", cam, background_color=Color.from_hex("#3A4655"))

    # Sphere at origin
    mat = MaterialFactory.create_diffuse(Color.from_hex("#227DD7"), 0.2)
    scene.add_object_by_context(SDF_Material(Sphere(), mat), "Sphere Min", Transform.Identity())

    # Ground
    matg = MaterialFactory.create_diffuse(Color.from_hex("#3F3F3F"), 0.9)
    scene.add_object_by_context(SDF_Material(Sphere(), matg), "Ground Min", Transform(np.array([0.0, -100.5, 0.0]), scale=np.full(3, 100)))

    # Single light
    light = Light(color=Color.from_hex("#FFFFFF"), intensity=350.0, radius=3)
    scene.add_object_by_context(light, "Sun Min", Transform(np.array([2.0, 3.0, -1.0])))

    return scene

def get_gradient_scene(width: int = 64, height: int = 64) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.5, -4.0]), np.array([0, 0.2, 0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=100.0,
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

    # Main Sphere (Mid-Ground): Highly Reflective Metal
    mat_metal = MaterialFactory.create_specular(Color.from_hex("#47505C"), 0.2, 0.9, 1.0, 1.0)
    sph_1 = SDF_Material(Sphere(), mat_metal)
    scene.add_object_by_context(sph_1, "Reflective Sphere", Transform(np.array([0.0, 2.25, 5.0])))
    cam.transform.look_at(scene.objects[-1].transform.position, np.array([0, 1, 0]))

    # Additional Object 1: Cube (Background/Visual Anchor) - Matte and Rough
    mat_matte = MaterialFactory.create_diffuse(Color.from_hex("#C27A23"), 0.8)
    bx_1 = SDF_Material(Cube(), mat_matte)
    scene.add_object_by_context(bx_1, "Matte Box", Transform(np.array([0.0, 0.0, 0.0]), np.array([0.0, np.deg2rad(15), 0.0])))

    # Additional Object 2: Small Emissive Sphere (Light Source Helper) - Floating in air
    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#EE1717"), 2)
    sph_2 = SDF_Material(Sphere(), mat_glow)
    scene.add_object_by_context(sph_2, "Emissive Orb", Transform(np.array([-0.5, 2.5, 1.5]), scale=np.full(3, 0.4)))

    # Cylinder object
    mat_cylinder = MaterialFactory.create_specular(Color.from_hex("#FFD700"), 0.1, 0.8, 0.9, 0.5)
    cyl_1 = SDF_Material(Cylinder(), mat_cylinder)
    scene.add_object_by_context(cyl_1, "Golden Cylinder", Transform(np.array([2.0, 1.0, 3.0]), np.array([0.0, np.deg2rad(30), 0.0])))

    # Pyramid object
    mat_pyramid = MaterialFactory.create_diffuse(Color.from_hex("#8B4513"), 0.7)
    pyr_1 = SDF_Material(Pyramid(), mat_pyramid)
    scene.add_object_by_context(pyr_1, "Wooden Pyramid", Transform(np.array([-2.5, 0.5, 2.0])))

    # Lights
    key_light = Light(color=Color.from_hex("#FFEDC7"), intensity=150.0, radius=0.5)
    scene.add_object_by_context(key_light, "Key Light", Transform(np.array([4.0, 5.0, 0.0])))

    fill_light = Light(color=Color.from_hex("#C7E5FF"), intensity=500.0, radius=4)
    scene.add_object_by_context(fill_light, "Fill Light", Transform(np.array([-5.0, 2.0, -5.0])))

    return scene

def get_emissive_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -3.5]), np.zeros(3))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene(name="emissive_scene", camera=cam, background_color=Color.from_hex("#000000"))

    # Emissive sphere
    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#FFEA62"), 1.2)
    v_emissive = SDF_Material(Sphere(), mat_glow)
    scene.add_object_by_context(v_emissive, name="GlowingSphere", transform=Transform(np.zeros(3), scale=np.full(3, 0.3)))

    # Reflective sphere
    mat_reflect = MaterialFactory.create_specular(Color.from_hex("#6B6666"), roughness=0.2, metallicness=0.75, specular_intensity=1.0, specular_tint_amount=1.0)
    v_mirror = SDF_Material(Sphere(), mat_reflect)
    scene.add_object_by_context(v_mirror, name="MirrorSphere", transform=Transform.Identity())

    # Ground
    matg = MaterialFactory.create_diffuse(Color.from_hex("#202020"), roughness=0.8)
    v_ground = SDF_Material(Sphere(), matg)
    scene.add_object_by_context(v_ground, name="Ground", transform=Transform(np.array([0.0, -100.5, 0.0]), scale=np.full(3, 100)))

    # Small ambient fill light
    fill = Light(color=Color.from_hex("#AAAACC"), intensity=1000.0, radius=10.0)
    scene.add_object_by_context(fill, name="FillEmiss", transform=Transform(np.array([-4.0, 2.0, -3.0])))

    return scene

def get_lit_studio_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -4.0]), np.array([-0.15, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene(name="lit_studio", camera=cam, background_color=Color.from_hex("#BEC2CF"))

    # Objects: two spheres and box as background
    mat1 = MaterialFactory.create_specular(Color.from_hex("#FFB86B"), 0.2, 0.1, 0.9, 0)
    v_s1 = SDF_Material(Sphere(), mat1)
    scene.add_object_by_context(v_s1, name="StudioBallA", transform=Transform(np.array([-0.6, 0.4, 0.5]), scale=np.full(3, 0.4)))

    mat2 = MaterialFactory.create_specular(Color.from_hex("#6B9BFF"), 0.2, 0.4, 0.9, 0)
    v_s2 = SDF_Material(Sphere(), mat2)
    scene.add_object_by_context(v_s2, name="StudioBallB", transform=Transform(np.array([0.8, 0.45, 0.2]), scale=np.full(3, 0.45)))

    # Background
    mat_plane = MaterialFactory.create_diffuse(Color.from_hex("#C1CBD0"), roughness=1.0)
    v_plane = SDF_Material(Plane(), mat_plane)
    scene.add_object_by_context(v_plane, name="StudioBack", transform=Transform(np.array([0.0, 0.5, 2.0]), np.array([np.deg2rad(90), 0.0, 0.0])))

    # Lights
    key = Light(color=Color.from_hex("#EEE0BA"), intensity=2500.0, radius=100)
    scene.add_object_by_context(key, name="StudioKey", transform=Transform(np.array([2.5, 3.5, -1.0])))

    rim = Light(color=Color.from_hex("#DC97C5"), intensity=50.0, radius=0.75)
    scene.add_object_by_context(rim, name="StudioRim", transform=Transform(np.array([-3.0, 2.0, 1.0])))

    fill = Light(color=Color.from_hex("#C7DBD8"), intensity=150.0, radius=2)
    scene.add_object_by_context(fill, name="StudioFill", transform=Transform(np.array([0.0, -2.5, -2.0])))

    return scene

def get_rgb_room_with_objects_scene(width: int = 126, height: int = 126) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.5, -7.5]), np.array([0.0, 0.0, 0.0]))
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
    v_floor = SDF_Material(Cube(), mat_white)
    scene.add_object_by_context(v_floor, "Floor", Transform(np.array([0.0, -0.5, 0.0])))
    
    # Ceiling
    v_ceiling = SDF_Material(Cube(), mat_white)
    scene.add_object_by_context(v_ceiling, "Ceiling", Transform(np.array([0.0, 6.5, 0.0])))

    # Back Wall
    v_back = SDF_Material(Cube(), mat_blue)
    scene.add_object_by_context(v_back, "BackWall", Transform(np.array([0.0, 3.0, 5.5])))
    
    # Left Wall (Red)
    v_left = SDF_Material(Cube(), mat_red)
    scene.add_object_by_context(v_left, "LeftWall", Transform(np.array([-5.5, 3.0, 0.0])))

    # Right Wall (Green)
    v_right = SDF_Material(Cube(), mat_green)
    scene.add_object_by_context(v_right, "RightWall", Transform(np.array([5.5, 3.0, 0.0])))
    
    # Tall Box (Rotated)
    v_tall_box = SDF_Material(Cube(), mat_white)
    scene.add_object_by_context(v_tall_box, "TallBox", Transform(np.array([-2.0, 1.5, 2.0]), np.array([0.0, np.deg2rad(20.0), 0.0])))
    
    # Sphere (Mirror)
    v_mirror_sphere = SDF_Material(Sphere(), mat_mirror)
    scene.add_object_by_context(v_mirror_sphere, "MirrorBall", Transform(np.array([2.0, 1.25, 3.0])))
    
    # Small Cube (Glass/Crystal in front)
    v_glass_cube = SDF_Material(Cube(), mat_glass)
    scene.add_object_by_context(v_glass_cube, "GlassCube", Transform(np.array([0.0, 0.75, -2.0]), np.array([0.0, np.deg2rad(-15.0), 0.0])))

    # Cylinder object
    mat_cylinder = MaterialFactory.create_specular(Color.from_hex("#FFD700"), 0.2, 0.7, 0.9, 0.5)
    v_cylinder = SDF_Material(Cylinder(), mat_cylinder)
    scene.add_object_by_context(v_cylinder, "CylinderObj", Transform(np.array([1.5, 1.0, 1.0])))

    # Lighting
    ceiling_light = Light(
        color=Color.from_hex("#FFECDE"), 
        intensity=1000.0, 
        radius=5
    )
    scene.add_object_by_context(ceiling_light, "CeilingLight", Transform(np.array([0.0, 5.8, 0.0])))

    cam.transform.look_at(np.array([0.0, 2.5, 0.0]))

    return scene

def get_cyberpunk_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -4.0]), np.array([-0.1, 0.0, 0.0]))
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
    mat_wet = MaterialFactory.create_diffuse(Color.from_hex("#151515"), roughness=0.2)
    v_road = SDF_Material(Cube(), mat_wet)
    scene.add_object_by_context(v_road, "Road", Transform(np.array([0.0, -1.0, 0.0])))

    # Hero Object: Chrome Sphere
    mat_chrome = MaterialFactory.create_specular(Color.from_hex("#313238"), roughness=0.2, metallicness=1.0)
    v_hero = SDF_Material(Sphere(), mat_chrome)
    scene.add_object_by_context(v_hero, "HeroSphere", Transform(np.array([0.0, 0.5, 0.0])))

    # Background Buildings
    v_bldg_left = SDF_Material(Cube(), MaterialFactory.create_diffuse(Color.from_hex("#4DBC3E"), roughness=0.9))
    scene.add_object_by_context(v_bldg_left, "BldgLeft", Transform(np.array([-2.5, 2.0, 2.0])))

    v_bldg_right = SDF_Material(Cube(), MaterialFactory.create_diffuse(Color.from_hex("#E28335"), roughness=0.9))
    scene.add_object_by_context(v_bldg_right, "BldgRight", Transform(np.array([2.5, 1.3, 2.2])))

    # Neon cylinder
    mat_neon_cyl = MaterialFactory.create_emissive(Color.from_hex("#FF00FF"), 3.0)
    v_neon_cyl = SDF_Material(Cylinder(), mat_neon_cyl)
    scene.add_object_by_context(v_neon_cyl, "NeonCylinder", Transform(np.array([1.0, 1.0, -1.0])))

    # Lighting
    light_pink = Light(color=Color.from_hex("#FF0099"), intensity=25.0, radius=0.2)
    scene.add_object_by_context(light_pink, "NeonPink", Transform(np.array([-3.0, 2.0, -2.0])))
    
    light_cyan = Light(color=Color.from_hex("#00F0FF"), intensity=20.0, radius=0.2)
    scene.add_object_by_context(light_cyan, "NeonCyan", Transform(np.array([-2.5, 1.5, 2.0])))

    light_blue = Light(color=Color.from_hex("#3700FF"), intensity=18.0, radius=0.2)
    scene.add_object_by_context(light_blue, "NeonBlue", Transform(np.array([3.0, 1.0, -1.0])))

    light_rim = Light(color=Color.from_hex("#FFFFFF"), intensity=15.0, radius=0.5)
    scene.add_object_by_context(light_rim, "StreetLight", Transform(np.array([0.0, 3.0, 4.0])))

    return scene

def get_material_deck_scene(width: int = 160, height: int = 80) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.5, -5.0]), np.array([0.2, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="material_deck", camera=cam, background_color=Color.from_hex("#000000"))

    # Floor
    v_floor = SDF_Material(Cube(), MaterialFactory.create_diffuse(Color.from_hex("#CCCCCC"), roughness=1.0))
    scene.add_object_by_context(v_floor, "Floor", Transform(np.array([0.0, -1.0, 0.0])))

    base_col = Color.from_hex("#D4AF37")
    
    # Spheres with varying roughness
    v_s1 = SDF_Material(Sphere(), MaterialFactory.create_specular(base_col, roughness=0.0))
    scene.add_object_by_context(v_s1, "S_Mirror", Transform(np.array([-3.0, 0.5, 0.0])))

    v_s2 = SDF_Material(Sphere(), MaterialFactory.create_specular(base_col, roughness=0.25))
    scene.add_object_by_context(v_s2, "S_Brushed", Transform(np.array([-1.5, 0.5, 0.0])))

    v_s3 = SDF_Material(Sphere(), MaterialFactory.create_specular(base_col, roughness=0.5))
    scene.add_object_by_context(v_s3, "S_Rough", Transform(np.array([0.0, 0.5, 0.0])))

    v_s4 = SDF_Material(Sphere(), MaterialFactory.create_specular(base_col, roughness=0.75))
    scene.add_object_by_context(v_s4, "S_Matte", Transform(np.array([1.5, 0.5, 0.0])))
    
    v_s5 = SDF_Material(Sphere(), MaterialFactory.create_diffuse(Color.from_hex("#FF0000"), roughness=0.1))
    scene.add_object_by_context(v_s5, "S_Plastic", Transform(np.array([3.0, 0.5, 0.0])))

    # Cylinders with varying metallicness
    v_cyl1 = SDF_Material(Cylinder(), MaterialFactory.create_specular(Color.from_hex("#FFD700"), roughness=0.0, metallicness=1.0))
    scene.add_object_by_context(v_cyl1, "C_Mirror", Transform(np.array([-4.5, 0.6, 0.0])))

    v_cyl2 = SDF_Material(Cylinder(), MaterialFactory.create_specular(Color.from_hex("#FFD700"), roughness=0.5, metallicness=0.5))
    scene.add_object_by_context(v_cyl2, "C_Matte", Transform(np.array([4.5, 0.6, 0.0])))

    # Lighting
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 1.0), intensity=150.0), "Main", Transform(np.array([0.0, 5.0, -5.0])))
    scene.add_object_by_context(Light(color=Color(0.8, 0.8, 1.0), intensity=500.0, radius=5), "Fill", Transform(np.array([5.0, 2.0, -2.0])))

    cam.transform.look_at(v_s3.transform.position)
    return scene

def get_refraction_lab_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.0, -4.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene(name="refraction_lab", camera=cam, background_color=Color(0.05, 0.05, 0.05))

    # Striped Background Wall
    v_wall = SDF_Material(Cube(), MaterialFactory.create_emissive(Color(1.0, 1.0, 1.0), 1.0))
    scene.add_object_by_context(v_wall, "BackWall", Transform(np.array([0.0, 2.0, 4.0])))

    # Blocker bars
    for i in range(-6, 7):
        v_bar = SDF_Material(Cube(), MaterialFactory.create_diffuse(Color(0.0, 0.0, 0.0), 1.0))
        scene.add_object_by_context(v_bar, f"Bar_{6 + i}", Transform(np.array([float(i), 2.0, 3.5])))

    # Glass Sphere (IOR 1.5)
    v_s_glass = SDF_Material(Sphere(), MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["acrylic"], 0))
    scene.add_object_by_context(v_s_glass, "AcrylicSphere", Transform(np.array([-1.2, 0.5, 0.0])))

    # Diamond Sphere (IOR 2.4)
    v_s_diamond = SDF_Material(Sphere(), MaterialFactory.create_glass(Color.from_hex("#B9D3E3"), Color(0.9, 0.9, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["diamond"], 0.2))
    scene.add_object_by_context(v_s_diamond, "DiamondSphere", Transform(np.array([0.0, 0.5, 0.0])))

    # Water Sphere / Bubble (IOR 1.33)
    v_s_water = SDF_Material(Sphere(), MaterialFactory.create_glass(Color.from_hex("#A6ADD5"), Color.from_hex("#1F1FFF"), 0.0, 0.0, REFRACTIVE_INDICES["water"], 0.1))
    scene.add_object_by_context(v_s_water, "WaterSphere", Transform(np.array([1.2, 0.5, 0.0])))

    # Lighting
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 1.0), intensity=150.0), "FrontLight", Transform(np.array([2.0, 3.0, -3.0])))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))
    return scene

def get_scifi_corridor_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A high-contrast scene featuring repetitive metallic geometry and emissive lighting.
    Focuses on reflections of light sources on rough metal.
    """
    cam_transform = Transform(np.array([0.0, 1.0, 5.0]), np.array([0.0, 0.0, 0.0]))
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
    v_floor = SDF_Material(Cube(), mat_floor)
    scene.add_object_by_context(v_floor, "Floor", Transform(np.array([0.0, -1.0, -10.0])))

    # Ceiling
    v_ceiling = SDF_Material(Cube(), mat_floor)
    scene.add_object_by_context(v_ceiling, "Ceiling", Transform(np.array([0.0, 3.0, -10.0])))

    # Repetitive Pillars and Lights
    for z in range(0, -20, -4):
        # Left Pillar
        v_p_left = SDF_Material(Cube(), mat_pillar)
        scene.add_object_by_context(v_p_left, f"PillarLeft_{z}", Transform(np.array([-2.5, 1.0, float(z)])))

        # Right Pillar
        v_p_right = SDF_Material(Cube(), mat_pillar)
        scene.add_object_by_context(v_p_right, f"PillarRight_{z}", Transform(np.array([2.5, 1.0, float(z)])))

        # Emissive Light Strips on floor edges
        v_l_strip = SDF_Material(Cube(), mat_light_strip)
        scene.add_object_by_context(v_l_strip, f"Strip_{z}", Transform(np.array([0.0, -0.9, float(z)])))

        # Actual Light Sources corresponding to strips
        light = Light(color=Color.from_hex("#00AAAA"), intensity=200.0, radius=2.0)
        scene.add_object_by_context(light, f"PointLight_{z}", Transform(np.array([0.0, 0.5, float(z)])))

    # End focal point
    v_sphere_end = SDF_Material(Sphere(), MaterialFactory.create_specular(Color.from_hex("#FF0000"), roughness=0.1, metallicness=1.0))
    scene.add_object_by_context(v_sphere_end, "EndSphere", Transform(np.array([0.0, 1.0, -18.0])))

    cam.transform.look_at(np.array([0, 1, -20]))
    return scene

def get_sunset_monolith_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A scene focusing on warm lighting, long shadows, and the contrast between
    a matte organic ground and a sharp, reflective geometric object.
    """
    cam_transform = Transform(np.array([3.0, 1.5, -4.0]), np.array([0.0, 0.0, 0.0]))
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
    mat_mono = MaterialFactory.create_specular(Color.from_hex("#050505"), roughness=0.05, metallicness=1.0)
    v_monolith = SDF_Material(Cube(), mat_mono)
    scene.add_object_by_context(v_monolith, "MonolithObj", Transform(np.array([0.0, 2.0, 0.0]), np.array([0.0, np.deg2rad(25), 0.0])))

    # Sand Dunes (Matte, rough)
    mat_sand = MaterialFactory.create_diffuse(Color.from_hex("#D6783B"), roughness=1.0)
    v_floor = SDF_Material(Sphere(), mat_sand)
    scene.add_object_by_context(v_floor, "SandGround", Transform(np.array([0.0, -51.0, 0.0])))

    # Floating particles/smaller rocks
    v_rock1 = SDF_Material(Sphere(), MaterialFactory.create_diffuse(Color.from_hex("#554433"), roughness=0.9))
    scene.add_object_by_context(v_rock1, "Rock1", Transform(np.array([-1.5, 0.3, 1.5])))

    # Lighting
    # Sun (Low angle, very bright, sharp shadows)
    sun = Light(color=Color.from_hex("#FF9944"), intensity=3000.0, radius=0.5)
    scene.add_object_by_context(sun, "Sun", Transform(np.array([-8.0, 2.0, 10.0])))

    # Skylight fill (Purple/Blue ambient)
    fill = Light(color=Color.from_hex("#5544AA"), intensity=5000.0, radius=20.0)
    scene.add_object_by_context(fill, "SkyFill", Transform(np.array([5.0, 10.0, -5.0])))

    cam.transform.look_at(np.array([0, 1.5, 0]))
    return scene

def get_pastel_blocks_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A 'toy' scene with soft, bright lighting and materials that look like plastic or chalk.
    No metallic or glass surfaces.
    """
    cam_transform = Transform(np.array([0.0, 3.0, -5.0]), np.array([0.0, 0.0, 0.0]))
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
    v_floor = SDF_Material(Cube(), mat_white)
    scene.add_object_by_context(v_floor, "Floor", Transform(np.array([0.0, -1.0, 0.0])))

    # Stacked Objects
    # Base Cube
    v_base = SDF_Material(Cube(), mat_mint)
    scene.add_object_by_context(v_base, "BaseObj", Transform(np.array([0.0, 0.0, 0.0]), np.array([0.0, np.deg2rad(15), 0.0])))

    # Middle Cylinder (Simulated by stretched sphere or cube? using Sphere for variety)
    v_mid = SDF_Material(Sphere(), mat_pink)
    scene.add_object_by_context(v_mid, "MidObj", Transform(np.array([0.0, 1.6, 0.0])))

    # Top floating cube
    v_top = SDF_Material(Cube(), mat_purple)
    scene.add_object_by_context(v_top, "TopObj", Transform(np.array([0.2, 2.8, 0.2]), np.array([np.deg2rad(45), np.deg2rad(45), 0.0])))

    # Lighting (Soft Studio setup)
    # Main soft light
    key = Light(color=Color.from_hex("#FFFBEB"), intensity=180.0, radius=5.0)
    scene.add_object_by_context(key, "Key", Transform(np.array([3.0, 5.0, -5.0])))

    # Fill light
    fill = Light(color=Color.from_hex("#E6E6FA"), intensity=100.0, radius=5.0)
    scene.add_object_by_context(fill, "Fill", Transform(np.array([-4.0, 2.0, -2.0])))

    cam.transform.look_at(np.array([0, 1.2, 0]))
    return scene

def get_glass_prism_scene(width: int = 120, height: int = 120) -> Scene:
    """
    Features multiple glass objects with different Refractive Indices (IOR) 
    to demonstrate varying levels of light bending.
    """
    cam_transform = Transform(np.array([0.0, 2.0, -6.0]), np.array([-0.2, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=70.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="glass_prism_row", camera=cam, background_color=Color.from_hex("#101015"))

    # 1. Diamond Sphere (High IOR: 2.42) - Center
    # High dispersion and internal reflection
    mat_diamond = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["diamond"], 
        transmission=1.0
    )
    v_sphere_diamond = SDF_Material(Sphere(), mat_diamond)
    scene.add_object_by_context(v_sphere_diamond, "CenterDiamond", Transform(np.array([0.0, 0.5, 0.0])))

    # 2. Water Sphere (Low IOR: 1.33) - Left
    # Subtle bending, looks more transparent
    mat_water = MaterialFactory.create_glass(
        Color(0.9, 0.9, 1.0), 
        Color(0.8, 0.9, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=1.33, 
        transmission=1.0
    )
    v_sphere_water = SDF_Material(Sphere(), mat_water)
    scene.add_object_by_context(v_sphere_water, "LeftWater", Transform(np.array([-1.5, 0.5, 0.0])))

    # 3. Heavy Flint Glass Cube (Medium-High IOR: 1.65) - Right
    mat_flint = MaterialFactory.create_glass(
        Color(1.0, 0.9, 0.9), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.01, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["glass_flint_heavy"], 
        transmission=1.0
    )
    v_cube_glass = SDF_Material(Cube(), mat_flint)
    # Rotate to show refraction through edges
    t_flint = Transform(np.array([1.5, 0.5, 0.0]))
    t_flint.rotate(np.deg2rad(30), np.array([0, 1, 0]))
    t_flint.rotate(np.deg2rad(10), np.array([1, 0, 0]))
    scene.add_object_by_context(v_cube_glass, "RightFlint", t_flint)

    # Checkerboard Floor (to make refraction obvious)
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#888888"), roughness=0.8)
    v_floor = SDF_Material(Cube(), mat_floor)
    scene.add_object_by_context(v_floor, "FloorBase", Transform(np.array([0.0, -10.5, 5.0])))

    # Striped Wall behind objects
    for i in range(-5, 6):
        # Alternating colors
        col = Color.from_hex("#FF0000") if i % 2 == 0 else Color.from_hex("#FFFFFF")
        v_bar = SDF_Material(Cube(), MaterialFactory.create_emissive(col, 2.0))
        scene.add_object_by_context(v_bar, f"Bar_{i}", Transform(np.array([float(i), 2.0, 4.0])))

    # Light
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 1.0), intensity=50.0), "TopLight", Transform(np.array([0.0, 5.0, -3.0])))

    return scene

def get_glass_sculpture_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A complex arrangement of overlapping glass plates and spheres.
    Good for testing recursion depth and transmission color absorption.
    """
    cam_transform = Transform(np.array([3.0, 2.5, -3.0]), np.array([-0.3, 0.7, 0.0]))
    cam = Camera(cam_transform, fov=60.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    cam.transform.look_at(np.array([0, 0.5, 0]))
    
    scene = Scene(name="glass_sculpture", camera=cam, background_color=Color.from_hex("#200505"))

    # Central Red Glass Sphere
    # Red transmission color: Light passing through will turn red
    mat_red_glass = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 0.0, 0.2), # Transmission Color
        roughness=0.02, 
        metallicness=0.0, 
        ior=REFRACTIVE_INDICES["glass"], 
        transmission=1.0
    )
    v_center_sphere = SDF_Material(Sphere(), mat_red_glass)
    scene.add_object_by_context(v_center_sphere, "RedOrb", Transform(np.array([0.0, 0.8, 0.0])))

    # Encasing Glass Cube (Clear)
    mat_clear = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(1.0, 1.0, 1.0), 
        roughness=0.0, 
        metallicness=0.0, 
        ior=1.1, # Low IOR to look like thin plastic or aerogel
        transmission=0.9
    )
    v_outer_box = SDF_Material(Cube(), mat_clear)
    scene.add_object_by_context(v_outer_box, "ClearBox", Transform(np.array([0.0, 0.8, 0.0])))

    # Back Mirror to reflect the back of the glass objects
    v_mirror = SDF_Material(Cube(), MaterialFactory.create_specular(Color(1.0, 1.0, 1.0), roughness=0.0, metallicness=1.0))
    scene.add_object_by_context(v_mirror, "MirrorBack", Transform(np.array([0.0, 2.0, 3.0])))

    # Lights
    # Cyan light to contrast with red glass
    scene.add_object_by_context(Light(color=Color.from_hex("#00FFFF"), intensity=150.0), "CyanKey", Transform(np.array([4.0, 4.0, -4.0])))
    # White rim
    scene.add_object_by_context(Light(color=Color.from_hex("#FFFFFF"), intensity=50.0), "Rim", Transform(np.array([-4.0, 1.0, 0.0])))

    return scene

def get_100_spheres_grid_scene(width: int = 128, height: int = 128) -> Scene:
    """
    Stress test scene: Generates a 10x10 grid of spheres with varying materials.
    Total objects: 100 spheres + 1 floor = 101 objects.
    """
    cam_transform = Transform(np.array([-8.0, 8.0, -8.0]), np.array([0.0, 0.0, 0.0]))
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
    shared_sphere_shape = Sphere()

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
            
            # Material: Varies per object
            color = Color(r / rows, 0.5, c / cols)
            
            if (r + c) % 2 == 0:
                mat = MaterialFactory.create_specular(color, roughness=0.2, metallicness=0.9)
            else:
                mat = MaterialFactory.create_diffuse(color, roughness=0.8)
            
            # SDF_Material: Links the shared shape and unique material
            scene.add_object_by_context(SDF_Material(shared_sphere_shape, mat), f"S_{r}_{c}", Transform(np.array([x, y, z])))

    # Floor
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#333333"), roughness=0.5)
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "Floor", Transform(np.array([0.0, -2.0, 0.0])))

    # Light
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 0.9), intensity=1000.0), "Sun", Transform(np.array([10.0, 20.0, -10.0])))
    
    # Ensure camera looks below the origin
    cam.transform.look_at(np.array([0, -1, 0]))

    return scene

def get_low_ior_scene(width: int = 120, height: int = 120) -> Scene:
    """
    Features a sphere with an IOR < 1.0 (0.8).
    This acts like an 'air bubble in glass' but inverted.
    """
    cam_transform = Transform(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=60.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="low_ior_anomaly", camera=cam, background_color=Color.from_hex("#000000"))

    # 1. The Low IOR Sphere
    mat_low_ior = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), 
        Color(0.8, 1.0, 0.9),
        roughness=0.0, 
        ior=0.8
    )
    scene.add_object_by_context(SDF_Material(Sphere(), mat_low_ior), "AnomalyObj", Transform.Identity())

    # 2. Background Grid
    # Reuse a single cube shape for all tiles
    tile_shape = Cube()
    
    mat_red = MaterialFactory.create_emissive(Color.from_hex("#FF4444"), 2.0)
    mat_blue = MaterialFactory.create_emissive(Color.from_hex("#4444FF"), 2.0)

    for x in range(-3, 4):
        for y in range(-3, 4):
            # Calculate position
            pos = np.array([x * 1.5, y * 1.5, 4.0])
            
            # Select material
            mat = mat_red if (x + y) % 2 == 0 else mat_blue
            
            scene.add_object_by_context(SDF_Material(tile_shape, mat), f"Tile_{x}_{y}", Transform(pos))

    # Light
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 1.0), intensity=1000.0), "Front", Transform(np.array([2.0, 2.0, -3.0])))

    return scene

def get_shape_showcase_scene(width: int = 160, height: int = 120) -> Scene:
    """
    A showcase scene featuring all available 3D shapes arranged in a grid.
    Demonstrates variety of geometries with different materials.
    """
    cam_transform = Transform(np.array([0.0, 3.0, -8.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=70.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="shape_showcase", camera=cam, background_color=Color.from_hex("#1a1a2e"))

    # Materials
    mat_metal = MaterialFactory.create_specular(Color.from_hex("#C0C0C0"), 0.1, 0.9, 0.8, 0.2)
    mat_glass = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, 1.5, 1.0)
    mat_diffuse = MaterialFactory.create_diffuse(Color.from_hex("#FF6B6B"), 0.3)
    mat_emissive = MaterialFactory.create_emissive(Color.from_hex("#4ECDC4"), 1.5)

    # Row 1: Spheres and Cubes
    sphere1 = SDF_Material(Sphere(), mat_metal)
    scene.add_object_by_context(sphere1, "Sphere1", Transform(np.array([-3.0, 1.0, 0.0])))
    
    cube1 = SDF_Material(Cube(), mat_diffuse)
    scene.add_object_by_context(cube1, "Cube1", Transform(np.array([-1.0, 1.0, 0.0])))
    
    sphere2 = SDF_Material(Sphere(), mat_glass)
    scene.add_object_by_context(sphere2, "Sphere2", Transform(np.array([1.0, 1.0, 0.0])))
    
    cube2 = SDF_Material(Cube(), mat_emissive)
    scene.add_object_by_context(cube2, "Cube2", Transform(np.array([3.0, 1.0, 0.0])))

    # Row 2: Cylinders and Pyramids
    cylinder1 = SDF_Material(Cylinder(), mat_metal)
    scene.add_object_by_context(cylinder1, "Cylinder1", Transform(np.array([-3.0, -1.0, 0.0])))
    
    pyramid1 = SDF_Material(Pyramid(), mat_diffuse)
    scene.add_object_by_context(pyramid1, "Pyramid1", Transform(np.array([-1.0, -1.0, 0.0])))
    
    cylinder2 = SDF_Material(Cylinder(), mat_glass)
    scene.add_object_by_context(cylinder2, "Cylinder2", Transform(np.array([1.0, -1.0, 0.0])))
    
    pyramid2 = SDF_Material(Pyramid(), mat_emissive)
    scene.add_object_by_context(pyramid2, "Pyramid2", Transform(np.array([3.0, -1.0, 0.0])))

    # Row 3: Prisms and Capsules
    prism1 = SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_metal)
    scene.add_object_by_context(prism1, "Prism1", Transform(np.array([-2.0, -3.0, 0.0])))
    
    capsule1 = SDF_Material(Capsule(), mat_glass)
    scene.add_object_by_context(capsule1, "Capsule1", Transform(np.array([0.0, -3.0, 0.0])))
    
    prism2 = SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_emissive)
    scene.add_object_by_context(prism2, "Prism2", Transform(np.array([2.0, -3.0, 0.0])))

    # Floor
    floor_mat = MaterialFactory.create_diffuse(Color.from_hex("#333333"), 0.8)
    floor = SDF_Material(Cube(), floor_mat)
    scene.add_object_by_context(floor, "Floor", Transform(np.array([0.0, -5.0, 0.0])))

    # Lighting
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 1.0), intensity=2000.0), "Main", Transform(np.array([5.0, 5.0, -5.0])))
    scene.add_object_by_context(Light(color=Color(0.8, 0.8, 1.0), intensity=1000.0, radius=5), "Fill", Transform(np.array([-5.0, 3.0, 5.0])))

    cam.transform.look_at(np.array([0, -1, 0]))
    return scene

def get_abstract_geometry_scene(width: int = 140, height: int = 100) -> Scene:
    """
    An abstract scene with geometric shapes arranged in a artistic composition.
    Features overlapping transparent shapes and dramatic lighting.
    """
    cam_transform = Transform(np.array([2.0, 2.0, -5.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=65.0, near=0.1, far=50.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="abstract_geometry", camera=cam, background_color=Color.from_hex("#0f0f23"))

    # Materials
    mat_transparent = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(0.9, 0.95, 1.0), 0.0, 0.0, 1.4, 0.8)
    mat_mirror = MaterialFactory.create_specular(Color.from_hex("#FFFFFF"), 0.0, 1.0, 1.0, 0.0)
    mat_emiss_red = MaterialFactory.create_emissive(Color.from_hex("#FF1744"), 2.0)
    mat_emiss_blue = MaterialFactory.create_emissive(Color.from_hex("#2979FF"), 2.0)

    # Central composition
    # Large transparent sphere
    sphere_large = SDF_Material(Sphere(), mat_transparent)
    scene.add_object_by_context(sphere_large, "LargeSphere", Transform(np.array([0.0, 0.0, 0.0])))
    
    # Intersecting cylinder
    cylinder = SDF_Material(Cylinder(), mat_mirror)
    t_cyl = Transform(np.array([0.5, 0.0, 0.0]))
    t_cyl.rotate(np.deg2rad(45), np.array([0, 0, 1]))
    scene.add_object_by_context(cylinder, "IntersectCylinder", t_cyl)
    
    # Floating cubes
    cube1 = SDF_Material(Cube(), mat_emiss_red)
    scene.add_object_by_context(cube1, "FloatCube1", Transform(np.array([-1.5, 1.0, 1.0])))
    
    cube2 = SDF_Material(Cube(), mat_emiss_blue)
    scene.add_object_by_context(cube2, "FloatCube2", Transform(np.array([1.5, -1.0, -1.0])))
    
    # Pyramid on top
    pyramid = SDF_Material(Pyramid(), mat_transparent)
    scene.add_object_by_context(pyramid, "TopPyramid", Transform(np.array([0.0, 1.8, 0.0])))
    
    # SignedDistanceShape3DExtrusion base
    prism = SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_mirror)
    scene.add_object_by_context(prism, "BasePrism", Transform(np.array([0.0, -1.5, 0.0])))

    # Lighting
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 1.0), intensity=500.0), "Key", Transform(np.array([3.0, 3.0, -3.0])))
    scene.add_object_by_context(Light(color=Color(0.5, 0.7, 1.0), intensity=120.0, radius=2), "Fill", Transform(np.array([-3.0, -1.0, 3.0])))

    cam.transform.look_at(np.array([0, 0, 0]))
    return scene

def get_industrial_shapes_scene(width: int = 150, height: int = 100) -> Scene:
    """
    An industrial-themed scene with metallic shapes, pipes, and machinery-like objects.
    """
    cam_transform = Transform(np.array([0.0, 2.0, -6.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=70.0, near=0.1, far=50.0, resolution_width=width, resolution_height=height)
    
    scene = Scene(name="industrial_shapes", camera=cam, background_color=Color.from_hex("#2c2c2c"))

    # Materials
    mat_rusty_metal = MaterialFactory.create_specular(Color.from_hex("#8B4513"), 0.4, 0.8, 0.7, 0.3)
    mat_steel = MaterialFactory.create_specular(Color.from_hex("#C0C0C0"), 0.1, 0.9, 0.9, 0.1)
    mat_brass = MaterialFactory.create_specular(Color.from_hex("#B87333"), 0.2, 0.7, 0.8, 0.4)
    mat_concrete = MaterialFactory.create_diffuse(Color.from_hex("#696969"), 0.9)
    mat_emiss_red = MaterialFactory.create_emissive(Color.from_hex("#FF3B3B"), 3.0)
    mat_emiss_blue = MaterialFactory.create_emissive(Color.from_hex("#3B7AFF"), 3.0)

    # Floor
    floor = SDF_Material(Cube(), mat_concrete)
    scene.add_object_by_context(floor, "ConcreteFloor", Transform(np.array([0.0, -2.0, 0.0])))

    # Main structure: large cube base
    base = SDF_Material(Cube(), mat_steel)
    scene.add_object_by_context(base, "BaseStructure", Transform(np.array([0.0, -1.0, 0.0])))

    # Pipes: cylinders
    pipe1 = SDF_Material(Cylinder(), mat_brass)
    t_pipe1 = Transform(np.array([-1.5, 0.0, 0.0]))
    t_pipe1.rotate(np.deg2rad(90), np.array([0, 1, 0]))
    scene.add_object_by_context(pipe1, "Pipe1", t_pipe1)
    
    pipe2 = SDF_Material(Cylinder(), mat_rusty_metal)
    t_pipe2 = Transform(np.array([1.5, 0.5, 0.0]))
    t_pipe2.rotate(np.deg2rad(45), np.array([1, 0, 0]))
    scene.add_object_by_context(pipe2, "Pipe2", t_pipe2)

    # Gears: thick cylinders
    gear1 = SDF_Material(Cylinder(), mat_steel)
    scene.add_object_by_context(gear1, "Gear1", Transform(np.array([0.0, 1.0, 1.0])))
    
    gear2 = SDF_Material(Cylinder(), mat_brass)
    scene.add_object_by_context(gear2, "Gear2", Transform(np.array([0.0, 1.0, -1.0])))

    # Support beams: prisms
    beam1 = SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_rusty_metal)
    t_beam1 = Transform(np.array([-2.0, 0.5, 2.0]))
    t_beam1.rotate(np.deg2rad(30), np.array([0, 1, 0]))
    scene.add_object_by_context(beam1, "Beam1", t_beam1)
    
    beam2 = SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_steel)
    t_beam2 = Transform(np.array([2.0, 0.5, -2.0]))
    t_beam2.rotate(np.deg2rad(-30), np.array([0, 1, 0]))
    scene.add_object_by_context(beam2, "Beam2", t_beam2)

    # Control panel: small cubes and pyramid
    panel_base = SDF_Material(Cube(), mat_steel)
    scene.add_object_by_context(panel_base, "PanelBase", Transform(np.array([0.0, 0.2, 2.5])))
    
    button1 = SDF_Material(Cube(), mat_emiss_red)
    scene.add_object_by_context(button1, "Button1", Transform(np.array([-0.3, 0.4, 2.5])))
    
    button2 = SDF_Material(Cube(), mat_emiss_blue)
    scene.add_object_by_context(button2, "Button2", Transform(np.array([0.3, 0.4, 2.5])))

    antenna = SDF_Material(Pyramid(), mat_brass)
    scene.add_object_by_context(antenna, "Antenna", Transform(np.array([0.0, 0.8, 2.5])))

    # Lighting: harsh industrial lighting
    scene.add_object_by_context(Light(color=Color(1.0, 1.0, 0.9), intensity=250.0), "Overhead", Transform(np.array([0.0, 4.0, 0.0])))
    scene.add_object_by_context(Light(color=Color(0.8, 0.8, 1.0), intensity=100.0), "Side", Transform(np.array([3.0, 1.0, -3.0])))

    cam.transform.look_at(np.array([0, 0, 0]))
    return scene
