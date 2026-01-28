import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

sys.path.insert(0, current_dir)

from src.Data.Transform import Transform
from src.Data.Color import Color, ColorGradient
from src.Data.Scene import Scene
from src.Data.Camera import Camera, CameraType
from src.Data.Context import Mesh_Material, SDF_Material
from src.Geometry.SDF import *
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
    scene.add_object_by_context(SDF_Material(Sphere(1), mat), "Sphere Min", Transform.Identity())

    # Ground
    matg = MaterialFactory.create_diffuse(Color.from_hex("#3F3F3F"), 0.9)
    scene.add_object_by_context(SDF_Material(Sphere(100), matg), "Ground Min", Transform(np.array([0.0, -101, 0.0]), scale=np.full(3, 100)))

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

    sky_colors = [Color.from_hex("#2D2515"), Color.from_hex("#42424E"), Color.from_hex("#5B6791"), Color.from_hex("#87BFC6")]
    sky_positions = np.array([0.0, 0.4, 0.45, 1.0])
    scene = Scene("gradient_scene", cam, background_color=ColorGradient(sky_colors, sky_positions))

    # Objects
    mat_metal = MaterialFactory.create_specular(Color.from_hex("#47505C"), 0.2, 0.9, 1.0, 1.0)
    sph_1 = SDF_Material(Sphere(), mat_metal)
    scene.add_object_by_context(sph_1, "Reflective Sphere", Transform(np.array([0.0, 2.25, -5.0])))
    cam.transform.look_at(np.array([0.0, 2.25, 0.0]), np.array([0, 1, 0]))

    mat_matte = MaterialFactory.create_diffuse(Color.from_hex("#C27A23"), 0.8)
    bx_1 = SDF_Material(Cube(), mat_matte)
    scene.add_object_by_context(bx_1, "Matte Box", Transform(np.array([0.0, 0.0, -2.0]), np.array([0.0, np.deg2rad(15), 0.0])))

    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#EE1717"), 2)
    sph_2 = SDF_Material(Sphere(0.4), mat_glow)
    scene.add_object_by_context(sph_2, "Emissive Orb", Transform(np.array([-0.5, 2.5, -1.5]), scale=np.full(3, 0.4)))

    mat_cylinder = MaterialFactory.create_specular(Color.from_hex("#FFD700"), 0.1, 0.8, 0.9, 0.5)
    cyl_1 = SDF_Material(Cylinder(), mat_cylinder)
    scene.add_object_by_context(cyl_1, "Golden Cylinder", Transform(np.array([2.0, 1.0, -3.0]), np.array([0.0, np.deg2rad(30), 0.0])))

    mat_pyramid = MaterialFactory.create_diffuse(Color.from_hex("#8B4513"), 0.7)
    pyr_1 = SDF_Material(Pyramid(), mat_pyramid)
    scene.add_object_by_context(pyr_1, "Wooden Pyramid", Transform(np.array([-2.5, 0.5, -2.0])))

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
        fov=70.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("emissive_scene", cam, background_color=Color.from_hex("#000000"))

    # Objects
    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#FFEA62"), 1.2)
    v_emissive = SDF_Material(Sphere(0.3), mat_glow)
    scene.add_object_by_context(v_emissive, "GlowingSphere", Transform(np.zeros(3), scale=np.full(3, 0.3)))

    mat_reflect = MaterialFactory.create_specular(Color.from_hex("#6B6666"), roughness=0.2, metallicness=0.75, specular_intensity=1.0, specular_tint_amount=1.0)
    v_mirror = SDF_Material(Sphere(), mat_reflect)
    scene.add_object_by_context(v_mirror, "MirrorSphere", Transform.Identity())

    matg = MaterialFactory.create_diffuse(Color.from_hex("#202020"), roughness=0.8)
    v_ground = SDF_Material(Sphere(100), matg)
    scene.add_object_by_context(v_ground, "Ground", Transform(np.array([0.0, -100.5, 0.0]), scale=np.full(3, 100)))

    # Lights
    fill = Light(color=Color.from_hex("#AAAACC"), intensity=1000.0, radius=10.0)
    scene.add_object_by_context(fill, "FillEmiss", Transform(np.array([-4.0, 2.0, -3.0])))

    return scene

def get_lit_studio_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -4.0]), np.array([-0.15, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("lit_studio", cam, background_color=Color.from_hex("#BEC2CF"))

    # Objects
    mat1 = MaterialFactory.create_specular(Color.from_hex("#FFB86B"), 0.2, 0.1, 0.9, 0)
    v_s1 = SDF_Material(Sphere(0.4), mat1)
    scene.add_object_by_context(v_s1, "StudioBallA", Transform(np.array([-0.6, 0.4, 0.5]), scale=np.full(3, 0.4)))

    mat2 = MaterialFactory.create_specular(Color.from_hex("#6B9BFF"), 0.2, 0.4, 0.9, 0)
    v_s2 = SDF_Material(Sphere(0.45), mat2)
    scene.add_object_by_context(v_s2, "StudioBallB", Transform(np.array([0.8, 0.45, 0.2]), scale=np.full(3, 0.45)))

    mat_plane = MaterialFactory.create_diffuse(Color.from_hex("#C1CBD0"), roughness=1.0)
    v_plane = SDF_Material(Plane(), mat_plane)
    scene.add_object_by_context(v_plane, "StudioBack", Transform(np.array([0.0, 0.0, -2.0]), np.array([np.deg2rad(90), 0.0, 0.0])))

    # Lights
    key = Light(color=Color.from_hex("#EEE0BA"), intensity=2500.0, radius=100)
    scene.add_object_by_context(key, "StudioKey", Transform(np.array([2.5, 3.5, 1.0])))

    rim = Light(color=Color.from_hex("#DC97C5"), intensity=50.0, radius=0.75)
    scene.add_object_by_context(rim, "StudioRim", Transform(np.array([-3.0, 2.0, -1.0])))

    fill = Light(color=Color.from_hex("#C7DBD8"), intensity=150.0, radius=2)
    scene.add_object_by_context(fill, "StudioFill", Transform(np.array([0.0, -2.5, 2.0])))

    return scene

def get_rgb_room_with_objects_scene(width: int = 126, height: int = 126) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.5, -7.5]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("rgb_cornell_box", cam, background_color=Color(0.0, 0.0, 0.0))

    # Materials
    mat_white = MaterialFactory.create_diffuse(Color.from_hex("#E0E0E0"), 1.0)
    mat_red = MaterialFactory.create_diffuse(Color.from_hex("#B03030"), 1.0)
    mat_green = MaterialFactory.create_diffuse(Color.from_hex("#30B030"), 1.0)
    mat_blue = MaterialFactory.create_diffuse(Color.from_hex("#3036B0"), 1.0)
    mat_mirror = MaterialFactory.create_specular(Color.from_hex("#FFFFFF"), 0.1, 1.0, 0)
    mat_glass = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["glass"], 0)
    mat_cyl = MaterialFactory.create_specular(Color.from_hex("#FFD700"), 0.2, 0.7, 0.9, 0.5)

    # Geometry
    scene.add_object_by_context(SDF_Material(Cube(), mat_white), "Floor", Transform(np.array([0.0, -0.5, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_white), "Ceiling", Transform(np.array([0.0, 6.5, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_blue), "BackWall", Transform(np.array([0.0, 3.0, 5.5])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_red), "LeftWall", Transform(np.array([-5.5, 3.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_green), "RightWall", Transform(np.array([5.5, 3.0, 0.0])))
    
    scene.add_object_by_context(SDF_Material(Cube(), mat_white), "TallBox", Transform(np.array([-2.0, 1.5, -2.0]), np.array([0.0, np.deg2rad(20.0), 0.0])))
    scene.add_object_by_context(SDF_Material(Sphere(), mat_mirror), "MirrorBall", Transform(np.array([2.0, 1.25, -3.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_glass), "GlassCube", Transform(np.array([0.0, 0.75, 2.0]), np.array([0.0, np.deg2rad(-15.0), 0.0])))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_cyl), "CylinderObj", Transform(np.array([1.5, 1.0, -1.0])))

    # Lights
    ceiling_light = Light(color=Color.from_hex("#FFECDE"), intensity=1000.0, radius=5)
    scene.add_object_by_context(ceiling_light, "CeilingLight", Transform(np.array([0.0, 5.8, 0.0])))

    cam.transform.look_at(np.array([0.0, 2.5, 0.0]))

    return scene

def get_cyberpunk_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.5, -4.0]), np.array([-0.1, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    sky_colors = [Color.from_hex("#050008"), Color.from_hex("#0B1333")]
    scene = Scene("cyberpunk_street", cam, background_color=ColorGradient(sky_colors, np.array([0.0, 1.0])))

    # Objects
    mat_wet = MaterialFactory.create_diffuse(Color.from_hex("#151515"), roughness=0.2)
    scene.add_object_by_context(SDF_Material(Cube(), mat_wet), "Road", Transform(np.array([0.0, -1.0, 0.0])))

    mat_chrome = MaterialFactory.create_specular(Color.from_hex("#313238"), roughness=0.2, metallicness=1.0)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_chrome), "HeroSphere", Transform(np.array([0.0, 0.5, 0.0])))

    mat_b_left = MaterialFactory.create_diffuse(Color.from_hex("#4DBC3E"), roughness=0.9)
    scene.add_object_by_context(SDF_Material(Cube(), mat_b_left), "BldgLeft", Transform(np.array([-2.5, 2.0, -2.0])))

    mat_b_right = MaterialFactory.create_diffuse(Color.from_hex("#E28335"), roughness=0.9)
    scene.add_object_by_context(SDF_Material(Cube(), mat_b_right), "BldgRight", Transform(np.array([2.5, 1.3, -2.2])))

    mat_neon_cyl = MaterialFactory.create_emissive(Color.from_hex("#FF00FF"), 3.0)
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_neon_cyl), "NeonCylinder", Transform(np.array([1.0, 1.0, 1.0])))
    # Lights
    l_pink = Light(color=Color.from_hex("#FF0099"), intensity=25.0, radius=0.2)
    scene.add_object_by_context(l_pink, "NeonPink", Transform(np.array([-3.0, 2.0, 2.0])))
    
    l_cyan = Light(color=Color.from_hex("#00F0FF"), intensity=20.0, radius=0.2)
    scene.add_object_by_context(l_cyan, "NeonCyan", Transform(np.array([-2.5, 1.5, -2.0])))

    l_blue = Light(color=Color.from_hex("#3700FF"), intensity=18.0, radius=0.2)
    scene.add_object_by_context(l_blue, "NeonBlue", Transform(np.array([3.0, 1.0, 1.0])))
    l_rim = Light(color=Color.from_hex("#FFFFFF"), intensity=15.0, radius=0.5)
    scene.add_object_by_context(l_rim, "StreetLight", Transform(np.array([0.0, 3.0, -4.0])))

    return scene

def get_material_deck_scene(width: int = 160, height: int = 80) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.5, -5.0]), np.array([0.2, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=86.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("material_deck", cam, background_color=Color.from_hex("#000000"))

    # Floor
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#CCCCCC"), roughness=1.0)
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "Floor", Transform(np.array([0.0, -1.0, 0.0])))

    base_col = Color.from_hex("#D4AF37")
    
    # Material Variations
    mat_s1 = MaterialFactory.create_specular(base_col, roughness=0.0)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_s1), "S_Mirror", Transform(np.array([-3.0, 0.5, 0.0])))

    mat_s2 = MaterialFactory.create_specular(base_col, roughness=0.25)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_s2), "S_Brushed", Transform(np.array([-1.5, 0.5, 0.0])))
    mat_s3 = MaterialFactory.create_specular(base_col, roughness=0.5)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_s3), "S_Rough", Transform(np.array([0.0, 0.5, 0.0])))

    mat_s4 = MaterialFactory.create_specular(base_col, roughness=0.75)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_s4), "S_Matte", Transform(np.array([1.5, 0.5, 0.0])))
    
    mat_s5 = MaterialFactory.create_diffuse(Color.from_hex("#FF0000"), roughness=0.1)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_s5), "S_Plastic", Transform(np.array([3.0, 0.5, 0.0])))

    mat_c1 = MaterialFactory.create_specular(Color.from_hex("#FFD700"), roughness=0.0, metallicness=1.0)
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_c1), "C_Mirror", Transform(np.array([-4.5, 0.6, 0.0])))

    mat_c2 = MaterialFactory.create_specular(Color.from_hex("#FFD700"), roughness=0.5, metallicness=0.5)
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_c2), "C_Matte", Transform(np.array([4.5, 0.6, 0.0])))

    # Lights
    l_main = Light(color=Color(1.0, 1.0, 1.0), intensity=150.0)
    scene.add_object_by_context(l_main, "Main", Transform(np.array([0.0, 5.0, -5.0])))
    
    l_fill = Light(color=Color(0.8, 0.8, 1.0), intensity=500.0, radius=5)
    scene.add_object_by_context(l_fill, "Fill", Transform(np.array([5.0, 2.0, -2.0])))

    cam.transform.look_at(np.array([0.0, 0.5, 0.0]))
    return scene

def get_refraction_lab_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.0, -4.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("refraction_lab", cam, background_color=Color(0.05, 0.05, 0.05))

    # Background
    mat_wall = MaterialFactory.create_emissive(Color(1.0, 1.0, 1.0), 1.0)
    scene.add_object_by_context(SDF_Material(Cube(), mat_wall), "BackWall", Transform(np.array([0.0, 2.0, -4.0])))

    # Bars
    mat_bar = MaterialFactory.create_diffuse(Color(0.0, 0.0, 0.0), 1.0)
    for i in range(-6, 7):
        scene.add_object_by_context(SDF_Material(Cube(), mat_bar), f"Bar_{6 + i}", Transform(np.array([i, 2.0, -3.5])))

    # Spheres
    mat_acrylic = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["acrylic"], 0)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_acrylic), "AcrylicSphere", Transform(np.array([-1.2, 0.5, 0.0])))

    mat_diamond = MaterialFactory.create_glass(Color.from_hex("#B9D3E3"), Color(0.9, 0.9, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["diamond"], 0.2)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_diamond), "DiamondSphere", Transform(np.array([0.0, 0.5, 0.0])))

    mat_water = MaterialFactory.create_glass(Color.from_hex("#A6ADD5"), Color.from_hex("#1F1FFF"), 0.0, 0.0, REFRACTIVE_INDICES["water"], 0.1)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_water), "WaterSphere", Transform(np.array([1.2, 0.5, 0.0])))

    # Lights
    l_front = Light(color=Color(1.0, 1.0, 1.0), intensity=150.0)
    scene.add_object_by_context(l_front, "FrontLight", Transform(np.array([2.0, 3.0, -3.0])))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))
    return scene

def get_scifi_corridor_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -5.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform, fov=80.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("scifi_corridor", cam, background_color=Color.from_hex("#020205"))

    # Materials
    mat_floor = MaterialFactory.create_specular(Color.from_hex("#2A2A2A"), roughness=0.3, metallicness=0.8)
    mat_pillar = MaterialFactory.create_specular(Color.from_hex("#111111"), roughness=0.5, metallicness=0.9)
    mat_light_strip = MaterialFactory.create_emissive(Color.from_hex("#00FFFF"), 5.0)

    # Structure
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "Floor", Transform(np.array([0.0, -1.0, -10.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "Ceiling", Transform(np.array([0.0, 3.0, -10.0])))

    # Pillars and Lights Loop
    for z in range(0, 20, 4):
        z_neg = -z
        scene.add_object_by_context(SDF_Material(Cube(), mat_pillar), f"PillarLeft_{z_neg}", Transform(np.array([-2.5, 1.0, z_neg])))
        scene.add_object_by_context(SDF_Material(Cube(), mat_pillar), f"PillarRight_{z_neg}", Transform(np.array([2.5, 1.0, z_neg])))
        scene.add_object_by_context(SDF_Material(Cube(), mat_light_strip), f"Strip_{z_neg}", Transform(np.array([0.0, -0.9, z_neg])))
        
        point_light = Light(color=Color.from_hex("#00AAAA"), intensity=200.0, radius=2.0)
        scene.add_object_by_context(point_light, f"PointLight_{z_neg}", Transform(np.array([0.0, 0.5, z_neg])))

    # End Object
    mat_end = MaterialFactory.create_specular(Color.from_hex("#FF0000"), roughness=0.1, metallicness=1.0)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_end), "EndSphere", Transform(np.array([0.0, 1.0, 18.0])))

    cam.transform.look_at(np.array([0, 1, -20]))
    return scene

def get_sunset_monolith_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([3.0, 1.5, -4.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform, fov=65.0, near=0.1, far=150.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )

    sky_colors = [Color.from_hex("#38160D"), Color.from_hex("#E85635"), Color.from_hex("#F29B36"), Color.from_hex("#685888")]
    sky_positions = np.array([0.0, 0.3, 0.5, 1.0])
    scene = Scene("sunset_monolith", cam, background_color=ColorGradient(sky_colors, sky_positions))

    # Objects
    mat_mono = MaterialFactory.create_specular(Color.from_hex("#050505"), roughness=0.05, metallicness=1.0)
    scene.add_object_by_context(SDF_Material(Cube(), mat_mono), "MonolithObj", Transform(np.array([0.0, 2.0, 0.0]), np.array([0.0, np.deg2rad(25), 0.0])))

    mat_sand = MaterialFactory.create_diffuse(Color.from_hex("#D6783B"), roughness=1.0)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_sand), "SandGround", Transform(np.array([0.0, -51.0, 0.0])))

    mat_rock = MaterialFactory.create_diffuse(Color.from_hex("#554433"), roughness=0.9)
    scene.add_object_by_context(SDF_Material(Sphere(), mat_rock), "Rock1", Transform(np.array([-1.5, 0.3, 1.5])))

    # Lights
    sun = Light(color=Color.from_hex("#FF9944"), intensity=3000.0, radius=0.5)
    scene.add_object_by_context(sun, "Sun", Transform(np.array([-8.0, 2.0, 10.0])))

    fill = Light(color=Color.from_hex("#5544AA"), intensity=5000.0, radius=20.0)
    scene.add_object_by_context(fill, "SkyFill", Transform(np.array([5.0, 10.0, -5.0])))

    cam.transform.look_at(np.array([0, 1.5, 0]))
    return scene

def get_pastel_blocks_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([0.0, 3.0, -5.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform, fov=60.0, near=0.1, far=50.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("pastel_blocks", cam, background_color=Color.from_hex("#F0F4F8"))

    # Materials
    mat_pink = MaterialFactory.create_diffuse(Color.from_hex("#FFB7B2"), roughness=0.6)
    mat_mint = MaterialFactory.create_diffuse(Color.from_hex("#B5EAD7"), roughness=0.6)
    mat_purple = MaterialFactory.create_diffuse(Color.from_hex("#E2F0CB"), roughness=0.6)
    mat_white = MaterialFactory.create_diffuse(Color.from_hex("#FFFFFF"), roughness=0.9)

    # Objects
    scene.add_object_by_context(SDF_Material(Cube(), mat_white), "Floor", Transform(np.array([0.0, -1.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_mint), "BaseObj", Transform(np.array([0.0, 0.0, 0.0]), np.array([0.0, np.deg2rad(15), 0.0])))
    scene.add_object_by_context(SDF_Material(Sphere(), mat_pink), "MidObj", Transform(np.array([0.0, 1.6, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_purple), "TopObj", Transform(np.array([0.2, 2.8, 0.2]), np.array([np.deg2rad(45), np.deg2rad(45), 0.0])))

    # Lights
    key = Light(color=Color.from_hex("#FFFBEB"), intensity=180.0, radius=5.0)
    scene.add_object_by_context(key, "Key", Transform(np.array([3.0, 5.0, -5.0])))

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
    cam = Camera(cam_transform, fov=70.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    
    scene = Scene("glass_prism_row", cam, background_color=Color.from_hex("#101015"))

    # 1. Diamond Sphere (High IOR: 2.42)
    mat_diamond = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), Color(1.0, 1.0, 1.0), 
        roughness=0.0, metallicness=0.0, ior=REFRACTIVE_INDICES["diamond"], transmission=1.0
    )
    scene.add_object_by_context(SDF_Material(Sphere(), mat_diamond), "CenterDiamond", Transform(np.array([0.0, 0.5, 0.0])))

    # 2. Water Sphere (Low IOR: 1.33)
    mat_water = MaterialFactory.create_glass(
        Color(0.9, 0.9, 1.0), Color(0.8, 0.9, 1.0), 
        roughness=0.0, metallicness=0.0, ior=1.33, transmission=1.0
    )
    scene.add_object_by_context(SDF_Material(Sphere(), mat_water), "LeftWater", Transform(np.array([-1.5, 0.5, 0.0])))

    # 3. Heavy Flint Glass Cube (Medium-High IOR: 1.65)
    mat_flint = MaterialFactory.create_glass(
        Color(1.0, 0.9, 0.9), Color(1.0, 1.0, 1.0), 
        roughness=0.01, metallicness=0.0, ior=REFRACTIVE_INDICES["glass_flint_heavy"], transmission=1.0
    )
    # Complex rotation for the cube
    t_flint = Transform(np.array([1.5, 0.5, 0.0]))
    t_flint.rotate(np.deg2rad(30), np.array([0, 1, 0]))
    t_flint.rotate(np.deg2rad(10), np.array([1, 0, 0]))
    scene.add_object_by_context(SDF_Material(Cube(), mat_flint), "RightFlint", t_flint)

    # Checkerboard Floor
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#888888"), roughness=0.8)
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "FloorBase", Transform(np.array([0.0, -10.5, 5.0])))

    # Striped Wall
    for i in range(-5, 6):
        col = Color.from_hex("#FF0000") if i % 2 == 0 else Color.from_hex("#FFFFFF")
        mat_bar = MaterialFactory.create_emissive(col, 2.0)
        scene.add_object_by_context(SDF_Material(Cube(), mat_bar), f"Bar_{i}", Transform(np.array([i, 2.0, -4.0])))

    # Light
    light = Light(color=Color(1.0, 1.0, 1.0), intensity=50.0)
    scene.add_object_by_context(light, "TopLight", Transform(np.array([0.0, 5.0, -3.0])))

    return scene

def get_glass_sculpture_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A complex arrangement of overlapping glass plates and spheres.
    """
    cam_transform = Transform(np.array([3.0, 2.5, -3.0]), np.array([-0.3, 0.7, 0.0]))
    cam = Camera(cam_transform, fov=60.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    cam.transform.look_at(np.array([0, 0.5, 0]))
    
    scene = Scene("glass_sculpture", cam, background_color=Color.from_hex("#200505"))

    # Central Red Glass Sphere
    mat_red_glass = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), Color(1.0, 0.0, 0.2), 
        roughness=0.02, metallicness=0.0, ior=REFRACTIVE_INDICES["glass"], transmission=1.0
    )
    scene.add_object_by_context(SDF_Material(Sphere(), mat_red_glass), "RedOrb", Transform(np.array([0.0, 0.8, 0.0])))

    # Encasing Glass Cube (Clear)
    mat_clear = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), Color(1.0, 1.0, 1.0), 
        roughness=0.0, metallicness=0.0, ior=1.1, transmission=0.9
    )
    scene.add_object_by_context(SDF_Material(Cube(), mat_clear), "ClearBox", Transform(np.array([0.0, 0.8, 0.0])))

    # Back Mirror
    mat_mirror = MaterialFactory.create_specular(Color(1.0, 1.0, 1.0), roughness=0.0, metallicness=1.0)
    scene.add_object_by_context(SDF_Material(Cube(), mat_mirror), "MirrorBack", Transform(np.array([0.0, 2.0, -3.0])))

    # Lights
    l_cyan = Light(color=Color.from_hex("#00FFFF"), intensity=150.0)
    scene.add_object_by_context(l_cyan, "CyanKey", Transform(np.array([4.0, 4.0, -4.0])))
    
    l_rim = Light(color=Color.from_hex("#FFFFFF"), intensity=50.0)
    scene.add_object_by_context(l_rim, "Rim", Transform(np.array([-4.0, 1.0, 0.0])))

    return scene

def get_100_spheres_grid_scene(width: int = 128, height: int = 128) -> Scene:
    """
    Stress test scene: Generates a 10x10 grid of spheres.
    """
    cam_transform = Transform(np.array([-8.0, 8.0, -8.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=60.0, resolution_width=width, resolution_height=height)
    
    sky_colors = [Color.from_hex("#2D2515"), Color.from_hex("#42424E"), Color.from_hex("#5B6791"), Color.from_hex("#87BFC6")]
    scene = Scene("100_spheres_grid", cam, background_color=ColorGradient(sky_colors, np.array([0.0, 0.4, 0.45, 1.0])))

    # Optimization: Create ONE shape instance and reuse it
    shared_sphere_shape = Sphere()

    rows, cols = 10, 10
    spacing = 1.5
    offset_x = -((rows - 1) * spacing) / 2
    offset_z = -((cols - 1) * spacing) / 2

    for r in range(rows):
        for c in range(cols):
            x = offset_x + (r * spacing)
            z = offset_z + (c * spacing)
            y = 0.5 + 0.5 * np.sin(r * 0.5) * np.cos(c * 0.5)
            
            color = Color(r / rows, 0.5, c / cols)
            if (r + c) % 2 == 0:
                mat = MaterialFactory.create_specular(color, roughness=0.2, metallicness=0.9)
            else:
                mat = MaterialFactory.create_diffuse(color, roughness=0.8)
            
            scene.add_object_by_context(SDF_Material(shared_sphere_shape, mat), f"S_{r}_{c}", Transform(np.array([x, y, z])))

    # Floor
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#333333"), roughness=0.5)
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "Floor", Transform(np.array([0.0, -2.0, 0.0])))

    # Light
    sun = Light(color=Color(1.0, 1.0, 0.9), intensity=1000.0)
    scene.add_object_by_context(sun, "Sun", Transform(np.array([10.0, 20.0, -10.0])))
    
    cam.transform.look_at(np.array([0, -1, 0]))
    return scene

def get_low_ior_scene(width: int = 120, height: int = 120) -> Scene:
    """
    Features a sphere with an IOR < 1.0 (0.8).
    """
    cam_transform = Transform(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=60.0, near=0.1, far=100.0, resolution_width=width, resolution_height=height)
    
    scene = Scene("low_ior_anomaly", cam, background_color=Color.from_hex("#000000"))

    # The Low IOR Sphere
    mat_low_ior = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), Color(0.8, 1.0, 0.9),
        roughness=0.0, ior=0.8
    )
    scene.add_object_by_context(SDF_Material(Sphere(), mat_low_ior), "AnomalyObj", Transform.Identity())

    # Background Grid
    tile_shape = Cube()
    mat_red = MaterialFactory.create_emissive(Color.from_hex("#FF4444"), 2.0)
    mat_blue = MaterialFactory.create_emissive(Color.from_hex("#4444FF"), 2.0)

    for x in range(-3, 4):
        for y in range(-3, 4):
            mat = mat_red if (x + y) % 2 == 0 else mat_blue
            scene.add_object_by_context(SDF_Material(tile_shape, mat), f"Tile_{x}_{y}", Transform(np.array([x * 1.5, y * 1.5, -4.0])))

    # Light
    front_light = Light(color=Color(1.0, 1.0, 1.0), intensity=1000.0)
    scene.add_object_by_context(front_light, "Front", Transform(np.array([2.0, 2.0, -3.0])))

    return scene

def get_shape_showcase_scene(width: int = 160, height: int = 120) -> Scene:
    """
    A showcase scene featuring all available 3D shapes arranged in a grid.
    """
    cam_transform = Transform(np.array([0.0, 3.0, -8.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=70.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    scene = Scene("shape_showcase", cam, background_color=Color.from_hex("#1a1a2e"))

    # Materials
    mat_metal = MaterialFactory.create_specular(Color.from_hex("#C0C0C0"), 0.1, 0.9, 0.8, 0.2)
    mat_glass = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, 1.5, 1.0)
    mat_diffuse = MaterialFactory.create_diffuse(Color.from_hex("#FF6B6B"), 0.3)
    mat_emiss = MaterialFactory.create_emissive(Color.from_hex("#4ECDC4"), 1.5)

    # Objects
    scene.add_object_by_context(SDF_Material(Sphere(), mat_metal), "Sphere1", Transform(np.array([-3.0, 1.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_diffuse), "Cube1", Transform(np.array([-1.0, 1.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Sphere(), mat_glass), "Sphere2", Transform(np.array([1.0, 1.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_emiss), "Cube2", Transform(np.array([3.0, 1.0, 0.0])))

    scene.add_object_by_context(SDF_Material(Cylinder(), mat_metal), "Cylinder1", Transform(np.array([-3.0, -1.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Pyramid(), mat_diffuse), "Pyramid1", Transform(np.array([-1.0, -1.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_glass), "Cylinder2", Transform(np.array([1.0, -1.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Pyramid(), mat_emiss), "Pyramid2", Transform(np.array([3.0, -1.0, 0.0])))

    # Assuming SignedDistanceShape3DExtrusion is a valid Shape class
    scene.add_object_by_context(SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_metal), "Prism1", Transform(np.array([-2.0, -3.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Capsule(), mat_glass), "Capsule1", Transform(np.array([0.0, -3.0, 0.0])))
    scene.add_object_by_context(SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_emiss), "Prism2", Transform(np.array([2.0, -3.0, 0.0])))

    scene.add_object_by_context(SDF_Material(Cube(), MaterialFactory.create_diffuse(Color.from_hex("#333333"), 0.8)), "Floor", Transform(np.array([0.0, -5.0, 0.0])))

    # Lights
    l_main = Light(color=Color(1.0, 1.0, 1.0), intensity=2000.0)
    scene.add_object_by_context(l_main, "Main", Transform(np.array([5.0, 5.0, -5.0])))
    
    l_fill = Light(color=Color(0.8, 0.8, 1.0), intensity=1000.0, radius=5)
    scene.add_object_by_context(l_fill, "Fill", Transform(np.array([-5.0, 3.0, 5.0])))

    cam.transform.look_at(np.array([0, -1, 0]))
    return scene

def get_abstract_geometry_scene(width: int = 140, height: int = 100) -> Scene:
    """
    An abstract scene with geometric shapes arranged in a artistic composition.
    """
    cam_transform = Transform(np.array([2.0, 2.0, -5.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=65.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    scene = Scene("abstract_geometry", cam, background_color=Color.from_hex("#0f0f23"))

    # Materials
    mat_transparent = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(0.9, 0.95, 1.0), 0.0, 0.0, 1.4, 0.8)
    mat_mirror = MaterialFactory.create_specular(Color.from_hex("#FFFFFF"), 0.0, 1.0, 1.0, 0.0)
    mat_emiss_red = MaterialFactory.create_emissive(Color.from_hex("#FF1744"), 2.0)
    mat_emiss_blue = MaterialFactory.create_emissive(Color.from_hex("#2979FF"), 2.0)

    # Objects
    scene.add_object_by_context(SDF_Material(Sphere(), mat_transparent), "LargeSphere", Transform(np.array([0.0, 0.0, 0.0])))
    
    t_cyl = Transform(np.array([0.5, 0.0, 0.0]))
    t_cyl.rotate(np.deg2rad(45), np.array([0, 0, 1]))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_mirror), "IntersectCylinder", t_cyl)
    
    scene.add_object_by_context(SDF_Material(Cube(), mat_emiss_red), "FloatCube1", Transform(np.array([-1.5, 1.0, 1.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_emiss_blue), "FloatCube2", Transform(np.array([1.5, -1.0, -1.0])))
    scene.add_object_by_context(SDF_Material(Pyramid(), mat_transparent), "TopPyramid", Transform(np.array([0.0, 1.8, 0.0])))
    scene.add_object_by_context(SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_mirror), "BasePrism", Transform(np.array([0.0, -1.5, 0.0])))

    # Lights
    l_key = Light(color=Color(1.0, 1.0, 1.0), intensity=500.0)
    scene.add_object_by_context(l_key, "Key", Transform(np.array([3.0, 3.0, -3.0])))
    
    l_fill = Light(color=Color(0.5, 0.7, 1.0), intensity=120.0, radius=2)
    scene.add_object_by_context(l_fill, "Fill", Transform(np.array([-3.0, -1.0, 3.0])))

    cam.transform.look_at(np.array([0, 0, 0]))
    return scene

def get_industrial_shapes_scene(width: int = 150, height: int = 100) -> Scene:
    """
    An industrial-themed scene with metallic shapes, pipes, and machinery-like objects.
    """
    cam_transform = Transform(np.array([0.0, 2.0, -6.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=70.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    scene = Scene("industrial_shapes", cam, background_color=Color.from_hex("#2c2c2c"))

    # Materials
    mat_rusty = MaterialFactory.create_specular(Color.from_hex("#8B4513"), 0.4, 0.8, 0.7, 0.3)
    mat_steel = MaterialFactory.create_specular(Color.from_hex("#C0C0C0"), 0.1, 0.9, 0.9, 0.1)
    mat_brass = MaterialFactory.create_specular(Color.from_hex("#B87333"), 0.2, 0.7, 0.8, 0.4)
    mat_concrete = MaterialFactory.create_diffuse(Color.from_hex("#696969"), 0.9)
    mat_e_red = MaterialFactory.create_emissive(Color.from_hex("#FF3B3B"), 3.0)
    mat_e_blue = MaterialFactory.create_emissive(Color.from_hex("#3B7AFF"), 3.0)

    # Base Structure
    scene.add_object_by_context(SDF_Material(Cube(), mat_concrete), "ConcreteFloor", Transform(np.array([0.0, -2.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_steel), "BaseStructure", Transform(np.array([0.0, -1.0, 0.0])))

    # Pipes (Rotated Cylinders)
    t_pipe1 = Transform(np.array([-1.5, 0.0, 0.0]))
    t_pipe1.rotate(np.deg2rad(90), np.array([0, 1, 0]))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_brass), "Pipe1", t_pipe1)
    
    t_pipe2 = Transform(np.array([1.5, 0.5, 0.0]))
    t_pipe2.rotate(np.deg2rad(45), np.array([1, 0, 0]))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_rusty), "Pipe2", t_pipe2)

    # Gears
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_steel), "Gear1", Transform(np.array([0.0, 1.0, 1.0])))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_brass), "Gear2", Transform(np.array([0.0, 1.0, -1.0])))

    # Beams
    t_beam1 = Transform(np.array([-2.0, 0.5, 2.0]))
    t_beam1.rotate(np.deg2rad(30), np.array([0, 1, 0]))
    scene.add_object_by_context(SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_rusty), "Beam1", t_beam1)
    
    t_beam2 = Transform(np.array([2.0, 0.5, -2.0]))
    t_beam2.rotate(np.deg2rad(-30), np.array([0, 1, 0]))
    scene.add_object_by_context(SDF_Material(SignedDistanceShape3DExtrusion(Square()), mat_steel), "Beam2", t_beam2)
    # Panel
    scene.add_object_by_context(SDF_Material(Cube(), mat_steel), "PanelBase", Transform(np.array([0.0, 0.2, 2.5])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_e_red), "Button1", Transform(np.array([-0.3, 0.4, 2.5])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_e_blue), "Button2", Transform(np.array([0.3, 0.4, 2.5])))
    scene.add_object_by_context(SDF_Material(Pyramid(), mat_brass), "Antenna", Transform(np.array([0.0, 0.8, 2.5])))

    # Lighting
    l_overhead = Light(color=Color(1.0, 1.0, 0.9), intensity=250.0)
    scene.add_object_by_context(l_overhead, "Overhead", Transform(np.array([0.0, 4.0, 0.0])))
    
    l_side = Light(color=Color(0.8, 0.8, 1.0), intensity=100.0)
    scene.add_object_by_context(l_side, "Side", Transform(np.array([3.0, 1.0, -3.0])))

    cam.transform.look_at(np.array([0, 0, 0]))
    return scene