import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
py_src_root = os.path.abspath(os.path.join(current_dir, os.pardir))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.insert(0, py_src_root)
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from src.Data.Transform import Transform
from src.Data.Color import Color, ColorGradient
from src.Data.Scene import Scene
from src.Data.Camera import Camera, CameraType
from src.Data.Context import Mesh_Material, SDF_Material
from src.Geometry.SDF import *
from src.Geometry.Operations import *
from src.Geometry.Mesh import *
from src.Lighting.Core import Light
from src.Lighting.Optics import REFRACTIVE_INDICES
from src.Material.Factory import MaterialFactory

def get_minimal_scene(width: int = 64, height: int = 64) -> Scene:
    # Moved camera back slightly and lowered angle for a more dramatic view
    cam_transform = Transform(np.array([0.0, 0.5, -6.0]), np.array([-0.05, 0.0, 0.0]))
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

    # Ground - Lowered slightly to ensure point of contact is visible
    matg = MaterialFactory.create_diffuse(Color.from_hex("#3F3F3F"), 0.9)
    scene.add_object_by_context(SDF_Material(Sphere(100), matg), "Ground Min", Transform(np.array([0.0, -101, 0.0]), scale=np.full(3, 100)))

    # Key light – warm, elevated, primary shadow caster
    key_light = Light(color=Color.from_hex("#FFF0D0"), intensity=280.0, radius=2.0)
    scene.add_object_by_context(key_light, "KeyLight", Transform(np.array([3.0, 4.0, -2.0])))

    # Fill light – cool, opposite side, softer falloff
    fill_light = Light(color=Color.from_hex("#C0D8FF"), intensity=100.0, radius=5.0)
    scene.add_object_by_context(fill_light, "FillLight", Transform(np.array([-3.0, 2.5, -1.0])))

    return scene

def get_gradient_scene(width: int = 64, height: int = 64) -> Scene:
    # Adjusted Camera to look at a centralized group of objects
    cam_transform = Transform(np.array([0.0, 2.0, -6.0]), np.array([-0.2, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=60.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )

    sky_colors = [Color.from_hex("#2D2515"), Color.from_hex("#42424E"), Color.from_hex("#5B6791"), Color.from_hex("#87BFC6")]
    sky_positions = np.array([0.0, 0.4, 0.45, 1.0])
    scene = Scene("gradient_scene", cam, background_color=ColorGradient(sky_colors, sky_positions))

    # --- Objects Rearranged into a tight composition ---
    
    # Central Reflective Sphere
    mat_metal = MaterialFactory.create_specular(Color.from_hex("#47505C"), 0.2, 0.9, 1.0, 1.0)
    sph_1 = SDF_Material(Sphere(1.0), mat_metal)
    scene.add_object_by_context(sph_1, "Reflective Sphere", Transform(np.array([0.0, 1.0, 0.0])))

    # Matte Box (Left)
    mat_matte = MaterialFactory.create_diffuse(Color.from_hex("#C27A23"), 0.8)
    bx_1 = SDF_Material(Cube(), mat_matte)
    # Scaled down and placed to the left
    scene.add_object_by_context(bx_1, "Matte Box", Transform(np.array([-1.8, 0.7, -0.5]), np.array([0.0, np.deg2rad(35), 0.0]), np.full(3, 0.7)))

    # Floating Emissive Orb (Above/Right)
    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#EE1717"), 4.0)
    sph_2 = SDF_Material(Sphere(0.3), mat_glow)
    scene.add_object_by_context(sph_2, "Emissive Orb", Transform(np.array([0.8, 2.2, 0.5])))

    # Golden Cylinder (Right)
    mat_cylinder = MaterialFactory.create_specular(Color.from_hex("#FFD700"), 0.1, 0.8, 0.9, 0.5)
    cyl_1 = SDF_Material(Cylinder(), mat_cylinder)
    scene.add_object_by_context(cyl_1, "Golden Cylinder", Transform(np.array([1.8, 0.8, -0.2]), np.array([0.0, np.deg2rad(-20), np.deg2rad(10)]), np.array([0.5, 0.8, 0.5])))

    # Wooden Pyramid (Back Left)
    mat_pyramid = MaterialFactory.create_diffuse(Color.from_hex("#8B4513"), 0.7)
    pyr_1 = SDF_Material(Pyramid(), mat_pyramid)
    scene.add_object_by_context(pyr_1, "Wooden Pyramid", Transform(np.array([-1.0, 0.5, 1.5]), rotation=np.array([0, np.deg2rad(45), 0])))

    # Lights
    key_light = Light(color=Color.from_hex("#FFEDC7"), intensity=320.0, radius=2.0)
    scene.add_object_by_context(key_light, "Key Light", Transform(np.array([4.0, 5.0, -3.0])))

    fill_light = Light(color=Color.from_hex("#C7E5FF"), intensity=120.0, radius=4.0)
    scene.add_object_by_context(fill_light, "Fill Light", Transform(np.array([-5.0, 2.0, -5.0])))

    cam.transform.look_at(np.array([0, 1, 0]), np.array([0, 1, 0]))
    return scene

def get_emissive_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 1.0, -4.0]), np.zeros(3))
    cam = Camera(
        cam_transform,
        fov=60.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("emissive_scene", cam, background_color=Color.from_hex("#000000"))

    # Objects
    # Moved glowing sphere to be clearly reflected in the mirror sphere
    mat_glow = MaterialFactory.create_emissive(Color.from_hex("#FFEA62"), 5.0)
    v_emissive = SDF_Material(Sphere(0.3), mat_glow)
    scene.add_object_by_context(v_emissive, "GlowingSphere", Transform(np.array([1.2, 0.5, -0.5])))

    mat_reflect = MaterialFactory.create_specular(Color.from_hex("#6B6666"), roughness=0.1, metallicness=0.9, specular_intensity=1.0)
    v_mirror = SDF_Material(Sphere(1.0), mat_reflect)
    scene.add_object_by_context(v_mirror, "MirrorSphere", Transform(np.array([-0.5, 1.0, 0.5])))

    matg = MaterialFactory.create_diffuse(Color.from_hex("#202020"), roughness=0.8)
    scene.add_object_by_context(SDF_Material(Sphere(100), matg), "Ground", Transform(np.array([0.0, -101, 0.0]), scale=np.full(3, 100)))

    # Lights – weak key so the mirror has something to reflect;
    # dim tinted fill preserved to keep the emissive glow as the hero.
    key = Light(color=Color.from_hex("#FFFFFF"), intensity=80.0, radius=1.5)
    scene.add_object_by_context(key, "KeyEmiss", Transform(np.array([2.0, 3.0, -2.0])))

    fill = Light(color=Color.from_hex("#333344"), intensity=60.0, radius=10.0)
    scene.add_object_by_context(fill, "FillEmiss", Transform(np.array([-4.0, 2.0, -3.0])))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))

    return scene

def get_lit_studio_scene(width: int = 100, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([0.0, 0.8, -3.0]), np.array([-0.1, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=70.0, near=0.1, far=100.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("lit_studio", cam, background_color=Color.from_hex("#BEC2CF"))

    # Objects - Center them better
    mat1 = MaterialFactory.create_specular(Color.from_hex("#FFB86B"), 0.2, 0.1, 0.9, 0)
    v_s1 = SDF_Material(Sphere(0.4), mat1)
    scene.add_object_by_context(v_s1, "StudioBallA", Transform(np.array([-0.4, 0.4, 0.0])))

    mat2 = MaterialFactory.create_specular(Color.from_hex("#6B9BFF"), 0.2, 0.4, 0.9, 0)
    v_s2 = SDF_Material(Sphere(0.45), mat2)
    scene.add_object_by_context(v_s2, "StudioBallB", Transform(np.array([0.4, 0.45, 0.2])))

    # Brought background closer and angled it up
    mat_plane = MaterialFactory.create_diffuse(Color.from_hex("#C1CBD0"), roughness=1.0)
    v_plane = SDF_Material(ShapeExtrusion(Rectangle(np.array([10, 10])), 0.1), mat_plane)
    scene.add_object_by_context(v_plane, "StudioBack", Transform(np.array([0.0, 0.0, 3.0])))

    # Lights – balanced three-point studio rig
    key = Light(color=Color.from_hex("#EEE0BA"), intensity=400.0, radius=3.0)
    scene.add_object_by_context(key, "StudioKey", Transform(np.array([2.5, 2.5, -2.0])))

    rim = Light(color=Color.from_hex("#DC97C5"), intensity=150.0, radius=1.5)
    scene.add_object_by_context(rim, "StudioRim", Transform(np.array([-3.0, 1.0, 1.0])))

    fill = Light(color=Color.from_hex("#C7DBD8"), intensity=180.0, radius=3.0)
    scene.add_object_by_context(fill, "StudioFill", Transform(np.array([0.0, -2.5, -2.0])))

    cam.transform.look_at(np.array([0, 0.4, 0]))

    return scene

def get_rgb_cornell_box_scene(width: int = 126, height: int = 126) -> Scene:
    # 1. Camera Setup
    cam_transform = Transform(np.array([0.0, 2.75, -7.5]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform,
        fov=55.0, near=1, far=1000,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene("rgb_cornell_box", cam, background_color=Color(0.0, 0.0, 0.0))

    # 2. Materials
    mat_white = MaterialFactory.create_diffuse(Color.from_hex("#E0E0E0"), 1.0)
    mat_red   = MaterialFactory.create_diffuse(Color.from_hex("#B03030"), 1.0)
    mat_green = MaterialFactory.create_diffuse(Color.from_hex("#30B030"), 1.0)
    mat_blue  = MaterialFactory.create_diffuse(Color.from_hex("#3036B0"), 1.0)
    mat_mirror= MaterialFactory.create_specular(Color.from_hex("#FFFFFF"), 0.1, 1.0, 0)
    mat_glass = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, REFRACTIVE_INDICES["glass"], 0)
    mat_cyl   = MaterialFactory.create_specular(Color.from_hex("#FFD700"), 0.2, 0.7, 0.9, 0.5)

    # 3. Room Dimensions
    room_x = (-3.0, 3.0)
    room_y = (-0.5, 5.5)
    room_z = (-3.0, 3.0)
    
    w = room_x[1] - room_x[0]
    h = room_y[1] - room_y[0]
    d = room_z[1] - room_z[0]
    
    y_center = (room_y[0] + room_y[1]) / 2.0
    floor_y = room_y[0]

    # 4. Room Geometry
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Rectangle(np.array([w, d])), 0.1), mat_white), "Floor", Transform(np.array([0.0, room_y[0], 0.0]), np.array([np.deg2rad(90), 0.0, 0.0])))
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Rectangle(np.array([w, d])), 0.1), mat_white), "Ceiling", Transform(np.array([0.0, room_y[1], 0.0]), np.array([np.deg2rad(90), 0.0, 0.0])))
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Rectangle(np.array([w, h])), 0.1), mat_blue), "BackWall", Transform(np.array([0.0, y_center, room_z[1]]), np.array([0.0, 0.0, 0.0])))
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Rectangle(np.array([d, h])), 0.1), mat_red), "LeftWall", Transform(np.array([room_x[0], y_center, 0.0]), np.array([0.0, np.deg2rad(90), 0.0])))
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Rectangle(np.array([d, h])), 0.1), mat_green), "RightWall", Transform(np.array([room_x[1], y_center, 0.0]), np.array([0.0, np.deg2rad(90), 0.0])))

    # 5. Objects 
    # Adjusted Heights: Unit objects usually extend +/- 1. So Center Y should be Floor Y + 1 * Scale Y.

    # Tall Box (Rotated) - Scale 1.5 in Height -> Total Height 3. Center at floor + 1.5
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_white), 
        "TallBox", 
        Transform(
            np.array([-1.5, floor_y + 1.5, 0.5]), 
            np.array([0.0, np.deg2rad(20), 0.0]), 
            np.array([0.8, 1.5, 0.8])
        )
    )

    # Mirror Sphere - Radius 0.8
    scene.add_object_by_context(
        SDF_Material(Sphere(0.8), mat_mirror), 
        "MirrorBall", 
        Transform(np.array([1.2, floor_y + 0.8, 1.2])) 
    )

    # Glass Cube - Scale 0.7
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_glass), 
        "GlassCube", 
        Transform(np.array([0.0, floor_y + 0.7, -1.0]), np.array([0.0, np.deg2rad(-15), 0.0]), np.full(3, 0.7))
    )
    
    # Gold Cylinder
    scene.add_object_by_context(
        SDF_Material(Cylinder(), mat_cyl), 
        "CylinderObj", 
        Transform(np.array([1.5, floor_y + 0.6, -1.5]), rotation=np.array([0, np.deg2rad(45), 0]), scale=np.array([0.6, 0.6, 0.6]))
    )

    # 6. Lights – area ceiling panel + subtle floor bounce
    ceiling_light = Light(color=Color.from_hex("#FFECDE"), intensity=1200.0, radius=3.0)
    scene.add_object_by_context(ceiling_light, "CeilingLight", Transform(np.array([0.0, room_y[1] - 0.5, 0.0])))

    floor_bounce = Light(color=Color.from_hex("#FFE8CC"), intensity=80.0, radius=4.0)
    scene.add_object_by_context(floor_bounce, "FloorBounce", Transform(np.array([0.0, floor_y + 0.1, 0.0])))

    cam.transform.look_at(np.array([0, 2, 0]))

    return scene

def get_cyberpunk_scene(width: int = 140, height: int = 100) -> Scene:
    """
    A dark, wet street scene with neon lights, verticality, and fog.
    """
    # Low angle camera looking up/down the street
    cam_transform = Transform(np.array([0.0, 1.0, -8.0]), np.array([-0.1, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=70.0, resolution_width=width, resolution_height=height)
    
    # Dark purple/black background (Smoggy night)
    scene = Scene("cyberpunk_street", cam, background_color=Color.from_hex("#05000a"))

    # Materials
    mat_asphalt_wet = MaterialFactory.create_specular(Color.from_hex("#111111"), roughness=0.2, metallicness=0.1)
    mat_concrete_dark = MaterialFactory.create_diffuse(Color.from_hex("#222222"), roughness=0.9)
    mat_neon_pink = MaterialFactory.create_emissive(Color.from_hex("#FF0099"), 3.0)
    mat_neon_cyan = MaterialFactory.create_emissive(Color.from_hex("#00FFFF"), 3.0)

    # Floor (Wet Road)
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_asphalt_wet), 
        "Road", 
        Transform(np.array([0.0, -1.0, 0.0]), scale=np.array([4.0, 0.1, 20.0]))
    )

    # Building Loop
    num_buildings = 5
    spacing = 4.0
    
    for i in range(num_buildings):
        z_pos = (i * spacing) - 5.0
        
        # Left Buildings (Cyan Theme)
        h_left = 3.0 + np.sin(i) # Varied height
        scene.add_object_by_context(
            SDF_Material(Cube(), mat_concrete_dark), 
            f"BuildL_{i}", 
            Transform(np.array([-4.5, h_left/2, z_pos]), scale=np.array([2.0, h_left, 1.5]))
        )
        # Left Neon Sign
        scene.add_object_by_context(
            SDF_Material(Cube(), mat_neon_cyan),
            f"SignL_{i}",
            Transform(np.array([-2.6, 1.5, z_pos]), scale=np.array([0.1, 1.0, 0.1]))
        )

        # Right Buildings (Pink Theme)
        h_right = 4.0 + np.cos(i)
        scene.add_object_by_context(
            SDF_Material(Cube(), mat_concrete_dark), 
            f"BuildR_{i}", 
            Transform(np.array([4.5, h_right/2, z_pos]), scale=np.array([2.0, h_right, 1.5]))
        )
        # Right Neon Sign (Horizontal bar)
        scene.add_object_by_context(
            SDF_Material(Cube(), mat_neon_pink),
            f"SignR_{i}",
            Transform(np.array([2.6, 2.5, z_pos]), scale=np.array([0.1, 0.1, 1.2]))
        )

    # Lighting – pink hero sign slightly brighter; dim ground fill for wet-road reflections
    scene.add_object_by_context(
        Light(color=Color.from_hex("#FF0099"), intensity=280.0, radius=4.0),
        "PinkLight", Transform(np.array([3.0, 2.0, 0.0]))
    )
    scene.add_object_by_context(
        Light(color=Color.from_hex("#00FFFF"), intensity=180.0, radius=4.0),
        "CyanLight", Transform(np.array([-3.0, 2.0, 5.0]))
    )
    # Low fill so the road surface picks up colour
    scene.add_object_by_context(
        Light(color=Color.from_hex("#1A0A2E"), intensity=60.0, radius=8.0),
        "GroundFill", Transform(np.array([0.0, 0.5, 2.0]))
    )

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

    # Lights – main is the dominant key; fill lifts shadows from the side
    l_main = Light(color=Color(1.0, 1.0, 1.0), intensity=400.0, radius=3.0)
    scene.add_object_by_context(l_main, "Main", Transform(np.array([0.0, 5.0, -5.0])))
    
    l_fill = Light(color=Color(0.8, 0.8, 1.0), intensity=150.0, radius=5.0)
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

    # Lights – backlight illuminates the striped bars seen through glass;
    # front fill prevents the camera-facing hemispheres from going dead black.
    l_back = Light(color=Color(1.0, 1.0, 1.0), intensity=300.0, radius=4.0)
    scene.add_object_by_context(l_back, "BackLight", Transform(np.array([0.0, 2.0, 4.0])))

    l_front = Light(color=Color(0.9, 0.9, 1.0), intensity=120.0, radius=3.0)
    scene.add_object_by_context(l_front, "FrontLight", Transform(np.array([2.0, 3.0, -3.0])))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))
    return scene

def get_sunset_monolith_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A cinematic scene featuring a mysterious monolith against a retro-wave sunset.
    """
    # Camera low to the ground, looking slightly up
    cam_transform = Transform(np.array([0.0, 1.0, -8.0]), np.array([-0.1, 0.0, 0.0]))
    cam = Camera(
        cam_transform, fov=60.0, near=0.1, far=150.0, 
        resolution_width=width, resolution_height=height
    )
    
    # Retro-wave gradient background (Purple to Orange)
    bg_colors = [Color.from_hex("#240046"), Color.from_hex("#7B2CBF"), Color.from_hex("#FF9E00")]
    scene = Scene("sunset_monolith", cam, background_color=ColorGradient(bg_colors, np.array([0.0, 0.5, 1.0])))

    # Materials
    # The Monolith: Pitch black, highly reflective, very smooth
    mat_obsidian = MaterialFactory.create_specular(
        Color.from_hex("#000000"), roughness=0.05, metallicness=0.9
    )
    
    # The Sun: Glowing warm sphere
    mat_sun = MaterialFactory.create_emissive(Color.from_hex("#FF4500"), 4.0)
    
    # The Ground: Dark, rocky/sandy
    mat_sand = MaterialFactory.create_diffuse(Color.from_hex("#1A1A1A"), roughness=0.9)

    # 1. The Monolith (Scale 1:4:9 ratio roughly)
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_obsidian),
        "Monolith",
        Transform(np.array([0.0, 2.0, 0.0]), scale=np.array([1.0, 4.0, 0.5]))
    )

    # 2. The "Sun" (Geometric sphere behind the monolith)
    # Placed far back to silhouette the monolith
    scene.add_object_by_context(
        SDF_Material(Sphere(), mat_sun),
        "RetroSun",
        Transform(np.array([0.0, 3.0, 15.0]), scale=np.full(3, 4.0))
    )

    # 3. Ground Terrain
    # Using a flattened, scaled cube for the floor
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_sand),
        "DesertFloor",
        Transform(np.array([0.0, -3.0, 0.0]), scale=np.array([20.0, 1.0, 20.0]))
    )

    # Lighting
    # Backlight (Rim lighting for the monolith)
    scene.add_object_by_context(
        Light(color=Color.from_hex("#FF6600"), intensity=800.0, radius=10.0),
        "BackLight",
        Transform(np.array([0.0, 4.0, 10.0]))
    )

    # Purple fill – lifted so the monolith front face reads purple, not pure black
    scene.add_object_by_context(
        Light(color=Color.from_hex("#5500AA"), intensity=120.0, radius=10.0),
        "FrontFill",
        Transform(np.array([-5.0, 2.0, -10.0]))
    )

    return scene



def get_scifi_corridor_scene(width: int = 160, height: int = 120) -> Scene:
    """
    A futuristic hallway using repetitive geometry and emissive lighting 
    to create a sense of depth and speed.
    """
    # Camera placed at the start of the tunnel, looking forward
    cam_transform = Transform(np.array([0.0, 1.5, -10.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(
        cam_transform, fov=80.0, near=0.1, far=100.0, 
        resolution_width=width, resolution_height=height
    )
    
    scene = Scene("scifi_corridor", cam, background_color=Color.from_hex("#050505"))

    # Materials
    mat_floor = MaterialFactory.create_specular(Color.from_hex("#222222"), roughness=0.2, metallicness=0.8)
    mat_wall = MaterialFactory.create_diffuse(Color.from_hex("#333333"), roughness=0.5)
    mat_light_strip = MaterialFactory.create_emissive(Color.from_hex("#00FFFF"), 5.0) # Cyan Glow
    mat_pillar = MaterialFactory.create_specular(Color.from_hex("#555555"), roughness=0.3, metallicness=0.6)

    # Floor and Ceiling (Long flattened cubes)
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_floor), 
        "Floor", 
        Transform(np.array([0.0, -1.0, 20.0]), scale=np.array([5.0, 0.1, 40.0]))
    )
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_wall), 
        "Ceiling", 
        Transform(np.array([0.0, 4.0, 20.0]), scale=np.array([5.0, 0.1, 40.0]))
    )

    # Repeating Ribs/Pillars
    # We create a loop to place 'ribs' along the Z-axis
    num_ribs = 8
    spacing = 4.0
    start_z = -8.0

    for i in range(num_ribs):
        z_pos = start_z + (i * spacing)
        
        # Left Pillar
        scene.add_object_by_context(
            SDF_Material(Cylinder(), mat_pillar), 
            f"PillarL_{i}", 
            Transform(np.array([-3.0, 1.5, z_pos]), scale=np.array([0.4, 2.5, 0.4]))
        )
        
        # Right Pillar
        scene.add_object_by_context(
            SDF_Material(Cylinder(), mat_pillar), 
            f"PillarR_{i}", 
            Transform(np.array([3.0, 1.5, z_pos]), scale=np.array([0.4, 2.5, 0.4]))
        )

        # Overhead Beam
        scene.add_object_by_context(
            SDF_Material(Cube(), mat_pillar),
            f"Beam_{i}",
            Transform(np.array([0.0, 3.5, z_pos]), scale=np.array([3.2, 0.3, 0.3]))
        )

        # Emissive Light Strips (Floor markers)
        scene.add_object_by_context(
            SDF_Material(Cube(), mat_light_strip),
            f"LightStrip_{i}",
            Transform(np.array([0.0, -0.85, z_pos]), scale=np.array([2.5, 0.05, 0.1]))
        )

    # Lighting
    # Near point light – lifts the immediate pillars and floor strips
    scene.add_object_by_context(
        Light(color=Color(1.0, 1.0, 1.0), intensity=300.0, radius=3.0),
        "PlayerLight",
        Transform(np.array([0.0, 2.0, -8.0]))
    )

    # Distant blue – atmospheric pull toward the tunnel end
    scene.add_object_by_context(
        Light(color=Color(0.0, 0.5, 1.0), intensity=400.0, radius=20.0),
        "EndLight",
        Transform(np.array([0.0, 2.0, 30.0]))
    )

    return scene

def get_pastel_blocks_scene(width: int = 120, height: int = 120) -> Scene:
    # Lowered camera angle slightly to make the stack look taller
    cam_transform = Transform(np.array([0.0, 2.5, -6.0]), np.array([-0.05, 0.0, 0.0]))
    cam = Camera(
        cam_transform, fov=55.0, near=0.1, far=50.0,
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    scene = Scene("pastel_blocks", cam, background_color=Color.from_hex("#F0F4F8"))

    # Materials
    mat_pink = MaterialFactory.create_diffuse(Color.from_hex("#FFB7B2"), roughness=0.6)
    mat_mint = MaterialFactory.create_diffuse(Color.from_hex("#B5EAD7"), roughness=0.6)
    mat_purple = MaterialFactory.create_diffuse(Color.from_hex("#E2F0CB"), roughness=0.6)
    mat_white = MaterialFactory.create_diffuse(Color.from_hex("#FFFFFF"), roughness=0.9)

    # Objects - Tighter stacking
    scene.add_object_by_context(SDF_Material(Cube(), mat_white), "Floor", Transform(np.array([0.0, -1.2, 0.0]), scale=np.array([10, 0.2, 10])))
    
    # Base rotated slightly
    scene.add_object_by_context(SDF_Material(Cube(), mat_mint), "BaseObj", Transform(np.array([0.0, 0.0, 0.0]), np.array([0.0, np.deg2rad(25), 0.0])))
    
    # Sphere resting perfectly on top
    scene.add_object_by_context(SDF_Material(Sphere(1.0), mat_pink), "MidObj", Transform(np.array([0.0, 2.0, 0.0])))
    
    # Top cube balancing on the sphere, tilted
    scene.add_object_by_context(SDF_Material(Cube(0.8), mat_purple), "TopObj", Transform(np.array([0.2, 3.5, 0.2]), np.array([np.deg2rad(30), np.deg2rad(45), np.deg2rad(10)])))

    # Lights – boosted so pastels stay bright and airy
    key = Light(color=Color.from_hex("#FFFBEB"), intensity=320.0, radius=5.0)
    scene.add_object_by_context(key, "Key", Transform(np.array([4.0, 6.0, -4.0])))

    fill = Light(color=Color.from_hex("#E6E6FA"), intensity=180.0, radius=5.0)
    scene.add_object_by_context(fill, "Fill", Transform(np.array([-4.0, 2.0, -4.0])))

    cam.transform.look_at(np.array([0, 1.5, 0]))
    return scene

def get_glass_prism_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([0.0, 2.0, -5.5]), np.array([-0.1, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=65.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    
    scene = Scene("glass_prism_row", cam, background_color=Color.from_hex("#101015"))

    # 1. Diamond Sphere (Center)
    mat_diamond = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), Color(1.0, 1.0, 1.0), 
        roughness=0.0, metallicness=0.0, ior=REFRACTIVE_INDICES["diamond"], transmission=1.0
    )
    scene.add_object_by_context(SDF_Material(Sphere(), mat_diamond), "CenterDiamond", Transform(np.array([0.0, 0.5, 0.0])))

    # 2. Water Sphere (Left)
    mat_water = MaterialFactory.create_glass(
        Color(0.9, 0.9, 1.0), Color(0.8, 0.9, 1.0), 
        roughness=0.0, metallicness=0.0, ior=1.33, transmission=1.0
    )
    scene.add_object_by_context(SDF_Material(Sphere(), mat_water), "LeftWater", Transform(np.array([-2.2, 0.5, 0.5])))

    # 3. Heavy Flint Glass Cube (Right)
    mat_flint = MaterialFactory.create_glass(
        Color(1.0, 0.9, 0.9), Color(1.0, 1.0, 1.0), 
        roughness=0.01, metallicness=0.0, ior=REFRACTIVE_INDICES["glass_flint_heavy"], transmission=1.0
    )
    # Rotated to catch light refractions better
    t_flint = Transform(np.array([2.2, 0.5, 0.5]))
    t_flint.rotate(np.deg2rad(45), np.array([0, 1, 0]))
    t_flint.rotate(np.deg2rad(15), np.array([1, 0, 1]))
    scene.add_object_by_context(SDF_Material(Cube(), mat_flint), "RightFlint", t_flint)

    # Checkerboard Floor - Lowered further to allow looking "through" objects
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#888888"), roughness=0.8)
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "FloorBase", Transform(np.array([0.0, -12.0, 5.0]), scale=np.array([20, 10, 20])))

    # Striped Wall - Brought closer to be more visible through refraction
    for i in range(-5, 6):
        col = Color.from_hex("#FF0000") if i % 2 == 0 else Color.from_hex("#FFFFFF")
        mat_bar = MaterialFactory.create_emissive(col, 3.0)
        scene.add_object_by_context(SDF_Material(Cube(), mat_bar), f"Bar_{i}", Transform(np.array([i * 1.2, 2.0, 4.0]), scale=np.array([0.5, 5.0, 0.5])))

    # Light – key from the front-top with area radius; backlight to illuminate
    # the striped wall visible through the refracting objects.
    key = Light(color=Color(1.0, 1.0, 1.0), intensity=250.0, radius=3.0)
    scene.add_object_by_context(key, "TopLight", Transform(np.array([2.0, 5.0, -4.0])))

    back = Light(color=Color(1.0, 0.95, 0.9), intensity=200.0, radius=5.0)
    scene.add_object_by_context(back, "BackLight", Transform(np.array([0.0, 2.0, 6.0])))

    cam.transform.look_at(np.array([0, 0.5, 0]))
    return scene

def get_glass_sculpture_scene(width: int = 120, height: int = 120) -> Scene:
    cam_transform = Transform(np.array([3.0, 2.5, -3.5]), np.array([-0.3, 0.7, 0.0]))
    cam = Camera(cam_transform, fov=60.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    
    scene = Scene("glass_sculpture", cam, background_color=Color.from_hex("#200505"))

    # Central Red Glass Sphere
    # Reduced radius to 0.75 so it sits visibly INSIDE the cube (radius 1.0) without touching walls
    mat_red_glass = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), Color(1.0, 0.0, 0.2), 
        roughness=0.02, metallicness=0.0, ior=REFRACTIVE_INDICES["glass"], transmission=1.0
    )
    scene.add_object_by_context(SDF_Material(Sphere(0.75), mat_red_glass), "RedOrb", Transform(np.array([0.0, 0.0, 0.0])))

    # Encasing Glass Cube (Clear)
    mat_clear = MaterialFactory.create_glass(
        Color(1.0, 1.0, 1.0), Color(1.0, 1.0, 1.0), 
        roughness=0.0, metallicness=0.0, ior=1.1, transmission=0.95
    )
    # Rotated slightly to show volume
    scene.add_object_by_context(SDF_Material(Cube(), mat_clear), "ClearBox", Transform(np.array([0.0, 0.0, 0.0]), np.array([np.deg2rad(15), np.deg2rad(15), 0])))

    # Back Mirror - Positioned to reflect back of the cube
    mat_mirror = MaterialFactory.create_specular(Color(1.0, 1.0, 1.0), roughness=0.0, metallicness=1.0)
    scene.add_object_by_context(SDF_Material(Cube(), mat_mirror), "MirrorBack", Transform(np.array([-2.0, 1.0, 4.0]), np.array([0, np.deg2rad(-20), 0]), scale=np.array([5, 5, 0.1])))

    # Lights – area radii added; rim lifted to illuminate cube edges
    l_cyan = Light(color=Color.from_hex("#00FFFF"), intensity=200.0, radius=2.5)
    scene.add_object_by_context(l_cyan, "CyanKey", Transform(np.array([4.0, 4.0, -4.0])))
    
    l_rim = Light(color=Color.from_hex("#FFFFFF"), intensity=150.0, radius=2.0)
    scene.add_object_by_context(l_rim, "Rim", Transform(np.array([-4.0, 0.0, -2.0])))

    cam.transform.look_at(np.array([0, 0, 0]))
    return scene

def get_100_spheres_grid_scene(width: int = 128, height: int = 128) -> Scene:
    # Moved camera further back to see the whole grid
    cam_transform = Transform(np.array([-12.0, 10.0, -12.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=50.0, resolution_width=width, resolution_height=height)
    
    sky_colors = [Color.from_hex("#2D2515"), Color.from_hex("#42424E"), Color.from_hex("#5B6791"), Color.from_hex("#87BFC6")]
    scene = Scene("100_spheres_grid", cam, background_color=ColorGradient(sky_colors, np.array([0.0, 0.4, 0.45, 1.0])))

    shared_sphere_shape = Sphere()

    rows, cols = 10, 10
    spacing = 1.5
    offset_x = -((rows - 1) * spacing) / 2
    offset_z = -((cols - 1) * spacing) / 2

    for r in range(rows):
        for c in range(cols):
            x = offset_x + (r * spacing)
            z = offset_z + (c * spacing)
            # Wavy height pattern
            y = 0.5 * np.sin(r * 0.5) + 0.5 * np.cos(c * 0.5)
            
            color = Color(r / rows, 0.5, c / cols)
            if (r + c) % 2 == 0:
                mat = MaterialFactory.create_specular(color, roughness=0.2, metallicness=0.9)
            else:
                mat = MaterialFactory.create_diffuse(color, roughness=0.8)
            
            scene.add_object_by_context(SDF_Material(shared_sphere_shape, mat), f"S_{r}_{c}", Transform(np.array([x, y, z])))

    # Floor
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#333333"), roughness=0.5)
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "Floor", Transform(np.array([0.0, -2.5, 0.0]), scale=np.array([20, 1, 20])))

    # Light – sun + cool sky fill so back-facing spheres aren't totally dark
    sun = Light(color=Color(1.0, 1.0, 0.9), intensity=1200.0, radius=3.0)
    scene.add_object_by_context(sun, "Sun", Transform(np.array([10.0, 20.0, -10.0])))

    sky_fill = Light(color=Color(0.5, 0.6, 0.8), intensity=200.0, radius=15.0)
    scene.add_object_by_context(sky_fill, "SkyFill", Transform(np.array([-10.0, 15.0, 10.0])))
    
    cam.transform.look_at(np.array([0, 0, 0]))
    return scene

def get_low_ior_scene(width: int = 120, height: int = 120) -> Scene:
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
            # Placed further back so they are clearly visible through the sphere
            scene.add_object_by_context(SDF_Material(tile_shape, mat), f"Tile_{x}_{y}", Transform(np.array([x * 1.5, y * 1.5, 3.0]), scale=np.array([0.6, 0.6, 0.1])))

    # Light – front key with radius + back fill to illuminate the grid tiles
    front_light = Light(color=Color(1.0, 1.0, 1.0), intensity=600.0, radius=3.0)
    scene.add_object_by_context(front_light, "Front", Transform(np.array([2.0, 2.0, -5.0])))

    back_fill = Light(color=Color(0.8, 0.8, 1.0), intensity=250.0, radius=4.0)
    scene.add_object_by_context(back_fill, "BackFill", Transform(np.array([-2.0, 0.0, 5.0])))

    return scene

def get_shape_showcase_scene(width: int = 160, height: int = 120) -> Scene:
    # Moved camera back and up for a better overview of the grid
    cam_transform = Transform(np.array([0.0, 6.0, -9.0]), np.array([-0.5, 0.0, 0.0]))
    cam = Camera(cam_transform, fov=60.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    scene = Scene("shape_showcase", cam, background_color=Color.from_hex("#1a1a2e"))

    # Materials
    mat_metal = MaterialFactory.create_specular(Color.from_hex("#C0C0C0"), 0.1, 0.9, 0.8, 0.2)
    mat_glass = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(1.0, 1.0, 1.0), 0.0, 0.0, 1.5, 1.0)
    mat_diffuse = MaterialFactory.create_diffuse(Color.from_hex("#FF6B6B"), 0.3)
    mat_emiss = MaterialFactory.create_emissive(Color.from_hex("#4ECDC4"), 1.5)

    # Objects organized in a tighter grid
    # Row 1
    scene.add_object_by_context(SDF_Material(Sphere(), mat_metal), "Sphere1", Transform(np.array([-3.0, 0.0, -2.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_diffuse), "Cube1", Transform(np.array([-1.0, 0.0, -2.0])))
    scene.add_object_by_context(SDF_Material(Sphere(), mat_glass), "Sphere2", Transform(np.array([1.0, 0.0, -2.0])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_emiss), "Cube2", Transform(np.array([3.0, 0.0, -2.0])))

    # Row 2
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_metal), "Cylinder1", Transform(np.array([-3.0, 0.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Pyramid(), mat_diffuse), "Pyramid1", Transform(np.array([-1.0, 0.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_glass), "Cylinder2", Transform(np.array([1.0, 0.0, 0.0])))
    scene.add_object_by_context(SDF_Material(Pyramid(), mat_emiss), "Pyramid2", Transform(np.array([3.0, 0.0, 0.0])))

    # Row 3
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Square()), mat_metal), "Prism1", Transform(np.array([-2.0, 0.0, 2.0])))
    scene.add_object_by_context(SDF_Material(Capsule(), mat_glass), "Capsule1", Transform(np.array([0.0, 0.0, 2.0])))
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Square()), mat_emiss), "Prism2", Transform(np.array([2.0, 0.0, 2.0])))

    scene.add_object_by_context(SDF_Material(Cube(), MaterialFactory.create_diffuse(Color.from_hex("#333333"), 0.8)), "Floor", Transform(np.array([0.0, -2.0, 0.0]), scale=np.array([10, 1, 10])))

    # Lights – key + cool fill so all three rows are readable
    l_main = Light(color=Color(1.0, 1.0, 1.0), intensity=800.0, radius=4.0)
    scene.add_object_by_context(l_main, "Main", Transform(np.array([5.0, 8.0, -5.0])))

    l_fill = Light(color=Color(0.6, 0.7, 1.0), intensity=250.0, radius=6.0)
    scene.add_object_by_context(l_fill, "Fill", Transform(np.array([-5.0, 4.0, 4.0])))
    
    cam.transform.look_at(np.array([0, 0, 0]))
    return scene

def get_abstract_geometry_scene(width: int = 140, height: int = 100) -> Scene:
    cam_transform = Transform(np.array([3.0, 3.0, -5.0]), np.array([-0.3, 0.3, 0.0]))
    cam = Camera(cam_transform, fov=65.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    scene = Scene("abstract_geometry", cam, background_color=Color.from_hex("#0f0f23"))

    # Materials
    mat_transparent = MaterialFactory.create_glass(Color.from_hex("#FFFFFF"), Color(0.9, 0.95, 1.0), 0.0, 0.0, 1.4, 0.8)
    mat_mirror = MaterialFactory.create_specular(Color.from_hex("#FFFFFF"), 0.0, 1.0, 1.0, 0.0)
    mat_emiss_red = MaterialFactory.create_emissive(Color.from_hex("#FF1744"), 2.0)
    mat_emiss_blue = MaterialFactory.create_emissive(Color.from_hex("#2979FF"), 2.0)

    # Objects clustered to look like one sculpture
    scene.add_object_by_context(SDF_Material(Sphere(), mat_transparent), "LargeSphere", Transform(np.array([0.0, 0.0, 0.0])))
    
    t_cyl = Transform(np.array([0.0, 0.0, 0.0]))
    t_cyl.rotate(np.deg2rad(45), np.array([0, 0, 1]))
    t_cyl.rotate(np.deg2rad(45), np.array([1, 0, 0]))
    scene.add_object_by_context(SDF_Material(Cylinder(0.3), mat_mirror), "IntersectCylinder", t_cyl)
    
    # Cubes orbiting the center
    scene.add_object_by_context(SDF_Material(Cube(0.5), mat_emiss_red), "FloatCube1", Transform(np.array([-1.2, 0.8, 0.8]), rotation=np.array([np.deg2rad(30), np.deg2rad(30), 0])))
    scene.add_object_by_context(SDF_Material(Cube(0.5), mat_emiss_blue), "FloatCube2", Transform(np.array([1.2, -0.8, -0.8]), rotation=np.array([np.deg2rad(60), np.deg2rad(10), 0])))
    
    scene.add_object_by_context(SDF_Material(Pyramid(), mat_transparent), "TopPyramid", Transform(np.array([0.0, 1.5, 0.0]), scale=np.array([0.5, 0.5, 0.5])))

    # Lights
    l_key = Light(color=Color(1.0, 1.0, 1.0), intensity=500.0, radius=2.5)
    scene.add_object_by_context(l_key, "Key", Transform(np.array([3.0, 3.0, -3.0])))
    
    l_fill = Light(color=Color(0.5, 0.7, 1.0), intensity=120.0, radius=3.0)
    scene.add_object_by_context(l_fill, "Fill", Transform(np.array([-3.0, -1.0, 3.0])))

    cam.transform.look_at(np.array([0, 0, 0]))
    return scene

def get_industrial_shapes_scene(width: int = 150, height: int = 100) -> Scene:
    # Lower camera angle for grandeur
    cam_transform = Transform(np.array([3.0, 1.5, -5.0]), np.array([-0.1, 0.4, 0.0]))
    cam = Camera(cam_transform, fov=60.0, near=0.1, far=120.0, resolution_width=width, resolution_height=height)
    scene = Scene("industrial_shapes", cam, background_color=Color.from_hex("#2c2c2c"))

    # Materials
    mat_rusty = MaterialFactory.create_specular(Color.from_hex("#8B4513"), 0.4, 0.8, 0.7, 0.3)
    mat_steel = MaterialFactory.create_specular(Color.from_hex("#C0C0C0"), 0.1, 0.9, 0.9, 0.1)
    mat_brass = MaterialFactory.create_specular(Color.from_hex("#B87333"), 0.2, 0.7, 0.8, 0.4)
    mat_concrete = MaterialFactory.create_diffuse(Color.from_hex("#696969"), 0.9)
    mat_e_red = MaterialFactory.create_emissive(Color.from_hex("#FF3B3B"), 3.0)
    mat_e_blue = MaterialFactory.create_emissive(Color.from_hex("#3B7AFF"), 3.0)

    # Base Structure
    scene.add_object_by_context(SDF_Material(Cube(), mat_concrete), "ConcreteFloor", Transform(np.array([0.0, -2.1, 0.0]), scale=np.array([10, 1, 10])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_steel), "BaseStructure", Transform(np.array([0.0, -0.6, 0.0]), scale=np.array([2, 0.5, 1.5])))

    # Pipes (Rotated Cylinders) connecting to base
    t_pipe1 = Transform(np.array([-1.5, -0.6, 0.0]))
    t_pipe1.rotate(np.deg2rad(90), np.array([0, 0, 1]))
    scene.add_object_by_context(SDF_Material(Cylinder(0.4), mat_brass), "Pipe1", t_pipe1)
    
    t_pipe2 = Transform(np.array([1.5, -0.6, 0.5]))
    t_pipe2.rotate(np.deg2rad(90), np.array([1, 0, 0]))
    scene.add_object_by_context(SDF_Material(Cylinder(0.3), mat_rusty), "Pipe2", t_pipe2)

    # Gears
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_steel), "Gear1", Transform(np.array([-0.5, 1.0, 0.5]), rotation=np.array([np.deg2rad(90), 0, 0]), scale=np.array([0.8, 0.2, 0.8])))
    scene.add_object_by_context(SDF_Material(Cylinder(), mat_brass), "Gear2", Transform(np.array([0.5, 1.0, -0.2]), rotation=np.array([np.deg2rad(90), 0, 0]), scale=np.array([0.6, 0.2, 0.6])))

    # Beams
    t_beam1 = Transform(np.array([-2.5, 1.5, 2.0]))
    t_beam1.rotate(np.deg2rad(15), np.array([0, 1, 0]))
    t_beam1.rotate(np.deg2rad(15), np.array([1, 0, 0]))
    scene.add_object_by_context(SDF_Material(ShapeExtrusion(Square()), mat_rusty), "Beam1", t_beam1)

    # Panel with buttons
    scene.add_object_by_context(SDF_Material(Cube(), mat_steel), "PanelBase", Transform(np.array([0.0, 0.5, 2.0]), scale=np.array([1, 0.5, 0.2])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_e_red), "Button1", Transform(np.array([-0.5, 0.6, 2.2]), scale=np.array([0.1, 0.1, 0.1])))
    scene.add_object_by_context(SDF_Material(Cube(), mat_e_blue), "Button2", Transform(np.array([0.5, 0.6, 2.2]), scale=np.array([0.1, 0.1, 0.1])))

    # Lighting – overhead key with area radius + warm side fill for the panel face
    l_overhead = Light(color=Color(1.0, 1.0, 0.9), intensity=350.0, radius=3.0)
    scene.add_object_by_context(l_overhead, "Overhead", Transform(np.array([2.0, 5.0, 2.0])))

    l_side = Light(color=Color(1.0, 0.9, 0.7), intensity=150.0, radius=4.0)
    scene.add_object_by_context(l_side, "SideFill", Transform(np.array([-4.0, 2.0, -2.0])))
    
    cam.transform.look_at(np.array([0, 0.5, 0]))
    return scene

def get_forest_clearing_scene(width: int = 140, height: int = 100) -> Scene:
    """
    A nature-inspired scene using primitives to approximate trees and foliage.
    """
    # Elevated camera looking slightly down
    cam_transform = Transform(np.array([5.0, 3.0, 5.0]), np.array([-0.3, 0.8, 0.0]))
    cam = Camera(
        cam_transform, 
        fov=65.0, near=0.1, far=150.0, 
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    # Sky gradient (Blue to pale yellow)
    sky_colors = [Color.from_hex("#87CEEB"), Color.from_hex("#E0F7FA")]
    scene = Scene("forest_clearing", cam, background_color=ColorGradient(sky_colors, np.array([0.0, 1.0])))

    # Materials
    mat_trunk = MaterialFactory.create_diffuse(Color.from_hex("#5C4033"), roughness=0.9)
    mat_leaves_dark = MaterialFactory.create_diffuse(Color.from_hex("#228B22"), roughness=0.8)
    mat_leaves_light = MaterialFactory.create_diffuse(Color.from_hex("#32CD32"), roughness=0.8)
    mat_grass = MaterialFactory.create_diffuse(Color.from_hex("#4CAF50"), roughness=1.0)
    mat_stone = MaterialFactory.create_specular(Color.from_hex("#808080"), roughness=0.7, metallicness=0.2)

    # Ground (Huge Sphere)
    scene.add_object_by_context(
        SDF_Material(Sphere(50.0), mat_grass), 
        "GroundHill", 
        Transform(np.array([0.0, -50.0, 0.0]))
    )

    # Trees Loop (Simple Cylinder + Sphere clusters)
    # Positioning trees in a circle around the center
    for i in range(6):
        angle = (i / 6) * 2 * np.pi
        x = np.cos(angle) * 3.5
        z = np.sin(angle) * 3.5
        
        # Trunk
        scene.add_object_by_context(
            SDF_Material(Cylinder(), mat_trunk), 
            f"Trunk_{i}", 
            Transform(np.array([x, 1.0, z]), scale=np.array([0.3, 1.0, 0.3]))
        )
        
        # Foliage (Cluster of 2 spheres)
        foliage_mat = mat_leaves_light if i % 2 == 0 else mat_leaves_dark
        
        scene.add_object_by_context(
            SDF_Material(Sphere(), foliage_mat), 
            f"LeavesBottom_{i}", 
            Transform(np.array([x, 2.2, z]), scale=np.full(3, 0.9))
        )
        scene.add_object_by_context(
            SDF_Material(Sphere(), foliage_mat), 
            f"LeavesTop_{i}", 
            Transform(np.array([x, 3.0, z]), scale=np.full(3, 0.7))
        )

    # Central Feature (Rock)
    scene.add_object_by_context(
        SDF_Material(Sphere(), mat_stone),
        "CenterRock",
        Transform(np.array([0.0, 0.3, 0.0]), scale=np.array([0.8, 0.5, 0.6]))
    )

    # Lighting – sun + sky bounce to fill shadows under the canopy
    sun_light = Light(color=Color.from_hex("#FFFACD"), intensity=1000.0, radius=5.0)
    scene.add_object_by_context(sun_light, "Sun", Transform(np.array([10.0, 15.0, 5.0])))

    sky_fill = Light(color=Color.from_hex("#87CEEB"), intensity=200.0, radius=12.0)
    scene.add_object_by_context(sky_fill, "SkyFill", Transform(np.array([-8.0, 10.0, -8.0])))

    return scene


def get_checkerboard_infinity_scene(width: int = 120, height: int = 120) -> Scene:
    """
    A surreal scene focusing on perspective and repetitive patterns.
    """
    cam_transform = Transform(np.array([0.0, 2.0, -6.0]), np.array([-0.2, 0.0, 0.0]))
    cam = Camera(
        cam_transform, 
        fov=80.0, near=0.1, far=200.0, 
        resolution_width=width, resolution_height=height,
        camera_type=CameraType.PERSPECTIVE
    )
    
    scene = Scene("checkerboard_infinity", cam, background_color=Color.from_hex("#000000"))

    # Materials
    mat_black = MaterialFactory.create_specular(Color.from_hex("#111111"), roughness=0.1, metallicness=0.5)
    mat_white = MaterialFactory.create_specular(Color.from_hex("#EEEEEE"), roughness=0.1, metallicness=0.5)
    mat_reflect_sphere = MaterialFactory.create_specular(Color.from_hex("#FF00FF"), roughness=0.0, metallicness=1.0)

    # Grid Floor Generation
    grid_range_x = 6
    grid_range_z = 12
    tile_size = 1.0
    
    for x in range(-grid_range_x, grid_range_x + 1):
        for z in range(0, grid_range_z): # Extending deep into Z
            # Checkerboard logic
            mat = mat_white if (x + z) % 2 == 0 else mat_black
            
            # Using thin cubes as tiles
            pos = np.array([x * tile_size, 0.0, z * tile_size])
            scene.add_object_by_context(
                SDF_Material(Cube(), mat), 
                f"Tile_{x}_{z}", 
                Transform(pos, scale=np.array([0.48, 0.1, 0.48])) # Slight gap between tiles
            )

    # Floating Mirror Sphere
    scene.add_object_by_context(
        SDF_Material(Sphere(), mat_reflect_sphere),
        "HeroSphere",
        Transform(np.array([0.0, 1.5, 4.0]), scale=np.full(3, 1.2))
    )

    # Lighting
    # A low, dramatic light to cast long shadows on the tiles
    scene.add_object_by_context(
        Light(color=Color.from_hex("#FFFFFF"), intensity=800.0),
        "LowLight",
        Transform(np.array([5.0, 2.0, -2.0]))
    )
    
    # Overhead fill – neutral cool white
    scene.add_object_by_context(
        Light(color=Color.from_hex("#CCCCDD"), intensity=200.0),
        "Fill",
        Transform(np.array([0.0, 10.0, 5.0]))
    )

    return scene


def get_orbital_dock_scene(width: int = 140, height: int = 100) -> Scene:
    """
    Sci-fi space scene with high contrast lighting, black background, and emissive elements.
    """
    cam_transform = Transform(np.array([-4.0, 2.0, -4.0]), np.array([-0.2, -0.78, 0.0]))
    cam = Camera(
        cam_transform, fov=60.0, resolution_width=width, resolution_height=height
    )
    cam.transform.look_at(np.array([0.0, 0.0, 0.0]))

    scene = Scene("orbital_dock", cam, background_color=Color.from_hex("#000000"))

    # Materials
    mat_hull = MaterialFactory.create_specular(Color.from_hex("#CCCCCC"), roughness=0.3, metallicness=0.8)
    mat_solar = MaterialFactory.create_specular(Color.from_hex("#111133"), roughness=0.1, metallicness=0.9)
    mat_engine_glow = MaterialFactory.create_emissive(Color.from_hex("#00CCFF"), 4.0)
    mat_sensor_red = MaterialFactory.create_emissive(Color.from_hex("#FF0000"), 2.0)

    # Main Ship Body (Cylinder + Cone/Pyramid approximation)
    scene.add_object_by_context(
        SDF_Material(Cylinder(), mat_hull),
        "ShipCore",
        Transform(np.array([0.0, 0.0, 0.0]), np.array([0, 0, np.deg2rad(90)]), scale=np.array([0.5, 2.0, 0.5]))
    )

    # Solar Panels (Thin Cubes)
    panel_scale = np.array([0.1, 1.5, 0.8])
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_solar),
        "SolarLeft",
        Transform(np.array([0.0, 0.0, 1.2]), scale=panel_scale)
    )
    scene.add_object_by_context(
        SDF_Material(Cube(), mat_solar),
        "SolarRight",
        Transform(np.array([0.0, 0.0, -1.2]), scale=panel_scale)
    )

    # Engine (Rear)
    scene.add_object_by_context(
        SDF_Material(Sphere(), mat_engine_glow),
        "Engine",
        Transform(np.array([-2.1, 0.0, 0.0]), scale=np.full(3, 0.4))
    )

    # Docking Ring (Torus approximation using 4 Cylinders)
    ring_rad = 1.5
    for i in range(4):
        scene.add_object_by_context(
            SDF_Material(Cylinder(), mat_hull),
            f"RingSeg_{i}",
            Transform(np.array([1.0, np.cos(i*1.57)*ring_rad, np.sin(i*1.57)*ring_rad]), 
                      scale=np.array([0.2, 0.2, 0.2]))
        )

    # Navigation Light
    scene.add_object_by_context(
        SDF_Material(Sphere(), mat_sensor_red),
        "NavLight",
        Transform(np.array([2.0, 0.6, 0.0]), scale=np.full(3, 0.1))
    )

    # Light Source (Distant Star – small area radius for softer shadow edges)
    sun = Light(color=Color.from_hex("#FFFFFF"), intensity=1800.0, radius=1.5) 
    scene.add_object_by_context(sun, "Star", Transform(np.array([10.0, 5.0, 10.0])))

    # Blue bounce light from a nearby planet – raised so the dark hull side is readable
    planet_bounce = Light(color=Color.from_hex("#002244"), intensity=300.0, radius=10.0)
    scene.add_object_by_context(planet_bounce, "PlanetFill", Transform(np.array([-5.0, -10.0, -5.0])))

    return scene


def get_sdf_boolean_scene(width: int = 120, height: int = 120) -> Scene:
    """
    Demonstrates SDF Boolean operations: Union, Subtraction, and Intersection.
    """
    cam_transform = Transform(np.array([0.0, 2.5, -6.0]), np.zeros(3))
    cam = Camera(cam_transform, fov=60.0, resolution_width=width, resolution_height=height)
    scene = Scene("sdf_boolean_lab", cam, background_color=Color.from_hex("#121212"))

    # Materials
    mat_sub = MaterialFactory.create_specular(Color.from_hex("#FF4444"), roughness=0.1) # Red for subtraction
    mat_int = MaterialFactory.create_specular(Color.from_hex("#44FF44"), roughness=0.1) # Green for intersection
    mat_floor = MaterialFactory.create_diffuse(Color.from_hex("#333333"), 0.9)

    # 1. SUBTRACTION: Sphere minus a Cube
    # Result: A sphere with a square chunk missing.
    shape_a = Sphere(radius=1.0)
    shape_b = Cube(size=1.2)
    
    # logic: max(distA, -distB)
    subtraction_shape = ShapeSubtraction(shape_a, shape_b) 
    scene.add_object_by_context(SDF_Material(subtraction_shape, mat_sub), "SubtractedObj", Transform(np.array([-1.5, 1.0, 0.0])))

    # 2. INTERSECTION: Sphere and Cube overlap
    # Result: A rounded cube / "spherified" box.
    # logic: max(distA, distB)
    intersection_shape = ShapeIntersection(shape_a, shape_b)
    scene.add_object_by_context(SDF_Material(intersection_shape, mat_int), "IntersectedObj", Transform(np.array([1.5, 1.0, 0.0])))

    # Floor
    scene.add_object_by_context(SDF_Material(Cube(), mat_floor), "Floor", Transform(np.array([0.0, -0.5, 0.0]), scale=np.array([10, 0.1, 10])))
    
    # Lighting – key + fill so the carved interior surfaces on both objects are readable
    main_light = Light(color=Color(1, 1, 1), intensity=800.0, radius=2.5)
    scene.add_object_by_context(main_light, "MainLight", Transform(np.array([0.0, 5.0, -5.0])))

    fill_light = Light(color=Color(0.7, 0.7, 0.9), intensity=250.0, radius=4.0)
    scene.add_object_by_context(fill_light, "FillLight", Transform(np.array([0.0, 2.0, 3.0])))

    cam.transform.look_at(np.array([0, 1, 0]))
    return scene