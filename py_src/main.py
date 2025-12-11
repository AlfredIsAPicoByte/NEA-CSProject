import re
import numpy as np
from PIL import Image
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.Algorithims import Algorithm
from src.Raytracing import Raytracer
from src.Camera import VCamera, CameraType
from src.Scene import Scene
from src.Geometry import Sphere, Cube, VObject
from src.Luminance import LightSource, Color, Material
from src.PrimaryStructures import Transform
from src.Sampling import Sampler

def build_a_scene(width=200, height=200):
    # Camera: positioned at z = -3 looking toward +z (default Transform rotation => forward = +z)
    cam_transform = Transform(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)

    # Scene
    scene = Scene(name="SimpleScene", camera=cam)

    # Geometry: Sphere at origin
    sphere_shape = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5, name="Ball")
    # Material (optional) - Luminance.Material expects Color, roughness, etc.
    mat = Material(color=Color.from_hex("#4E70E0"), emissive=Color(.5,.5,1), roughness=0.5, glossiness=0.2, metallic=0.0)
    sphere_shape.material = mat
    vobj = VObject(shape=sphere_shape, name="SphereObject")
    scene.add_object(vobj)

    # Ground approximated with a large sphere (simple trick)
    ground = Sphere(center=np.array([0.0, 100.5, 0.0]), radius=100.0, name="Ground")
    mat = Material(color=Color.from_hex("#8FBF7F"), emissive=Color.from_hex("#8FBF7F"), roughness=1.0, glossiness=0.0, metallic=0.0)
    ground.material = mat
    ground_obj = VObject(shape=ground, name="GroundObject")
    scene.add_object(ground_obj)

    box = Cube(center=np.array([-1, 4, 5]), side_length=2, name="Box")
    box.Rotate(10, [0, 1, 0])
    mat = Material(color=Color.from_hex("#A38A5A"), emissive=Color(0,0,0), roughness=0, glossiness=0.0, metallic=0.5)
    box.material = mat
    box_obj = VObject(shape=box, name="BoxObject")
    scene.add_object(box_obj)

    # Light source
    light = LightSource(position=np.array([2.0, 3.0, -1.0]), color=Color.from_hex("#FFFFFF"), intensity=1.5, name="Sun")
    scene.add_light(light)

    return scene

def render_and_save(scene, algorithim: Algorithm, out_path="render_out_strat.png"):
    im = scene.render(algorithim)
    
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

if __name__ == "__main__":    # For a quick test, use small resolution; change to 800 for final
    width, height = 200, 200
    scene = build_a_scene(width, height)
    raytracer = Raytracer()
    render_and_save(scene, raytracer, out_path="render_out.png")
