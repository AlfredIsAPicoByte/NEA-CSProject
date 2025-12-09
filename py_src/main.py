import re
import numpy as np
from PIL import Image
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.Raytracing import Raytracer
from src.Camera import VCamera, CameraType
from src.Scene import Scene
from src.Geometry import Sphere, VObject
from src.Luminance import LightSource, Color, Material
from src.PrimaryStructures import Transform
from src.Sampling import RandomSampler

def build_a_scene(width=200, height=200):
    # Camera: positioned at z = -3 looking toward +z (default Transform rotation => forward = +z)
    cam_transform = Transform(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.ones(3))
    cam = VCamera(cam_transform, fov=60.0, near=0.1, far=100.0, width=width, height=height, camType=CameraType.PERSPECTIVE)

    # Scene
    scene = Scene(name="SimpleScene", camera=cam)

    # Geometry: Sphere at origin
    sphere_shape = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5, name="Ball")
    # Material (optional) - Luminance.Material expects Color, roughness, etc.
    mat = Material(color=Color.use_hex("#F07A3B"), emissive=Color(0,0,0), roughness=0.5, glossiness=0.2, metallic=0.0)
    sphere_shape.material = mat
    vobj = VObject(shape=sphere_shape, name="SphereObject")
    scene.add_object(vobj)

    # Ground approximated with a large sphere (simple trick)
    ground = Sphere(center=np.array([0.0, -100.5, 0.0]), radius=100.0, name="Ground")
    ground.material = Material(color=Color.use_hex("#8FBF7F"), emissive=Color(0,0,0), roughness=1.0, glossiness=0.0, metallic=0.0)
    ground_obj = VObject(shape=ground, name="GroundObject")
    scene.add_object(ground_obj)

    # Light source
    light = LightSource(position=np.array([2.0, 3.0, -1.0]), color=Color.use_hex("#FFFFFF"), intensity=1.5, name="Sun")
    scene.add_light(light)

    return scene, cam

def render_and_save(scene, cam, out_path="render_out.png", spp=1, seed=0):
    tracer = Raytracer(rays_per_pixel=spp)
    sampler = RandomSampler(samples_per_pixel=spp, seed=seed)
    
    im = scene.render(tracer)

    im.save(out_path)
    print(f"Saved render to {out_path}")

if __name__ == "__main__":    # For a quick test, use small resolution; change to 800 for final
    width, height = 200, 200
    scene, cam = build_a_scene(width, height)
    render_and_save(scene, cam, out_path="render_out.png", spp=1, seed=42)
