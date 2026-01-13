import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from Scene import Scene
from Geometry import Primitive, Plane
from Luminance import LightSource, Color
from Camera import VCamera
from PrimaryStructures import Ray, Transform
from Shading import LambertShading, ShadowSettings, AmbienceSettings, BackgroundSettings
from Luminance import PBRMaterial, MaterialType


def test_self_shadow_ignored():
    scene = Scene()

    # Large plane centered at origin
    plane = Plane()
    plane_obj = Primitive(transform=Transform.identity(), shape=plane, material=None)

    # Make a light very close above the plane center
    light = LightSource(position=np.array([0.0, 5.0, 0.0]), color=Color(1,1,1), intensity=1.0, radius=0.0)
    scene.add_object(plane_obj)
    scene.add_light(light)

    # Hit point near the center of plane
    point = np.array([0.0, 0.0, 0.0]) + np.array([0.0, 1e-3, 0.0])  # slightly above the plane

    # Create shading and call visibility
    amb = AmbienceSettings(enabled=False)
    shadow = ShadowSettings(enabled=True, samples=1, bias=1e-4)
    bg = BackgroundSettings(enabled=False)
    shader = LambertShading(amb, shadow, bg)

    sampler = None

    # should return 1.0 since light is above with no occluders
    vis = shader._calculate_shadow_visibility(scene, point, light, light.get_light_direction(point), sampler, exclude_obj=plane_obj)

    assert vis == 1.0


if __name__ == '__main__':
    test_self_shadow_ignored()
    print('Shadow test passed')
