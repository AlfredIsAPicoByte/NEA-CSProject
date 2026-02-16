import pytest
import numpy as np
from src.Data.Color import Color
from src.Data.Transform import Transform
from src.Data.Ray import Ray
from src.Data.Camera import Camera, CameraType
from src.Geometry.SDF import Sphere, Cube
from src.Material.Factory import MaterialFactory


# Define pytest markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "fast: marks tests as fast (select with '-m \"fast\"')"
    )


# ---------------------------------------------------------------------------
# Transform fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_transform():
    """An identity Transform – no translation, rotation, or scale."""
    return Transform.Identity()


@pytest.fixture
def unit_translation():
    """Transform that moves +1 on every axis."""
    return Transform(np.ones(3))


@pytest.fixture
def ninety_deg_y_rotation():
    """Transform with a 90° rotation around the Y axis."""
    return Transform(np.zeros(3), rotation=np.array([0.0, np.deg2rad(90), 0.0]))


@pytest.fixture
def double_scale():
    """Transform that scales uniformly by 2."""
    return Transform(np.zeros(3), scale=np.full(3, 2.0))


# ---------------------------------------------------------------------------
# Color fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_color():
    return Color(1.0, 1.0, 1.0)


@pytest.fixture
def black_color():
    return Color(0.0, 0.0, 0.0)


@pytest.fixture
def red_color():
    return Color(1.0, 0.0, 0.0)


@pytest.fixture
def blue_color():
    return Color(0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Ray fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ray():
    """A ray originating at (0,0,−5) and pointing along +Z."""
    return Ray(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]))


@pytest.fixture
def sideways_ray():
    """A ray originating at (0,0,−5) and pointing along +X (misses centred geometry)."""
    return Ray(np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.0, 0.0]))


@pytest.fixture
def offset_ray():
    """A ray clearly above the origin, travelling along +Z (misses a unit sphere)."""
    return Ray(np.array([0.0, 5.0, -5.0]), np.array([0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# Geometry fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_sphere():
    return Sphere(1.0)


@pytest.fixture
def unit_cube():
    return Cube(1.0)


# ---------------------------------------------------------------------------
# Material fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def diffuse_white():
    return MaterialFactory.create_diffuse(Color(1.0, 1.0, 1.0), roughness=1.0)


@pytest.fixture
def mirror_material():
    return MaterialFactory.create_specular(Color(1.0, 1.0, 1.0), roughness=0.0, metallicness=1.0)


@pytest.fixture
def emissive_white():
    return MaterialFactory.create_emissive(Color(1.0, 1.0, 1.0), strength=1.0)


# ---------------------------------------------------------------------------
# Camera fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def default_camera():
    """A perspective camera at the origin looking along +Z, 16×16."""
    cam_transform = Transform(np.zeros(3))
    return Camera(
        cam_transform,
        fov=60.0, near=0.1, far=100.0,
        resolution_width=16, resolution_height=16,
        camera_type=CameraType.PERSPECTIVE
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')"
    )