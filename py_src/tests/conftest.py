import pytest
import numpy as np
from src.Data.Color import Color
from src.Data.Transform import Transform

@pytest.fixture
def identity_transform():
    """Provides an identity transform for tests"""
    return Transform.Identity()

@pytest.fixture
def white_color():
    """Provides white color for tests"""
    return Color(1.0, 1.0, 1.0)

@pytest.fixture
def sample_ray():
    """Provides a sample ray"""
    from src.Data.Ray import Ray
    return Ray(np.array([0, 0, -5]), np.array([0, 0, 1]))