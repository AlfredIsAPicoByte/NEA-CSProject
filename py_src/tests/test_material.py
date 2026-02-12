import numpy as np
import pytest
from dataclasses import dataclass, field

from src.Data.Color import Color
from src.Material.MaterialFactory import MaterialFactory

class TestMaterialFactory:
    def test_create_diffuse(self):
        """Diffuse material should scatter equally"""
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=1.0)
        assert mat is not None
        # Test BSDF properties

    def test_create_specular(self):
        """Specular material should reflect light"""
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0)
        assert mat.roughness == 0.0

    @pytest.mark.parametrize("roughness", [0.0, 0.5, 1.0])
    def test_material_roughness_variations(self, roughness):
        """Test material with different roughness values"""
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=roughness)
        assert abs(mat.roughness - roughness) < 1e-6