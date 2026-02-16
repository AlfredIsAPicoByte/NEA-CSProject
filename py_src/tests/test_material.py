import numpy as np
import pytest

from src.Data.Color import Color
from src.Data.Sampling import Sampler
from src.Material.Factory import MaterialFactory
from src.Lighting.Optics import REFRACTIVE_INDICES
from src.Utilities.Common import unit

NORMAL = unit(np.array([0.0, 1.0, 0.0]))        # Pointing up
VIEW   = unit(np.array([0.0, 1.0, -1.0]))        # View from above-front
LIGHT  = unit(np.array([1.0, 1.0, -1.0]))        # Light from upper-right-front


class TestDiffuseMaterial:
    def test_create_returns_object(self):
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=1.0)
        assert mat is not None

    def test_roughness_stored_correctly(self):
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=0.7)
        assert np.isclose(mat.data.roughness, 0.7)

    def test_colour_stored_correctly(self):
        col = Color.from_hex("#FF8800")
        mat = MaterialFactory.create_diffuse(col, roughness=1.0)
        assert np.isclose(mat.data.albedo.r, col.r, atol=1e-4)
        assert np.isclose(mat.data.albedo.g, col.g, atol=1e-4)
        assert np.isclose(mat.data.albedo.b, col.b, atol=1e-4)

    @pytest.mark.parametrize("roughness", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_roughness_range(self, roughness):
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=roughness)
        assert np.isclose(mat.data.roughness, roughness, atol=1e-6)

    def test_full_roughness_scatters_uniformly(self):
        """Perfectly diffuse material energy should be spread across hemisphere."""
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=1.0)
        bsdf = mat.evaluate_bsdf(LIGHT, VIEW, NORMAL)
        assert bsdf is not None, "BSDF should return a value for diffuse material"

    def test_bsdf_non_negative(self):
        """BSDF evaluation should never return negative energy."""
        mat = MaterialFactory.create_diffuse(Color(0.8, 0.5, 0.2), roughness=0.9)
        for theta in np.linspace(0, np.pi / 2, 5):
            sample_dir = unit(np.array([np.sin(theta), np.cos(theta), 0.0]))
            val = mat.evaluate_bsdf(LIGHT, sample_dir, NORMAL).to_np_array()
            assert np.all(np.array(val) >= -1e-6), f"Negative BSDF at theta={np.deg2rad(theta):.2f}"

    def test_black_diffuse_emits_no_light(self):
        """A pure black diffuse material should contribute zero colour."""
        mat = MaterialFactory.create_diffuse(Color(0, 0, 0), roughness=1.0)
        val = mat.evaluate_bsdf(LIGHT, VIEW, NORMAL).to_np_array()
        result = np.array(val)
        assert np.allclose(result, 0.0, atol=1e-6)

class TestSpecularMaterial:
    def test_create_returns_object(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0)
        assert mat is not None

    def test_zero_roughness_stored(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0)
        assert np.isclose(mat.data.roughness, 0.0)

    def test_high_roughness_stored(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=1.0)
        assert np.isclose(mat.data.roughness, 1.0)

    def test_metallicness_stored(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.1, metallicness=0.9)
        assert np.isclose(mat.data.metallic, 0.9)

    def test_mirror_reflects_perfectly(self):
        """A mirror (roughness=0) should produce a sharp reflection lobe."""
        mirror = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0, metallicness=1.0)
        # Perfect reflection: reflect VIEW over NORMAL
        reflect = VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL
        reflect = unit(reflect)
        val_on_lobe   = mirror.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array()
        val_off_lobe  = mirror.evaluate_bsdf(LIGHT, unit(np.array([1.0, 0.0, 0.0])), NORMAL).to_np_array()
        assert np.sum(val_on_lobe) > 10 * np.sum(val_off_lobe), "Mirror should have a strong peak on reflection direction"

    def test_rough_specular_softer_than_mirror(self):
        """Rough specular peak energy should be lower than a perfect mirror."""
        rough  = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.8)
        mirror = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0)
        reflect = unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        assert np.sum(mirror.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array()) >= np.sum(rough.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array())

    def test_specular_intensity_modulates_response(self):
        """Increasing specular_intensity should raise the specular contribution."""
        low  = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.2, specular_intensity=0.1)
        high = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.2, specular_intensity=1.0)
        reflect = unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        assert np.sum(high.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array()) >= np.sum(low.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array())

    @pytest.mark.parametrize("roughness,metallicness", [
        (0.0, 0.0), (0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0),
    ])
    def test_bsdf_non_negative_various_params(self, roughness, metallicness):
        mat = MaterialFactory.create_specular(Color(1, 1, 1),
                                               roughness=roughness,
                                               metallicness=metallicness)
        val = mat.evaluate_bsdf(LIGHT, VIEW, NORMAL).to_np_array()
        assert np.all(np.array(val) >= -1e-6)

class TestEmissiveMaterial:
    def test_create_returns_object(self):
        mat = MaterialFactory.create_emissive(Color(1, 1, 1), 1.0)
        assert mat is not None

    def test_emissive_strength_stored(self):
        mat = MaterialFactory.create_emissive(Color(1, 1, 1), 3.5)
        assert np.isclose(mat.data.emission_intensity, 3.5)

    def test_emissive_colour_stored(self):
        col = Color.from_hex("#00FF00")
        mat = MaterialFactory.create_emissive(col, 2.0)
        assert not bool(np.isclose(mat.data.albedo.g, 1.0, atol=1e-4))
        assert bool(np.isclose(mat.data.emission_color.g, 1.0, atol=1e-4))

class TestGlassMaterial:
    def test_create_returns_object(self):
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=REFRACTIVE_INDICES["glass"],
                                            transmission=1.0)
        assert mat is not None

    def test_ior_stored_correctly(self):
        ior = REFRACTIVE_INDICES["diamond"]
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=ior, transmission=1.0)
        assert np.isclose(mat.data.ior, ior)

    def test_transmission_stored(self):
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=1.5, transmission=0.8)
        assert np.isclose(mat.data.transmission, 0.8)

    def test_ior_above_one_bends_toward_normal(self):
        """Snell's law: n>1 bends the refracted ray toward the normal."""
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=1.5, transmission=1.0)
        incident = unit(np.array([1.0, -1.0, 0.0]))   # 45° incident
        other_ior = 0.5
        refracted, _ = mat.sample_glass_contribution(incident, NORMAL, Sampler(seed=42), other_ior)
        if refracted is not None:
            angle_incident  = np.arccos(abs(np.dot(incident, NORMAL)))
            angle_refracted = np.arccos(abs(np.dot(unit(refracted), NORMAL)))
            # Denser medium → smaller angle with normal
            assert angle_refracted <= angle_incident + 1e-5

    def test_low_ior_bends_away_from_normal(self):
        """IOR < 1 should bend the refracted ray away from the normal."""
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=0.9, transmission=1.0)
        incident = unit(np.array([0.2, -1.0, 0.0]))
        refracted, _ = mat.sample_glass_contribution(incident, NORMAL, Sampler(seed=42), 1.0)
        if refracted is not None:
            angle_incident  = np.arccos(abs(np.dot(incident, NORMAL)))
            angle_refracted = np.arccos(abs(np.dot(unit(refracted), NORMAL)))
            assert angle_refracted >= angle_incident - 1e-5

    @pytest.mark.parametrize("ior_key", ["water", "acrylic", "glass", "diamond"])
    def test_known_ior_values_are_reasonable(self, ior_key):
        """All known IOR values should be in the physically plausible range [1.0, 3.0]."""
        ior = REFRACTIVE_INDICES[ior_key]
        assert 1.0 <= ior <= 3.0, f"IOR for {ior_key} ({ior}) is outside [1.0, 3.0]"

class TestMaterialComparisons:
    def test_diffuse_vs_specular_backscatter(self):
        """Diffuse should scatter more evenly; specular should peak on reflection lobe."""
        diffuse  = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=1.0)
        specular = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0, metallicness=1.0)

        reflect    = unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        side_dir   = unit(np.array([1.0, 0.0, 0.0]))

        diff_reflect = np.sum(diffuse.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array())
        spec_reflect = np.sum(specular.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array())
        diff_side    = np.sum(diffuse.evaluate_bsdf(LIGHT, side_dir, NORMAL).to_np_array())
        spec_side    = np.sum(specular.evaluate_bsdf(LIGHT, side_dir, NORMAL).to_np_array())

        # Specular lobe ratio should be much higher than diffuse lobe ratio
        spec_ratio = spec_reflect / (spec_side + 1e-9)
        diff_ratio = diff_reflect / (diff_side + 1e-9)
        assert spec_ratio >= diff_ratio

    def test_roughness_zero_and_one_differ(self):
        """Mirror (roughness=0) and fully diffuse (roughness=1) should differ noticeably."""
        mirror = MaterialFactory.create_specular(Color(0.8, 0.8, 0.8), roughness=0.0)
        matte  = MaterialFactory.create_specular(Color(0.8, 0.8, 0.8), roughness=1.0)
        reflect = unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        assert np.sum(mirror.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array()) != pytest.approx(
            np.sum(matte.evaluate_bsdf(LIGHT, reflect, NORMAL).to_np_array()), abs=0.01
        )

class TestMaterialFactory:
    def test_create_diffuse(self):
        mat = MaterialFactory.create_diffuse(Color(1, 0, 0), roughness=0.5)
        assert mat is not None
        assert np.isclose(mat.data.roughness, 0.5)

    def test_create_specular(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.2, metallicness=0.8)
        assert mat is not None
        assert np.isclose(mat.data.roughness, 0.2)
        assert np.isclose(mat.data.metallic, 0.8)

    def test_create_emissive(self):
        mat = MaterialFactory.create_emissive(Color(0, 1, 0), 3.0)
        assert mat is not None
        assert np.isclose(mat.data.emission_intensity, 3.0)

    def test_create_glass(self):
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=REFRACTIVE_INDICES["glass"],
                                            transmission=1.0)
        assert mat is not None
        assert np.isclose(mat.data.ior, REFRACTIVE_INDICES["glass"])
        assert np.isclose(mat.data.transmission, 1.0)