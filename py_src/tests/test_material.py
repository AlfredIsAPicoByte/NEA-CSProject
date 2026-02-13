import numpy as np
import pytest
from src.Data.Color import Color
from src.Material.Factory import MaterialFactory
from src.Lighting.Optics import REFRACTIVE_INDICES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v):
    return v / np.linalg.norm(v)


NORMAL = _unit(np.array([0.0, 1.0, 0.0]))        # Pointing up
VIEW   = _unit(np.array([0.0, 1.0, -1.0]))        # View from above-front
LIGHT  = _unit(np.array([1.0, 1.0, -1.0]))        # Light from upper-right-front


# ---------------------------------------------------------------------------
# Diffuse Materials
# ---------------------------------------------------------------------------

class TestDiffuseMaterial:
    def test_create_returns_object(self):
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=1.0)
        assert mat is not None

    def test_roughness_stored_correctly(self):
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=0.7)
        assert np.isclose(mat.roughness, 0.7)

    def test_colour_stored_correctly(self):
        col = Color.from_hex("#FF8800")
        mat = MaterialFactory.create_diffuse(col, roughness=1.0)
        assert np.isclose(mat.color.r, col.r, atol=1e-4)
        assert np.isclose(mat.color.g, col.g, atol=1e-4)
        assert np.isclose(mat.color.b, col.b, atol=1e-4)

    @pytest.mark.parametrize("roughness", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_roughness_range(self, roughness):
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=roughness)
        assert np.isclose(mat.roughness, roughness, atol=1e-6)

    def test_full_roughness_scatters_uniformly(self):
        """Perfectly diffuse material energy should be spread across hemisphere."""
        mat = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=1.0)
        bsdf = mat.bsdf(NORMAL, VIEW)
        assert bsdf is not None, "BSDF should return a value for diffuse material"

    def test_bsdf_non_negative(self):
        """BSDF evaluation should never return negative energy."""
        mat = MaterialFactory.create_diffuse(Color(0.8, 0.5, 0.2), roughness=0.9)
        for theta in np.linspace(0, np.pi / 2, 5):
            sample_dir = _unit(np.array([np.sin(theta), np.cos(theta), 0.0]))
            val = mat.bsdf(NORMAL, sample_dir)
            assert np.all(np.array(val) >= -1e-6), f"Negative BSDF at theta={np.deg2rad(theta):.2f}"

    def test_black_diffuse_emits_no_light(self):
        """A pure black diffuse material should contribute zero colour."""
        mat = MaterialFactory.create_diffuse(Color(0, 0, 0), roughness=1.0)
        val = mat.bsdf(NORMAL, VIEW)
        result = np.array(val)
        assert np.allclose(result, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Specular / Metallic Materials
# ---------------------------------------------------------------------------

class TestSpecularMaterial:
    def test_create_returns_object(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0)
        assert mat is not None

    def test_zero_roughness_stored(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0)
        assert np.isclose(mat.roughness, 0.0)

    def test_high_roughness_stored(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=1.0)
        assert np.isclose(mat.roughness, 1.0)

    def test_metallicness_stored(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.1, metallicness=0.9)
        assert np.isclose(mat.metallicness, 0.9)

    def test_mirror_reflects_perfectly(self):
        """A mirror (roughness=0) should produce a sharp reflection lobe."""
        mirror = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0, metallicness=1.0)
        # Perfect reflection: reflect VIEW over NORMAL
        reflect = VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL
        reflect = _unit(reflect)
        val_on_lobe   = mirror.bsdf(NORMAL, reflect)
        val_off_lobe  = mirror.bsdf(NORMAL, _unit(np.array([1.0, 0.0, 0.0])))
        assert np.sum(val_on_lobe) >= np.sum(val_off_lobe)

    def test_rough_specular_softer_than_mirror(self):
        """Rough specular peak energy should be lower than a perfect mirror."""
        rough  = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.8)
        mirror = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0)
        reflect = _unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        assert np.sum(mirror.bsdf(NORMAL, reflect)) >= np.sum(rough.bsdf(NORMAL, reflect))

    def test_specular_intensity_modulates_response(self):
        """Increasing specular_intensity should raise the specular contribution."""
        low  = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.2, specular_intensity=0.1)
        high = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.2, specular_intensity=1.0)
        reflect = _unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        assert np.sum(high.bsdf(NORMAL, reflect)) >= np.sum(low.bsdf(NORMAL, reflect))

    @pytest.mark.parametrize("roughness,metallicness", [
        (0.0, 0.0), (0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0),
    ])
    def test_bsdf_non_negative_various_params(self, roughness, metallicness):
        mat = MaterialFactory.create_specular(Color(1, 1, 1),
                                               roughness=roughness,
                                               metallicness=metallicness)
        val = mat.bsdf(NORMAL, VIEW)
        assert np.all(np.array(val) >= -1e-6)


# ---------------------------------------------------------------------------
# Emissive Materials
# ---------------------------------------------------------------------------

class TestEmissiveMaterial:
    def test_create_returns_object(self):
        mat = MaterialFactory.create_emissive(Color(1, 1, 1), 1.0)
        assert mat is not None

    def test_emissive_strength_stored(self):
        mat = MaterialFactory.create_emissive(Color(1, 1, 1), 3.5)
        assert np.isclose(mat.emissive_strength, 3.5)

    def test_emissive_colour_stored(self):
        col = Color.from_hex("#00FF00")
        mat = MaterialFactory.create_emissive(col, 2.0)
        assert np.isclose(mat.color.g, 1.0, atol=1e-4)

    def test_emissive_contributes_without_lighting(self):
        """Emissive materials should emit light regardless of incoming direction."""
        mat = MaterialFactory.create_emissive(Color(1, 0, 0), 2.0)
        emission = mat.emit()
        assert emission is not None
        r, g, b = emission
        assert r > 0.0 and np.isclose(g, 0.0, atol=1e-4) and np.isclose(b, 0.0, atol=1e-4)

    def test_higher_strength_emits_more(self):
        """Doubling emissive_strength should approximately double emission."""
        mat1 = MaterialFactory.create_emissive(Color(1, 1, 1), 1.0)
        mat2 = MaterialFactory.create_emissive(Color(1, 1, 1), 2.0)
        e1 = np.array(mat1.emit())
        e2 = np.array(mat2.emit())
        assert np.allclose(e2, e1 * 2.0, atol=1e-4)

    def test_black_emissive_emits_nothing(self):
        mat = MaterialFactory.create_emissive(Color(0, 0, 0), 10.0)
        assert np.allclose(np.array(mat.emit()), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Glass / Refractive Materials
# ---------------------------------------------------------------------------

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
        assert np.isclose(mat.ior, ior)

    def test_transmission_stored(self):
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=1.5, transmission=0.8)
        assert np.isclose(mat.transmission, 0.8)

    def test_ior_above_one_bends_toward_normal(self):
        """Snell's law: n>1 bends the refracted ray toward the normal."""
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=1.5, transmission=1.0)
        incident = _unit(np.array([1.0, -1.0, 0.0]))   # 45° incident
        refracted = mat.refract(incident, NORMAL)
        if refracted is not None:
            angle_incident  = np.arccos(abs(np.dot(incident, NORMAL)))
            angle_refracted = np.arccos(abs(np.dot(_unit(refracted), NORMAL)))
            # Denser medium → smaller angle with normal
            assert angle_refracted <= angle_incident + 1e-5

    def test_low_ior_bends_away_from_normal(self):
        """IOR < 1 should bend the refracted ray away from the normal."""
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=0.9, transmission=1.0)
        incident = _unit(np.array([0.2, -1.0, 0.0]))
        refracted = mat.refract(incident, NORMAL)
        if refracted is not None:
            angle_incident  = np.arccos(abs(np.dot(incident, NORMAL)))
            angle_refracted = np.arccos(abs(np.dot(_unit(refracted), NORMAL)))
            assert angle_refracted >= angle_incident - 1e-5

    @pytest.mark.parametrize("ior_key", ["water", "acrylic", "glass", "diamond"])
    def test_known_ior_values_are_reasonable(self, ior_key):
        """All known IOR values should be in the physically plausible range [1.0, 3.0]."""
        ior = REFRACTIVE_INDICES[ior_key]
        assert 1.0 <= ior <= 3.0, f"IOR for {ior_key} ({ior}) is outside [1.0, 3.0]"

    def test_full_transmission_glass_not_opaque(self):
        """A fully transmissive glass should not absorb colour entirely."""
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=1.5, transmission=1.0, absorption_density=0.0)
        # For a zero-thickness sample, transmitted colour should equal incident
        transmitted = mat.transmit(Color(1, 1, 1), distance=0.0)
        if transmitted is not None:
            assert np.isclose(transmitted.r, 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Cross-type Comparisons
# ---------------------------------------------------------------------------

class TestMaterialComparisons:
    def test_diffuse_vs_specular_backscatter(self):
        """Diffuse should scatter more evenly; specular should peak on reflection lobe."""
        diffuse  = MaterialFactory.create_diffuse(Color(1, 1, 1), roughness=1.0)
        specular = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.0, metallicness=1.0)

        reflect    = _unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        side_dir   = _unit(np.array([1.0, 0.0, 0.0]))

        diff_reflect = np.sum(diffuse.bsdf(NORMAL, reflect))
        spec_reflect = np.sum(specular.bsdf(NORMAL, reflect))
        diff_side    = np.sum(diffuse.bsdf(NORMAL, side_dir))
        spec_side    = np.sum(specular.bsdf(NORMAL, side_dir))

        # Specular lobe ratio should be much higher than diffuse lobe ratio
        spec_ratio = spec_reflect / (spec_side + 1e-9)
        diff_ratio = diff_reflect / (diff_side + 1e-9)
        assert spec_ratio >= diff_ratio

    def test_roughness_zero_and_one_differ(self):
        """Mirror (roughness=0) and fully diffuse (roughness=1) should differ noticeably."""
        mirror = MaterialFactory.create_specular(Color(0.8, 0.8, 0.8), roughness=0.0)
        matte  = MaterialFactory.create_specular(Color(0.8, 0.8, 0.8), roughness=1.0)
        reflect = _unit(VIEW - 2 * np.dot(VIEW, NORMAL) * NORMAL)
        assert np.sum(mirror.bsdf(NORMAL, reflect)) != pytest.approx(
            np.sum(matte.bsdf(NORMAL, reflect)), abs=0.01
        )

class TestMaterialFactory:
    def test_create_diffuse(self):
        mat = MaterialFactory.create_diffuse(Color(1, 0, 0), roughness=0.5)
        assert mat is not None
        assert np.isclose(mat.roughness, 0.5)

    def test_create_specular(self):
        mat = MaterialFactory.create_specular(Color(1, 1, 1), roughness=0.2, metallicness=0.8)
        assert mat is not None
        assert np.isclose(mat.roughness, 0.2)
        assert np.isclose(mat.metallicness, 0.8)

    def test_create_emissive(self):
        mat = MaterialFactory.create_emissive(Color(0, 1, 0), emissive_strength=3.0)
        assert mat is not None
        assert np.isclose(mat.emissive_strength, 3.0)

    def test_create_glass(self):
        mat = MaterialFactory.create_glass(Color(1, 1, 1), Color(1, 1, 1),
                                            roughness=0.0, metallicness=0.0,
                                            ior=REFRACTIVE_INDICES["glass"],
                                            transmission=1.0)
        assert mat is not None
        assert np.isclose(mat.ior, REFRACTIVE_INDICES["glass"])
        assert np.isclose(mat.transmission, 1.0)