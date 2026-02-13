from src.Data.Color import Color
from .Core import MaterialType, MaterialData, PBRMaterial

class MaterialFactory:
    @classmethod
    def create_diffuse(cls, albedo: Color, roughness: float = 0.5):
        """
        Creates a standard non-metallic (dielectric) material like plastic, wood, or chalk.
        """
        data = MaterialData(
            name="DiffuseMat",
            type=MaterialType.DIFFUSE,
            albedo=albedo,
            roughness=roughness,
            metallic=0.0,             # Non-metal
            specular_intensity=0.0,   # Most dielectrics have approx 4% reflectance
            transmission=0.0
        )
        return PBRMaterial(data)

    @classmethod
    def create_specular(cls, albedo: Color, roughness: float = 0.2, metallicness: float = 1.0, specular_intensity: float = 1.0, specular_tint_amount: float = 0.5):
        """
        Creates a reflective material like gold, aluminum, or copper.
        """
        data = MaterialData(
            name="MetalMat",
            type=MaterialType.SPECULAR,
            albedo=albedo,
            roughness=roughness,
            metallic=metallicness,
            specular_intensity=specular_intensity,
            specular_tint=specular_tint_amount,
            transmission=0.0
        )
        return PBRMaterial(data)

    @classmethod
    def create_glass(cls, albedo: Color, absorption_color: Color, roughness: float = 0.0, metallicness: float = 0.0, ior: float = 1.5, transmission: float = 1.0, absorption_density: float = 1.0):
        """
        Creates a dielectric transparent material (Refractive).
        """
        data = MaterialData(
            name="GlassMat",
            type=MaterialType.GLASS,
            
            # Surface Properties
            albedo=albedo,            # Surface tint (usually White for clear glass)
            roughness=roughness,      # 0.0 = Clear, 0.5 = Frosted
            metallic=metallicness,    # Usually 0.0
            
            # Volumetric/Transmission Properties
            ior=ior,
            transmission=transmission,         # Enables Refraction logic
            absorption_color=absorption_color, # The color inside the glass (Beer's Law)
            absorption_density=absorption_density
        )
        return PBRMaterial(data)

    @classmethod
    def create_transparent(cls, albedo: Color):
        """
        Creates a "See-Through" material using Alpha Blending (Ghosts, Holograms, Decals).
        Different from Glass because it does not refract light.
        """
        data = MaterialData(
            name="TransparentMat",
            type=MaterialType.TRANSPARENT,
            albedo=albedo,            # albedo.a (Alpha) controls opacity
            roughness=0.8,            # Usually fairly rough to avoid sharp specular highlights on a ghost
            metallic=0.0,
            transmission=0.0          # 0 because we use Alpha Blending, not Refraction
        )
        return PBRMaterial(data)

    @classmethod
    def create_emissive(cls, color: Color, intensity: float = 1.0):
        """
        Creates a glowing material (Light Bulb, Neon Sign).
        """
        data = MaterialData(
            name="EmissiveMat",
            type=MaterialType.EMISSIVE,
            albedo=Color(0.0, 0.0, 0.0),  # Emissive materials don't reflect light, so albedo is usually black
            roughness=1.0,                # Doesn't matter much for emissive, but we can set it to max roughness to avoid any specular highlights
            metallic=0.0,
            transmission=0.0,
            emission_color=color,
            emission_intensity=intensity,
        )
        return PBRMaterial(data)