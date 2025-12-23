#include "Material.h"

void Material::SerializeFields(json& j) const
{
    j["albedo"] = { albedo.x, albedo.y, albedo.z };
    j["metallic"] = metallic;
    j["roughness"] = roughness;
    j["useAlbedoMap"] = useAlbedoMap;
    j["useNormalMap"] = useNormalMap;
    j["useMRMap"] = useMRMap;
    j["emissive"] = { emissive.x, emissive.y, emissive.z };
    j["constructor"] = "Material";
}

void Material::DeserializeFields(const json& j)
{
    if (j.contains("albedo"))
        albedo = glm::vec3(j["albedo"][0], j["albedo"][1], j["albedo"][2]);
    if (j.contains("metallic"))
        metallic = j["metallic"];
    if (j.contains("roughness"))
        roughness = j["roughness"];
    if (j.contains("useAlbedoMap"))
        useAlbedoMap = j["useAlbedoMap"];
    if (j.contains("useNormalMap"))
        useNormalMap = j["useNormalMap"];
    if (j.contains("useMRMap"))
        useMRMap = j["useMRMap"];
    if (j.contains("emissive"))
        emissive = glm::vec3(j["emissive"][0], j["emissive"][1], j["emissive"][2]);
}

void CreateMaterialUBO()
{
    if (g_MaterialUBO == 0) {
        glGenBuffers(1, &g_MaterialUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, g_MaterialUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(GPUMaterial), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, MATERIAL_UBO_BINDING_POINT, g_MaterialUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
}

void UpdateMaterialUBO(const Material& mat)
{
    GPUMaterial data;
    data.albedoColor = glm::vec4(mat.albedo, 0.0f);
    data.metaFlags = glm::vec4(mat.metallic, mat.roughness, mat.useAlbedoMap, mat.useNormalMap);
    data.emissive = glm::vec4(mat.emissive, 0.0f);

    glBindBuffer(GL_UNIFORM_BUFFER, g_MaterialUBO);
    // update entire buffer (fast enough for demo); for many materials consider persistent mapping or glBufferSubData ranges
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(GPUMaterial), &data);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

// Nothing to free per-Material right now; satisfies the linker
void Material::CleanUp()
{
    // No per-instance GL resources allocated here
}