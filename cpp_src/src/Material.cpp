#include "Material.h"


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

// Call before drawing an object with its material
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