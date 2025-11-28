// Example functions: create UBO, update it and bind to binding point (call once at init and per-material update)
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "Material.h"

static GLuint g_MaterialUBO = 0;
static constexpr GLuint MATERIAL_UBO_BINDING_POINT = 1;

// std140-compatible block size: we store three vec4s (albedo+pad, meta flags, emissive+pad)
struct MaterialUBOData {
    glm::vec4 albedoColor;    // rgb + padding
    glm::vec4 metaFlags;      // x = metallic, y = roughness, z = useAlbedoMap, w = useNormalMap (as floats)
    glm::vec4 emissive;       // rgb + padding
};

void CreateMaterialUBO()
{
    if (g_MaterialUBO == 0) {
        glGenBuffers(1, &g_MaterialUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, g_MaterialUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(MaterialUBOData), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, MATERIAL_UBO_BINDING_POINT, g_MaterialUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }
}

// Call before drawing an object with its material
void UpdateMaterialUBO(const Material& mat)
{
    MaterialUBOData data;
    data.albedoColor = glm::vec4(mat.albedo, 0.0f);
    data.metaFlags = glm::vec4(mat.metallic, mat.roughness, mat.useAlbedoMap, mat.useNormalMap);
    data.emissive = glm::vec4(mat.emissive, 0.0f);

    glBindBuffer(GL_UNIFORM_BUFFER, g_MaterialUBO);
    // update entire buffer (fast enough for demo); for many materials consider persistent mapping or glBufferSubData ranges
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(MaterialUBOData), &data);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}