#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>

#include "IVirtualObject.h"
#include "Debugger.h"

static GLuint g_MaterialUBO = 0;
static constexpr GLuint MATERIAL_UBO_BINDING_POINT = 1;

// C++ side material representation (not the UBO layout exactly, just convenience)
struct GPUMaterial {
    glm::vec4 albedoColor;    // rgb + padding
    glm::vec4 metaFlags;      // x = metallic, y = roughness, z = useAlbedoMap, w = useNormalMap (as floats)
    glm::vec4 emissive;       // rgb + padding
};

struct Material: public IVirtualObject {
    glm::vec3 albedo = glm::vec3(1.0f);
    float    padding1 = 0.0f;
    float    metallic = 0.0f;
    float    roughness = 1.0f;
    float    useAlbedoMap = 0.0f;   // treated as float flags for UBO
    float    useNormalMap = 0.0f;
    float    useMRMap = 0.0f;
    float    padding2 = 0.0f;
    glm::vec3 emissive = glm::vec3(0.0f);
    float     padding3 = 0.0f;

    json ToJSON() const override;
    void FromJSON(const json& j) override;
};