#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <json.hpp>

#include "IVirtualObject.hpp"
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

    Material(const glm::vec3& albedoColor, float metal, float rough, float useAlbedo,
             float useNormal, float useMR, const glm::vec3& emissiveColor)
        : albedo(albedoColor), metallic(metal), roughness(rough),
          useAlbedoMap(useAlbedo), useNormalMap(useNormal), useMRMap(useMR),
          emissive(emissiveColor) {}

    void CleanUp() override;

protected:
    void SerializeFields(json& j) const override;
    void DeserializeFields(const json& j) override;
};

void CreateMaterialUBO();
void UpdateMaterialUBO(const Material& mat);