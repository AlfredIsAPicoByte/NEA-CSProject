#pragma once
#include <glm/glm.hpp>

// C++ side material representation (not the UBO layout exactly, just convenience)
struct Material {
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
};