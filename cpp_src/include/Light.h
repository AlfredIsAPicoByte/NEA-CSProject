#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <glad/glad.h>
#include <algorithm>

#include "shaderClass.h"
#include "IVirtualObject.h"
#include "Debugger.h"

static GLuint g_LightsUBO = 0; // Global UBO for lights
static constexpr GLuint LIGHTS_BINDING_POINT = 2; // Binding point for the lights UBO
static constexpr int MAX_LIGHTS = 64;

struct GPULight {
    glm::vec4 posType;       // xyz=pos, w=type
    glm::vec4 dirType;         // xyz=dir, w=type
    glm::vec4 colorIntensity;  // rgb=color, w=intensity
    glm::vec4 radius;         // x=radius, y=innerRadius, z=outerRadius
};

struct Light: public IVirtualObject {
    glm::vec3 position;
    float     radius;
    glm::vec2 coneAngles;
    glm::vec3 color;
    float     intensity;
    glm::vec3 direction;
    int       type; // 0=point,1=dir,2=spot

    Light() : position(0.0f), coneAngles(.01f), color(1.0f), intensity(1.0f), type(0) {}
    Light(const glm::vec3& pos, const glm::vec3& col, float inten, int t = 0)
        : position(pos), radius(1.0f), coneAngles(.01f), color(col), intensity(inten), direction(0.0f, -1.0f, 0.0f), type(t) {}

    json ToJSON() const override;
    void FromJSON(const json& j) override;
    void CleanUp() override;
};

void CreateLightsUBO();
void UpdateLightsUBO(const std::vector<Light>& lights);