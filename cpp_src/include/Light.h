#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <glad/glad.h>
#include <algorithm>
#include "colorClass.h"
#include "shaderClass.h"
#include "Debugger.h"

static GLuint g_LightsUBO = 0; // Global UBO for lights
static constexpr GLuint LIGHTS_BINDING_POINT = 2; // Binding point for the lights UBO
static constexpr int MAX_LIGHTS = 64;

struct GPULight {
    glm::vec4 posRadius;       // xyz=pos, w=radius
    glm::vec4 colorIntensity;  // rgb=color, w=intensity
    glm::vec4 dirType;         // xyz=dir, w=type
};

struct Light {
    glm::vec3 position;
    float     radius;
    glm::vec3 color;
    float     intensity = 1.0f;
    glm::vec3 direction;
    int       type; // 0=point,1=dir,2=spot

    // Constructor
    Light(const glm::vec3& pos, float rad, const glm::vec3& col, int t)
        : position(pos), radius(rad), color(col), type(t) {}

    Light(const glm::vec3& pos, float rad, const Color& col, int t)
        : position(pos), radius(rad), color(col.toVec3()), type(t) {}

    Light() : position(0.0f), radius(1.0f), color(1.0f), type(0) {}
};

void CreateLightsUBO();
void UpdateLightsUBO(const std::vector<Light>& lights);