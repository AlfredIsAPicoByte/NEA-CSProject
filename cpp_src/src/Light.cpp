#include "Light.h"

/// @brief Create the lights UBO
void CreateLightsUBO()
{
    if (g_LightsUBO != 0) return;
    glGenBuffers(1, &g_LightsUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, g_LightsUBO);

    // std140: u_lightCount (4 bytes) padded to 16 + 3 * vec4 * MAX_LIGHTS
    GLsizeiptr size = 16 + sizeof(float) * 4 * 3 * MAX_LIGHTS;
    glBufferData(GL_UNIFORM_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, LIGHTS_BINDING_POINT, g_LightsUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

/// @brief Update the lights UBO with the provided lights
/// @param lights Vector of Light structures
void UpdateLightsUBO(const std::vector<Light>& lights)
{
    if (g_LightsUBO == 0) CreateLightsUBO();

    int count = static_cast<int>(std::min((size_t)MAX_LIGHTS, lights.size()));
    std::vector<GPULight> gpu(MAX_LIGHTS);

    for (int i = 0; i < count; ++i) {
        gpu[i].posRadius = glm::vec4(lights[i].position, lights[i].radius);
        gpu[i].colorIntensity = glm::vec4(lights[i].color, lights[i].intensity);
        gpu[i].dirType = glm::vec4(lights[i].direction, static_cast<float>(lights[i].type));
    }

    glBindBuffer(GL_UNIFORM_BUFFER, g_LightsUBO);
    // write count at offset 0 (std140 padding will align to 16 bytes)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(int), &count);

    GLsizeiptr vec4Size = sizeof(float) * 4;
    GLsizeiptr baseOffset = 16; // after padded u_lightCount

    glBufferSubData(GL_UNIFORM_BUFFER, baseOffset + vec4Size * MAX_LIGHTS * 0, vec4Size * MAX_LIGHTS, (void*)(&gpu[0].posRadius));
    glBufferSubData(GL_UNIFORM_BUFFER, baseOffset + vec4Size * MAX_LIGHTS * 1, vec4Size * MAX_LIGHTS, (void*)(&gpu[0].colorIntensity));
    glBufferSubData(GL_UNIFORM_BUFFER, baseOffset + vec4Size * MAX_LIGHTS * 2, vec4Size * MAX_LIGHTS, (void*)(&gpu[0].dirType));

    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}