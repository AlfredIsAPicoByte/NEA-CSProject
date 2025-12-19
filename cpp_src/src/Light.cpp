#include "Light.h"

json Light::ToJSON() const
{
    json j;
    j["position"] = { position.x, position.y, position.z };
    j["radius"] = radius;
    j["coneAngles"] = { coneAngles.x, coneAngles.y };
    j["color"] = { color.x, color.y, color.z };
    j["intensity"] = intensity;
    j["direction"] = { direction.x, direction.y, direction.z };
    j["type"] = type;
    j["constructor"] = "Light";
    return j;
}
void Light::FromJSON(const json& j)
{
    position = glm::vec3(j["position"][0], j["position"][1], j["position"][2]);
    radius = j["radius"];
    coneAngles = glm::vec2(j["coneAngles"][0], j["coneAngles"][1]);
    color = glm::vec3(j["color"][0], j["color"][1], j["color"][2]);
    intensity = j["intensity"];
    direction = glm::vec3(j["direction"][0], j["direction"][1], j["direction"][2]);
    type = j["type"];
}

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

void UpdateLightsUBO(const std::vector<Light>& lights)
{
    if (g_LightsUBO == 0) CreateLightsUBO();

    int count = static_cast<int>(std::min((size_t)MAX_LIGHTS, lights.size()));
    std::vector<GPULight> gpu(MAX_LIGHTS);

    for (int i = 0; i < count; ++i) {
        gpu[i].posType = glm::vec4(lights[i].position, static_cast<float>(lights[i].type));
        gpu[i].dirType = glm::vec4(lights[i].direction, static_cast<float>(lights[i].type));
        gpu[i].colorIntensity = glm::vec4(lights[i].color, lights[i].intensity);
        gpu[i].radius = glm::vec4(lights[i].radius, lights[i].coneAngles.x, lights[i].coneAngles.y, 0.0f); // z unused for now
    }

    glBindBuffer(GL_UNIFORM_BUFFER, g_LightsUBO);
    // write count at offset 0 (std140 padding will align to 16 bytes)
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(int), &count);

    GLsizeiptr vec4Size = sizeof(float) * 4;
    GLsizeiptr baseOffset = 16; // after padded u_lightCount

    glBufferSubData(GL_UNIFORM_BUFFER, baseOffset + vec4Size * MAX_LIGHTS * 0, vec4Size * MAX_LIGHTS, (void*)(&gpu[0].posType));
    glBufferSubData(GL_UNIFORM_BUFFER, baseOffset + vec4Size * MAX_LIGHTS * 1, vec4Size * MAX_LIGHTS, (void*)(&gpu[0].dirType));
    glBufferSubData(GL_UNIFORM_BUFFER, baseOffset + vec4Size * MAX_LIGHTS * 2, vec4Size * MAX_LIGHTS, (void*)(&gpu[0].colorIntensity));
    glBufferSubData(GL_UNIFORM_BUFFER, baseOffset + vec4Size * MAX_LIGHTS * 3, vec4Size * MAX_LIGHTS, (void*)(&gpu[0].radius));

    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}
// Nothing to free per-Light right now; satisfies the linker
void Light::CleanUp()
{
    // No per-instance GL resources allocated here
}