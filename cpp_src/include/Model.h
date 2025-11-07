# pragma once

#include <json.hpp>
#include "Mesh.h"
#include "Debug.h"

using json = nlohmann::json;

class Model {
public:
    Model(const std::string& path);
    void Draw(Shader& shader, Camera& camera);
private:
    const char* file;
    std::vector<unsigned char> data;
    json JSON;

    std::vector<unsigned char> get_data();
    std::vector<float> get_floats(json accessor);
    std::vector<GLuint> get_indices(json accessor);
    
    std::vector<Vertex> assembleVerticies(
        const std::vector<float>& positions,
        const std::vector<float>& normals,
        const std::vector<float>& texUVs
    );

    std::vector<glm::vec2> groupFloatsVec2(const std::vector<float>& floats);
    std::vector<glm::vec3> groupFloatsVec3(const std::vector<float>& floats);
    std::vector<glm::vec4> groupFloatsVec4(const std::vector<float>& floats);

};