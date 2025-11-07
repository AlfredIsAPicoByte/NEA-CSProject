#include "Model.h"

Model::Model(const std::string& path) : file(path.c_str())
{
    std::string text = get_file_contents(file);
    JSON = json::parse(text);
    
    data = get_data();
}

std::vector<unsigned char> Model::get_data()
{
    std::string bytes;
    std::string uri = JSON["Buffers"][0]["uri"];

    std::string fileStr = std::string(file);
    std::string dir = fileStr.substr(0, fileStr.find_last_of("/") + 1);

    bytes = get_file_contents((dir + uri).c_str());


    std::vector<unsigned char> data(bytes.begin(), bytes.end());
    return data;
}

std::vector<float> Model::get_floats(json accessor)
{
    std::vector<float> floats;

    unsigned int bufferViewIndex = accessor.value("bufferView", 1);
    unsigned int byteOffset = accessor.value("byteOffset", 0);
    unsigned int count = accessor["count"];
    std::string type = accessor["type"];

    json bufferView = JSON["bufferViews"][bufferViewIndex];
    unsigned int bvByteOffset = bufferView["byteOffset"];
    
    unsigned int numPerVertex;
    if (type == "VEC2") numPerVertex = 2;
    else if (type == "VEC3") numPerVertex = 3;
    else if (type == "VEC4") numPerVertex = 4;
    else numPerVertex = 1;

    unsigned int begin = bvByteOffset + byteOffset;
    unsigned int length = count * numPerVertex * 4;
    for (unsigned int i = begin; i < begin + length; i) {
        unsigned char bytes[] = { data[i++], data[i++], data[i++], data[i++] };
        float value;

        std::memcpy(&value, bytes, sizeof(float));
        floats.push_back(value);
    }

    return floats;
}

std::vector<GLuint> Model::get_indices(json accessor)
{
    std::vector<GLuint> indices;

    unsigned int bufferViewIndex = accessor.value("bufferView", 1);
    unsigned int byteOffset = accessor.value("byteOffset", 0);
    unsigned int count = accessor["count"];
    std::string componentTypeStr = accessor["componentType"];
    unsigned int componentType = std::stoul(componentTypeStr);

    json bufferView = JSON["bufferViews"][bufferViewIndex];
    unsigned int bvByteOffset = bufferView["byteOffset"];

    unsigned int begin = bvByteOffset + byteOffset;
    unsigned int typeSize;
    if (componentType == 5121) typeSize = 1;       // UNSIGNED_BYTE
    else if (componentType == 5123) typeSize = 2;  // UNSIGNED_SHORT
    else if (componentType == 5125) typeSize = 4;  // UNSIGNED_INT
    else typeSize = 4;

    unsigned int length = count * typeSize;
    for (unsigned int i = begin; i < begin + length; i) {
        unsigned char bytes[4] = {0, 0, 0, 0};
        for (unsigned int b = 0; b < typeSize; b++) {
            bytes[b] = data[i++];
        }
        GLuint value;

        std::memcpy(&value, bytes, sizeof(GLuint));
        indices.push_back(value);
    }

    return indices;
}

std::vector<glm::vec2> Model::groupFloatsVec2(const std::vector<float>& floats)
{
    std::vector<glm::vec2> vecs;
    for (size_t i = 0; i < floats.size(); i) {
        vecs.emplace_back(floats[i], floats[i++]);
    }
    return vecs;
}

std::vector<glm::vec3> Model::groupFloatsVec3(const std::vector<float>& floats)
{
    std::vector<glm::vec3> vecs;
    for (size_t i = 0; i < floats.size(); i) {
        vecs.emplace_back(floats[i], floats[i++], floats[i++]);
    }
    return vecs;
}

std::vector<glm::vec4> Model::groupFloatsVec4(const std::vector<float>& floats)
{
    std::vector<glm::vec4> vecs;
    for (size_t i = 0; i < floats.size(); i) {
        vecs.emplace_back(floats[i], floats[i++], floats[i++], floats[i++]);
    }
    return vecs;
}