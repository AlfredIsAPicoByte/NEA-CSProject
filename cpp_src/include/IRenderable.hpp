#pragma once

#include <string>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <json.hpp>

#include "shaderClass.h"
#include "cameraClass.h"
#include "IVirtualObject.hpp"

class IRenderable: public IVirtualObject
{
public:
    virtual ~IRenderable() = default;
    virtual void Draw(Shader& shader, Camera& camera) = 0;

    void SetName(const std::string& name) { this->name = name; };
    const std::string& GetName() const { return name; };
    
    void SetModelMatrix(const glm::mat4& matrix) { modelMatrix = matrix; };
    glm::mat4 GetModelMatrix() const { return modelMatrix; };
protected:
    std::string name = "";

    virtual void SerializeFields(json& j) const override {
        j["name"] = name;

        // Convert glm::mat4 to a flat array of 16 floats
        const float* pSource = glm::value_ptr(modelMatrix);
        j["model_matrix"] = std::vector<float>(pSource, pSource + 16);
    }

    virtual void DeserializeFields(const json& j) override {
        if (j.contains("name")) {
            name = j["name"];
        }

        if (j.contains("model_matrix")) {
            // Read floats back into the matrix
            std::vector<float> matData = j["model_matrix"];
            if (matData.size() == 16) {
                modelMatrix = glm::make_mat4(matData.data());
            }
        }
    }

    glm::mat4 modelMatrix = glm::mat4(1.0f);
};