#pragma once

#include <string>
#include <memory>
#include <glm/glm.hpp>

#include "shaderClass.h"
#include "cameraClass.h"
#include "IVirtualObject.h"

class IRenderable: public IVirtualObject
{
public:
    virtual ~IRenderable() = default;
    virtual void Draw(Shader& shader, Camera& camera) = 0;

    void SetName(const std::string& name) 
    {
        this->name = name;
    };
    const std::string& GetName() const
    {
        return name;
    };
    void SetModelMatrix(const glm::mat4& matrix)
    {
        modelMatrix = matrix;
    };
    glm::mat4 GetModelMatrix() const
    {
        return modelMatrix;
    };
protected:
    std::string name = "";

    glm::mat4 modelMatrix = glm::mat4(1.0f);
};