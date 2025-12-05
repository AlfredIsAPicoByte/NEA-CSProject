#pragma once

#include <string>
#include <memory>
#include <glm/glm.hpp>
#include "shaderClass.h"
#include "cameraClass.h"

class IRenderable {
public:
    virtual ~IRenderable() = default;
    virtual void Draw(Shader& shader, Camera& camera) = 0;
    virtual const std::string& GetName() const = 0;
    virtual glm::mat4 GetModelMatrix() const = 0;
};