#pragma once

#include <memory>

#include "IRenderable.h"
#include "Model.h"
#include "Debugger.h"

class ModelMeshAdapter : public IRenderable {
public:
    ModelMeshAdapter(std::shared_ptr<Model> mdl, size_t meshIndex)
      : model(mdl), index(meshIndex) {}

    void Draw(Shader& shader, Camera& camera) override {
        model->Draw(shader, camera); // implement DrawMesh in Model
    }
    const std::string& GetName() const override { return model->GetMeshes()[index].GetName(); }
    glm::mat4 GetModelMatrix() const override { return model->GetModelMatrixForMesh(index); }

private:
    std::shared_ptr<Model> model;
    size_t index;
};