#pragma once
#include "IRenderable.h"
#include "Model.h"

class ModelMeshAdapter : public IRenderable {
public:
    ModelMeshAdapter(std::shared_ptr<Model> mdl, size_t meshIndex)
      : model(mdl), index(meshIndex) {}

    void Draw(Shader& shader, Camera& camera) override {
        model->Draw(shader, camera); // implement DrawMesh in Model
    }
    const std::string& GetName() const override { return model->GetMeshName(index); }
    glm::mat4 GetModelMatrix() const override { return model->GetMeshMatrix(index); }

private:
    std::shared_ptr<Model> model;
    size_t index;
};