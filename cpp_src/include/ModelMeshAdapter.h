#pragma once

#include <memory>
#include <stdexcept>

#include "Model.h"
#include "IRenderable.h"

class ModelMeshAdapter : public IRenderable {
public:
    // Accept a shared_ptr<Model> and the mesh index this adapter represents
    ModelMeshAdapter(std::shared_ptr<Model> mdl, size_t meshIndex)
      : model(mdl), index(meshIndex) 
    {
        if (!model) throw std::invalid_argument("ModelMeshAdapter: null model");
    }

    // Draw only the specific mesh this adapter represents
    void Draw(Shader& shader, Camera& camera) override {
        auto &meshes = model->GetMeshes();
        if (index < meshes.size()) {
            // Ensure mesh uses the model matrix for this mesh
            meshes[index].SetModelMatrix(model->GetModelMatrixForMesh((unsigned int)index));
            meshes[index].Draw(shader, camera);
        }
    }

    // Forward the cleanup to the specific mesh
    void CleanUp() override {
        auto &meshes = model->GetMeshes();
        if (index < meshes.size()) {
            meshes[index].CleanUp();
        }
    }

    const std::shared_ptr<Model>& GetModel() const { return model; }
    size_t GetMeshIndex() const { return index; }

    void SetName(const std::string& name) { model->GetMeshes()[index].SetName(name); }
    const std::string& GetName() const { return model->GetMeshes()[index].GetName(); }
    void SetModelMatrix(const glm::mat4& matrix) { model->SetModelMatrixForMesh((unsigned int)index, matrix); }
    glm::mat4 GetModelMatrix() const { return model->GetModelMatrixForMesh((unsigned int)index); }

	json ToJSON() const override { return model->GetMeshes()[index].ToJSON(); }
	void FromJSON(const json& j) override { model->GetMeshes()[index].FromJSON(j); }
private:
    std::shared_ptr<Model> model;
    size_t index;
};