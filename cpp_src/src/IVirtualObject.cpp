#include "IVirtualObject.hpp"

#include "VBO.h"
#include "Mesh.h"
#include "Model.h"
#include "ModelAdapter.h"
#include "Light.hpp"
#include "Material.hpp"
#include "cameraClass.h"
#include "textureClass.h"

void IVirtualObject::FromJSON(const json& j) {
    try {
        if (j.contains("l_id")) {
            l_id = j["l_id"];
        }
        DeserializeFields(j);

        for (const auto& childJson : j.value("children", json::array())) {
            if (!childJson.contains("type")) {
                continue;
            }
            
            std::string childType = childJson["type"];
            auto child = CreateFromType(childType);
            
            if (child) {
                child->FromJSON(childJson);
                AddChild(std::move(child));
            }
        }
    }
    catch (const json::exception& e) {
        AppendError("JSON error in FromJSON: " + std::string(e.what()) + ", JSON content: " + j.dump(2));
        throw; // Re-throw to see the stack trace
    }
}

std::unique_ptr<IVirtualObject> IVirtualObject::CreateFromType(const std::string& type) {
    
    if (type == "Mesh") {
        return std::make_unique<Mesh>(std::vector<Vertex> {}, std::vector<GLuint> {}, std::vector<Texture> {});
    }
    else if (type == "Model") {
        return std::make_unique<Model>("");
    }
    else if (type == "ModelAdapter") {
        return std::make_unique<ModelAdapter>(std::make_unique<Model>(""), size_t());
    }
    else if (type == "Camera") {
        return std::make_unique<Camera>(800, 600, glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    }
    else if (type == "Light") {
        return std::make_unique<Light>(glm::vec3(0.0f), glm::vec3(1.0f), 1.0f, 0);
    }
    else if (type == "Material") {
        return std::make_unique<Material>(glm::vec3(1.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, glm::vec3(0.0f));
    }
    
    return nullptr;
}