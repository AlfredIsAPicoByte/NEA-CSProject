#include "IVirtualObject.hpp"

#include "VBO.h"
#include "Mesh.h"
#include "Model.h"
#include "Light.hpp"
#include "Material.hpp"
#include "cameraClass.h"
#include "textureClass.h"

void IVirtualObject::FromJSON(const json& j) {
    if (j.contains("l_id")) {
        l_id = j["l_id"];
    }
    DeserializeFields(j);

    for (const auto& childJson : j.value("children", json::array())) {
        if (!childJson.contains("type")) {
            continue;
        }
        
        std::string childType = childJson["type"];
        std::vector<std::string> childArgs;

        for(auto item: childJson["args"]) { 
            childArgs.push_back(std::string(item));
        }

        auto child = CreateFromType(childType, childArgs);
        
        if (child) {
            child->FromJSON(childJson);
            AddChild(std::move(child));
        }
    }
}

std::unique_ptr<IVirtualObject> IVirtualObject::CreateFromType(const std::string& type, const std::vector<std::string>& args) {
    if (type == "Camera") {
        return std::make_unique<Camera>(args.size() >= 3 ? std::stoi(args[0]) : 800,
                                        args.size() >= 3 ? std::stoi(args[1]) : 600,
                                        args.size() >= 3 ? glm::vec3(std::stof(args[2]), std::stof(args[3]), std::stof(args[4])) : glm::vec3(0.0f, 0.0f, 3.0f),
                                        args.size() >= 6 ? glm::vec3(std::stof(args[5]), std::stof(args[6]), std::stof(args[7])) : glm::vec3(0.0f, 0.0f, -1.0f),
                                        args.size() >= 9 ? glm::vec3(std::stof(args[8]), std::stof(args[9]), std::stof(args[10])) : glm::vec3(0.0f, 1.0f, 0.0f));
    }
    else if (type == "Light") {
        return std::make_unique<Light>(args.size() >= 4 ? glm::vec3(std::stof(args[0]), std::stof(args[1]), std::stof(args[2])) : glm::vec3(0.0f),
                                       args.size() >= 7 ? glm::vec3(std::stof(args[3]), std::stof(args[4]), std::stof(args[5])) : glm::vec3(1.0f),
                                       args.size() >= 8 ? std::stof(args[6]) : 1.0f,
                                       args.size() >= 9 ? std::stoi(args[7]) : 0);
    }
    else if (type == "Model") {
        return std::make_unique<Model>(args.size() >= 1 ? args[0].c_str() : "");
    }
    // else if (type == "Mesh") {
    //     return std::make_unique<Mesh>(args.size() >= 3 ? std::vector<Vertex>() : std::vector<Vertex>(),
    //                                   args.size() >= 3 ? std::vector<GLuint>(static_cast<GLuint>(std::stoi(args[15]))) : std::vector<GLuint>(glm::uint(0)),
    //                                   args.size() >= 3 ? std::vector<Texture>() : std::vector<Texture>());
    // }
    else if (type == "Material") {
        return std::make_unique<Material>(args.size() >= 7 ? glm::vec3(std::stof(args[0]), std::stof(args[1]), std::stof(args[2])) : glm::vec3(1.0f),
                                          args.size() >= 8 ? std::stof(args[3]) : 0.0f,
                                          args.size() >= 9 ? std::stof(args[4]) : 1.0f,
                                          args.size() >= 10 ? std::stof(args[5]) : 0.0f,
                                          args.size() >= 11 ? std::stof(args[6]) : 0.0f,
                                          args.size() >= 12 ? std::stof(args[7]) : 0.0f,
                                          args.size() >= 15 ? glm::vec3(std::stof(args[8]), std::stof(args[9]), std::stof(args[10])) : glm::vec3(0.0f));
    }
    
    return nullptr;
}