#pragma once

#include <cstdlib>
#include <memory>
#include <ctime>
#include <random>
#include <json.hpp>
#include <glm/glm.hpp>

class Camera;
class Model;
class Mesh;
class ModelAdapter;
class Texture;
struct Vertex;
struct Light;
struct Material;

using json = nlohmann::json;

class IVirtualObject : public std::enable_shared_from_this<IVirtualObject> {
public:
    IVirtualObject() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 16777215);
        l_id = dis(gen);
    }

    virtual ~IVirtualObject() = default;
    virtual void CleanUp() = 0;

    int GetLocalID() const {
        return l_id;
    }

    json ToJSON() const {
        json j;
        j["l_id"] = l_id;
        SerializeFields(j);
        for (const auto& child : children) {
            j["children"].push_back(child->ToJSON());
        }
        return j;
    }

    void FromJSON(const json& j);  // Move to .cpp file

    void AddChild(std::unique_ptr<IVirtualObject> child) {
        child->parent = shared_from_this();
        children.push_back(std::move(child));
    }

    void RemoveChild(IVirtualObject* child) {
        children.erase(std::remove_if(children.begin(), children.end(),
            [child](const std::unique_ptr<IVirtualObject>& ptr) { return ptr.get() == child; }),
            children.end());
    }

    void SetParent(std::shared_ptr<IVirtualObject> newParent) {
        parent = newParent;
    }

    void ClearParent() {
        parent.reset();
    }

    void ClearChildren() {
        children.clear();
    }

    IVirtualObject(const IVirtualObject& v) : l_id(v.l_id) {}
    
    IVirtualObject& operator=(const IVirtualObject& v) {
        if (this != &v) {
            l_id = v.l_id;
        }
        return *this;
    }

    IVirtualObject(IVirtualObject&& v) noexcept : l_id(v.l_id) {
        v.l_id = 0; 
    }

    IVirtualObject& operator=(IVirtualObject&& v) noexcept {
        if (this != &v) {
            l_id = v.l_id;
            v.l_id = 0;
        }
        return *this;
    }

protected:
    virtual void SerializeFields(json& j) const = 0;
    virtual void DeserializeFields(const json& j) = 0;

private:
    unsigned int l_id;
    std::vector<std::unique_ptr<IVirtualObject>> children;
    std::weak_ptr<IVirtualObject> parent;

    static std::unique_ptr<IVirtualObject> CreateFromType(const std::string& type);

    template<typename T>
    static T SafeGet(const json& j, const std::string& key, const T& defaultValue) {
        if (j.contains(key)) {
            try {
                return j[key].get<T>();
            }
            catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }
    
    static glm::vec3 SafeGetVec3(const json& j, const std::string& key, const glm::vec3& defaultValue) {
        if (j.contains(key) && j[key].is_array() && j[key].size() >= 3) {
            return glm::vec3(j[key][0], j[key][1], j[key][2]);
        }
        return defaultValue;
    }
};