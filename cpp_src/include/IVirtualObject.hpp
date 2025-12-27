#pragma once

#include <cstdlib>
#include <ctime>
#include <random>
#include <json.hpp>

using json = nlohmann::json;

class IVirtualObject {
public:
IVirtualObject() {
        // Use modern C++ random to avoid srand issues and collisions
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 4294967295); // 32-bit range
        l_id = dis(gen);
    };

    virtual ~IVirtualObject() = default;
    virtual void CleanUp() = 0;

    int GetLocalID() const {
        return l_id;
    };

    // Converts this objects data into json to be stored
    json ToJSON() const {
        json j;
        j["l_id"] = l_id;     // Serialize base ID
        SerializeFields(j);   // Delegate to child classes for specific data
        for (const auto& child : children) {
            j["children"].push_back(child->ToJSON());
        }
        return j;
    }

    // Retrives data form a json form storage
    void FromJSON(const json& j) {
        if (j.contains("l_id")) {
            l_id = j["l_id"]; // Deserialize base ID
        }
        DeserializeFields(j); // Delegate to child classes for specific data

        for (const auto& childJson : j.value("children", json::array())) {
            // Here we would need a factory method to create the correct IVirtualObject derived type
            // For simplicity, we'll assume a generic IVirtualObject can be created
            auto child = std::make_unique<IVirtualObject>();
            child->FromJSON(childJson);
            AddChild(std::move(child));
        }
    }

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
};