#pragma once

#include <json.hpp>

using json = nlohmann::json;

class IVirtualObject {
public:
    virtual ~IVirtualObject() = default;
    virtual void CleanUp() = 0;

    void SetID(int newID) {
        id = newID;
    };
    int GetID() const {
        return id;
    };

    virtual json ToJSON() const = 0;
    virtual void FromJSON(const json& j) = 0;
private:
    unsigned int id;
};