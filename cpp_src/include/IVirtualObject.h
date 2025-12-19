#pragma once

#include <json.hpp>

using json = nlohmann::json;

class IVirtualObject {
public:
    virtual ~IVirtualObject() = default;
    virtual void CleanUp() = 0;

    void SetLocaID(int newID) {
        l_id = newID;
    };
    int GetLocalID() const {
        return l_id;
    };

    virtual json ToJSON() const = 0;
    virtual void FromJSON(const json& j) = 0;
private:
    unsigned int l_id;
};