#pragma once

#include <cstdlib>
#include <ctime>
#include <json.hpp>

using json = nlohmann::json;

class IVirtualObject {
public:
    IVirtualObject() {
        srand(clock());
        l_id = rand() % 8193;
    };

    virtual ~IVirtualObject() = default;
    virtual void CleanUp() = 0;

    int GetLocalID() const {
        return l_id;
    };

    virtual json ToJSON() const = 0;
    virtual void FromJSON(const json& j) = 0;

    IVirtualObject(const IVirtualObject& v) { this = v; }
    IVirtualObject& operator=(const IVirtualObject& v) 
    IVirtualObject(IVirtualObject&& v) = v;
    IVirtualObject& operator=(IVirtualObject&& v) = v;
private:
    unsigned int l_id;
};