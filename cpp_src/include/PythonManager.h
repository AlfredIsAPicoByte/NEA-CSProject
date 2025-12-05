#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "Debugger.h"

#ifdef USE_EMBEDDED_PYTHON
#include <pybind11/embed.h>

namespace py = pybind11;

#else
// Stub mode: no pybind11 / Python available. Provide minimal placeholders
// so files that reference py::object or the PythonManager API still compile.
namespace py {
    struct object {
        object() {}
        object attr(const std::string&) const { return object(); }
        bool contains(const std::string&) const { return false; }
        template<typename... Args>
        object operator()(Args&&...) const { return object(); }
    };
    inline object none() { return object(); }
}
#endif



const std::vector<std::string> requiredPythonPackages = {
    "numpy",
    "pybind11",
};

class PythonManager {
public:
    static PythonManager& Instance() {
        static PythonManager instance;
        return instance;
    }

    PythonManager();
    ~PythonManager();

    void Initialize();
    void Finalize();

    void InstallPackage(const std::string& packageName);
    void ImportModule(const std::string& moduleName);
    void ValidatePackageInstallation(const std::string& packageName);
    void EnsurePythonPackagesInstalled(const std::vector<std::string>& packages);
    void EnsureRequiredPackagesInstalled() { EnsurePythonPackagesInstalled(requiredPythonPackages); }
    void GetPackagesInstalledStatus(const std::vector<std::string>& packages);

    void AddModulePath(const std::string& path);
    py::object LoadModule(const std::string& moduleName);
    py::object CallFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args = {});

    PythonManager(const PythonManager&) = delete;
    PythonManager& operator=(const PythonManager&) = delete;
    PythonManager(PythonManager&&) = delete;
    PythonManager& operator=(PythonManager&&) = delete;
};