#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_set>

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

namespace {
    std::vector<std::string> LoadRequiredPackages() {
        std::vector<std::string> packages = {
            "numpy",
            "pybind11",
            "Pillow",
            "scipy",
        };
        
        // Load additional requirements from requirements.txt if it exists
        std::filesystem::path reqFile = std::filesystem::current_path() / "requirements.txt";
        if (std::filesystem::exists(reqFile)) {
            std::ifstream file(reqFile);
            std::string line;
            while (std::getline(file, line)) {
                // Remove whitespace and skip comments/empty lines
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (!line.empty() && line[0] != '#') {
                    packages.push_back(line);
                }
            }
        }
        return packages;
    }
}

const std::vector<std::string> requiredPythonPackages = LoadRequiredPackages();

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

    py::object GetNumPy() const { return numpyModule; }

    PythonManager(const PythonManager&) = delete;
    PythonManager& operator=(const PythonManager&) = delete;
    PythonManager(PythonManager&&) = delete;
    PythonManager& operator=(PythonManager&&) = delete;

private:
    py::object numpyModule;
    // Track packages we've attempted to install in this process to avoid repeated install attempts
    std::unordered_set<std::string> attemptedInstalls;

    // Indicates whether the Python interpreter was successfully initialized
    bool pythonInitialized = false;

    // If we decode and set PYTHONHOME we keep the wide string here so it can be freed on finalize
    wchar_t* pythonHome = nullptr;
};