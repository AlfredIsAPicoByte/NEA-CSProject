#pragma once

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <pybind11/embed.h>

#include "Debug.h"

namespace py = pybind11;

const std::vector<std::string> requiredPythonPackages = {
    "numpy",
    "pybind11",
    // add more as needed
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
    void EnsureRequiredPackagesInstalled() {
        EnsurePythonPackagesInstalled(requiredPythonPackages);
    }
    void GetPackagesInstalledStatus(const std::vector<std::string>& packages);

    // match the implementation: use pybind11::object
    void AddModulePath(const std::string& path);
    py::object LoadModule(const std::string& moduleName);
    py::object CallFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args = {});

    PythonManager(const PythonManager&) = delete;
    PythonManager& operator=(const PythonManager&) = delete;
    PythonManager(PythonManager&&) = delete;
    PythonManager& operator=(PythonManager&&) = delete;
};
