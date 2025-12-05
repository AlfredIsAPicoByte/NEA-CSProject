// The implementation below requires pybind11 and the Python dev headers/lib.
// Compile it only when embedding is enabled.
#ifdef USE_EMBEDDED_PYTHON

#include "pythonManager.h"

namespace fs = std::filesystem;

PythonManager::PythonManager() { Initialize(); }

void PythonManager::Initialize() {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
        try {
            fs::path scriptPath = fs::current_path() / "../py_src/src";
            py::module_ sys = py::module_::import("sys");
            sys.attr("path").attr("append")(scriptPath.string());
            AppendPythonMessage(std::string("Added Python path: ") + scriptPath.string());
        } catch (const std::exception& e) {
            AppendPythonError(std::string("Failed to add Python path: ") + e.what());
        }
    }
}

PythonManager::~PythonManager() { Finalize(); }

void PythonManager::Finalize() {
    if (Py_IsInitialized()) py::finalize_interpreter();
}

void PythonManager::InstallPackage(const std::string& packageName) {
    py::module_ sys = py::module_::import("sys");
    py::module_ subprocess = py::module_::import("subprocess");
    try {
        subprocess.attr("check_call")(py::make_tuple(sys.attr("executable"), "-m", "pip", "install", packageName));
        AppendPythonMessage("Successfully installed Python package: " + packageName);
    } catch (const py::error_already_set& error) {
        AppendPythonError("Failed to install Python package '" + packageName + "': " + error.what());
    } catch (const std::exception& e) {
        AppendPythonError("Exception during installation of package '" + packageName + "': " + e.what());
    }
}

void PythonManager::ImportModule(const std::string& moduleName) {
    py::module_::import(moduleName.c_str());
    AppendPythonMessage("Successfully imported Python module: " + moduleName);
}

void PythonManager::ValidatePackageInstallation(const std::string& packageName) {
    try {
        py::module_::import(packageName.c_str());
        AppendPythonMessage("Python package '" + packageName + "' is already installed.");
    } catch (const py::error_already_set&) {
        AppendPythonMessage("Python package '" + packageName + "' not found. Installing...");
        InstallPackage(packageName);
    }
}

void PythonManager::EnsurePythonPackagesInstalled(const std::vector<std::string>& packages) {
    for (const auto& package : packages) ValidatePackageInstallation(package);
}

void PythonManager::GetPackagesInstalledStatus(const std::vector<std::string>& packages) {
    for (const auto& package : packages) {
        try {
            py::module_::import(package.c_str());
            AppendPythonMessage("Python package '" + package + "' is installed.");
        } catch (const py::error_already_set&) {
            AppendPythonMessage("Python package '" + package + "' is NOT installed.");
        }
    }
}

void PythonManager::AddModulePath(const std::string& path) {
    try {
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(path);
        AppendPythonMessage(std::string("Added Python path: ") + path);
    } catch (const std::exception& e) {
        AppendPythonError(std::string("Failed to add Python path: ") + e.what());
    }
}

py::object PythonManager::LoadModule(const std::string& moduleName) {
    try {
        return py::module_::import(moduleName.c_str());
    } catch (const py::error_already_set& error) {
        AppendPythonError(std::string("Error loading module '" + moduleName + "': " + error.what()));
        return py::none();
    }
}

py::object PythonManager::CallFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args) {
    try {
        if (!module.contains(funcName.c_str())) {
            AppendPythonError(std::string("Function '" + funcName + "' not found in module"));
            return py::none();
        }
        py::object func = module.attr(funcName.c_str());
        if (args.empty()) return func();
        return func(args);
    } catch (const py::error_already_set& error) {
        AppendPythonError(std::string("Error calling function '" + funcName + "': " + error.what()));
        return py::none();
    }
}

#else
// When Python embedding is disabled the header provides stubbed inline
// implementations. Nothing to compile here.
#endif