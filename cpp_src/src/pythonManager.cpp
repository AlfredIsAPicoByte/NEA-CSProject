#include "pythonManager.h"

namespace fs = std::filesystem;

PythonManager::PythonManager() {
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
        try {
            fs::path scriptPath = fs::current_path() / "../py_src/src";
            py::module_ sys = py::module_::import("sys");
            sys.attr("path").attr("append")(scriptPath.string());
            AppendMessage(std::string("[PythonManager] Added Python path: ") + scriptPath.string());
        } catch (const std::exception& e) {
            AppendError(std::string("[PythonManager] Failed to add Python path: ") + e.what());
        }
    }
}

PythonManager::~PythonManager() {
    if (Py_IsInitialized()) {
        py::finalize_interpreter();
    }
}

py::object PythonManager::loadModule(const std::string& moduleName) {
    try {
        return py::module_::import(moduleName.c_str());
    } catch (const py::error_already_set& error) {
        AppendError(std::string("[PythonManager] Error loading module '" + moduleName + "': " + error.what()));
        return py::none();
    }
}

py::object PythonManager::callFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args) {
    try {
        if (!module.contains(funcName.c_str())) {
            AppendError(std::string("[PythonManager] Function '" + funcName + "' not found in module"));
            return py::none();
        }

        py::object func = module.attr(funcName.c_str());

        if (args.empty()) {
            return func();
        } else {
            return func(args);
        }
    } catch (const py::error_already_set& error) {
        AppendError(std::string("[PythonManager] Error calling function '" + funcName + "': " + error.what()));
        return py::none();
    }
}
