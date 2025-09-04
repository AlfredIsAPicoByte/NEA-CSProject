#include "pythonManger.h"

py::object PythonManager::loadModule (const std::string& moduleName) {
    try {
        return py::module_::import(moduleName.c_str());
    } catch (const py::error_already_set& e) {
        // Handle error (e.g., log it)
        return py::none();
    }
}

py::object PythonManager::callFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args = {}) {
    py::object func = module.attr(funcName.c_str());
    if (args.empty()) {
        return func();
    } else {
        return func(*args);
    }
}