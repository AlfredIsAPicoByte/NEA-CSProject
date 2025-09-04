#pragma once
#include <pybind11/embed.h>
#include <string>
#include <vector>

namespace py = pybind11;

class PythonManager {
public:
    PythonManager() { py::initialize_interpreter(); }
    ~PythonManager() { py::finalize_interpreter(); }

    py::object loadModule(const std::string& moduleName);

    py::object callFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args = {});
};