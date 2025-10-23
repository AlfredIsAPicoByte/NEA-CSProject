#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <pybind11/embed.h>

namespace py = pybind11;

class PythonManager {
public:
    PythonManager();
    ~PythonManager();

    py::object loadModule(const std::string& moduleName);

    py::object callFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args = {});
};