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
    PythonManager();
    ~PythonManager();

    // Loads a Python module by name
    PyObject* loadModule(const std::string& moduleName);

    // Calls a function from a module with no arguments
    PyObject* callFunction(PyObject* module, const std::string& funcName, const std::vector<PyObject*>& args = {});
};
