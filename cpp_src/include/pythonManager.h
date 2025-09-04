#pragma once

#include <Python.h>
#include <string>
#include <vector>

class PythonManager {
public:
    PythonManager();
    ~PythonManager();

    // Loads a Python module by name
    PyObject* loadModule(const std::string& moduleName);

    // Calls a function from a module with no arguments
    PyObject* callFunction(PyObject* module, const std::string& funcName, const std::vector<PyObject*>& args = {});
};
