#include "pythonManager.h"

PythonManager::PythonManager() {
    Py_Initialize();
}

PythonManager::~PythonManager() {
    Py_Finalize();
}

PyObject* PythonManager::loadModule(const std::string& moduleName) {
    PyObject* pName = PyUnicode_DecodeFSDefault(moduleName.c_str());
    if (!pName) return nullptr;

    PyObject* pModule = PyImport_Import(pName);
    Py_XDECREF(pName);

    return pModule; // nullptr if the module couldn't be loaded
}

PyObject* PythonManager::callFunction(PyObject* module, const std::string& funcName, const std::vector<PyObject*>& args) {
    if (!module) return nullptr;

    PyObject* pFunc = PyObject_GetAttrString(module, funcName.c_str());
    if (!pFunc || !PyCallable_Check(pFunc)) {
        Py_XDECREF(pFunc);
        return nullptr; // Function not found or not callable
    }

    PyObject* pArgs = PyTuple_New(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        Py_XINCREF(args[i]); // Increment reference count for each argument
        PyTuple_SetItem(pArgs, i, args[i]); // Note: This steals a reference to args[i]
    }

    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    Py_XDECREF(pArgs);
    Py_XDECREF(pFunc);

    return pValue; // nullptr if the call failed
}