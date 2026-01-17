#include "PythonManager.h"

// Helper to map pip package names to their Python import names when they differ.
// Add entries here as needed (e.g. Pillow -> PIL).
static std::string GetImportNameForPackage(const std::string& pkg) {
    if (pkg == "Pillow") return "PIL";
    // Add more mappings if needed
    return pkg;
}

PythonManager::PythonManager() { Initialize(); }
PythonManager::~PythonManager() { Finalize(); }

#ifdef USE_EMBEDDED_PYTHON
void PythonManager::Initialize() {
    if (!Py_IsInitialized()) {
        try {
            // If a virtual environment is active (VIRTUAL_ENV), set it as PYTHONHOME so
            // the embedded interpreter can find the platform independent libraries.
            char *venvBuf = nullptr;
            size_t len = 0;
            if (_dupenv_s(&venvBuf, &len, "VIRTUAL_ENV") == 0 && venvBuf) {
                wchar_t* decoded = Py_DecodeLocale(venvBuf, nullptr);
                if (decoded) {
                    pythonHome = decoded; // keep pointer to free later
                    Py_SetPythonHome(pythonHome);
                    AppendPythonWarning("Py_SetPythonHome is deprecated");
                    AppendPythonMessage(std::string("Set PYTHONHOME to VIRTUAL_ENV: ") + venvBuf);
                }
                free(venvBuf);
            } else if (std::filesystem::exists("venv")) {
                std::filesystem::path venvPath = std::filesystem::absolute("venv");
                 AppendPythonMessage("Found local 'venv' folder, setting PYTHONHOME: " + venvPath.string());
                 pythonHome = Py_DecodeLocale(venvPath.string().c_str(), nullptr);
                 Py_SetPythonHome(pythonHome);
            } else {
                AppendPythonMessage("VIRTUAL_ENV not set; embedded interpreter will use system Python unless configured otherwise.");
            }

            py::initialize_interpreter();
            pythonInitialized = true;

            // Embedded interpreters often skip the 'site' initialization which adds pip packages.
            try {
                py::module_ site = py::module_::import("site");
                
                // Method A: Force 'site' to re-scan for packages
                if (py::hasattr(site, "main")) {
                    site.attr("main")(); 
                }
                
                // Method B: Explicitly add the user site-packages if Method A failed to grab them
                // This gets the standard location for pip packages on the current OS
                py::object getUserSite = site.attr("getusersitepackages");
                py::module_ sys = py::module_::import("sys");
                sys.attr("path").attr("append")(getUserSite());

                // Debug: Verify numpy is now findable
                try {
                    py::module_::import("numpy");
                    AppendPythonMessage("Verified: 'numpy' is accessible.");
                } catch(...) {
                    AppendPythonWarning("Warning: 'numpy' could not be imported immediately after init.");
                }

            } catch (const std::exception& e) {
                AppendPythonWarning(std::string("Failed to auto-configure site-packages: ") + e.what());
            }
            
            // Log prefix/executable info for easier diagnostics when embedding fails
            try {
                py::module_ sys = py::module_::import("sys");
                AppendPythonMessage(std::string("sys.prefix: ") + std::string(py::str(sys.attr("prefix")))); 
                AppendPythonMessage(std::string("sys.exec_prefix: ") + std::string(py::str(sys.attr("exec_prefix"))));
                try {
                    std::string pyExe = py::str(sys.attr("executable"));
                    AppendPythonMessage(std::string("Python executable: ") + pyExe);

                    // If VIRTUAL_ENV is set, warn if the embedded interpreter's executable
                    // doesn't appear to be inside that venv (common cause for install/import mismatch).
                    char *venvBuf = nullptr;
                    size_t len = 0;
                    if (_dupenv_s(&venvBuf, &len, "VIRTUAL_ENV") == 0 && venvBuf) {
                        std::string venvStr(venvBuf);
                        free(venvBuf);
                        if (pyExe.find(venvStr) == std::string::npos) {
                            AppendPythonWarning("Python executable does not appear to be inside VIRTUAL_ENV; pip installs may go to a different environment.");
                        }
                    }
                } catch (...) {
                    AppendPythonMessage("Python executable: (unknown)");
                }
            } catch (...) {
                AppendPythonWarning("Unable to query sys.* after interpreter initialization");
            }

            try {
                // Resolve an absolute path to the project's Python sources (py_src/src)
                std::filesystem::path scriptPath = std::filesystem::absolute(std::filesystem::current_path() / "py_src" / "src");
                if (!std::filesystem::exists(scriptPath)) {
                    // Try a fallback relative to parent directory (useful when running from build folders)
                    scriptPath = std::filesystem::absolute(std::filesystem::current_path().parent_path() / "py_src" / "src");
                }
                if (std::filesystem::exists(scriptPath)) {
                    py::module_ sys = py::module_::import("sys");
                    sys.attr("path").attr("append")(scriptPath.string());
                    AppendPythonMessage(std::string("Added Python path: ") + scriptPath.string());

                    // Ensure required packages (e.g. numpy) are installed for the embedded interpreter
                    EnsureRequiredPackagesInstalled();
                } else {
                    AppendPythonError(std::string("Python script path not found: ") + scriptPath.string());
                }
            } catch (const std::exception& e) {
                AppendPythonError(std::string("Failed to add Python path: ") + e.what());
            }

        } catch (const py::error_already_set& e) {
            AppendPythonError(std::string("Failed to initialize Python interpreter: ") + e.what());
            AppendPythonError("Check your Python installation, PYTHONHOME/VIRTUAL_ENV, and CMake Python settings (Python3_ROOT etc).");
            pythonInitialized = false;
            return;
        } catch (const std::exception& e) {
            AppendPythonError(std::string("Exception during Python initialization: ") + e.what());
            pythonInitialized = false;
            return;
        }
    }
}

void PythonManager::Finalize() {
    if (Py_IsInitialized()) py::finalize_interpreter();

    // Free locale-decoded PYTHONHOME string if allocated
    if (pythonHome) {
        PyMem_RawFree(pythonHome);
        pythonHome = nullptr;
    }
    pythonInitialized = false;
}

void PythonManager::InstallPackage(const std::string& packageName) {
    if (!pythonInitialized) {
        AppendPythonError("Python interpreter not initialized; cannot install packages");
        return;
    }

    // Avoid repeated installation attempts
    if (attemptedInstalls.find(packageName) != attemptedInstalls.end()) {
        AppendPythonMessage("Installation already attempted for package: " + packageName);
        return;
    }
    attemptedInstalls.insert(packageName);

    // 1. Import dependencies UP FRONT
    py::module_ sys = py::module_::import("sys");
    py::module_ subprocess = py::module_::import("subprocess");

    // 2. Determine the correct executable
    std::string cmdExecutable;
    try {
        std::string currentExe = sys.attr("executable").cast<std::string>();
        // If sys.executable contains "python" (e.g. "python.exe"), use it.
        if (currentExe.find("python") != std::string::npos) {
            cmdExecutable = currentExe;
        } else {
            // Otherwise, we are likely running inside the C++ host app.
            // Fallback to system command.
            #ifdef _WIN32
            cmdExecutable = "python"; 
            #else
            cmdExecutable = "python3";
            #endif
            AppendPythonWarning("sys.executable points to host app. Falling back to system '" + cmdExecutable + "' for pip calls.");
        }
    } catch(...) {
        cmdExecutable = "python";
    }

    // 3. Prepare the command: [python, -m, pip, install, package]
    auto commandArgs = py::make_tuple(cmdExecutable, "-m", "pip", "install", packageName);

    try {
#ifdef _WIN32
        // Windows: Try to hide the console window
        int creationFlags = 0;
        try {
            // Try newer CREATE_NO_WINDOW flag
            if (py::hasattr(subprocess, "CREATE_NO_WINDOW")) {
                 creationFlags = subprocess.attr("CREATE_NO_WINDOW").cast<int>();
            }
        } catch (...) { creationFlags = 0; }

        if (creationFlags != 0) {
            subprocess.attr("check_call")(commandArgs, py::arg("creationflags") = creationFlags);
        } else {
            // Fallback to STARTUPINFO for older methods
            try {
                py::object STARTUPINFO = subprocess.attr("STARTUPINFO");
                py::object si = STARTUPINFO();
                si.attr("dwFlags") = si.attr("dwFlags").cast<int>() | subprocess.attr("STARTF_USESHOWWINDOW").cast<int>();
                si.attr("wShowWindow") = subprocess.attr("SW_HIDE");
                subprocess.attr("check_call")(commandArgs, py::arg("startupinfo") = si);
            } catch (...) {
                // Last resort: show the window
                subprocess.attr("check_call")(commandArgs);
            }
        }
#else
        // Linux/Mac: No special flags needed
        subprocess.attr("check_call")(commandArgs);
#endif
        
        AppendPythonMessage("Successfully installed Python package: " + packageName);

        // 4. Invalidate import caches
        // Important: Python might not "see" the new package immediately unless we clear caches.
        try {
            py::module_::import("importlib").attr("invalidate_caches")();
        } catch(...) {}

        // 5. Verify Import
        try {
            std::string importName = GetImportNameForPackage(packageName);
            py::module_::import(importName.c_str());
            AppendPythonMessage(std::string("Verified import for package: ") + packageName + " (imported as: " + importName + ")");
        } catch (const py::error_already_set& e) {
            AppendPythonError(std::string("Package installed but import failed for '") + packageName + "': " + e.what());
        }

    } catch (const py::error_already_set& error) {
        AppendPythonError(std::string("Failed to install Python package '") + packageName + "': " + error.what());
    } catch (const std::exception& e) {
        AppendPythonError(std::string("Exception during installation of package '") + packageName + "': " + e.what());
    }
}

void PythonManager::ImportModule(const std::string& moduleName) {
    try {
        py::module_::import(moduleName.c_str());
        AppendPythonMessage("Successfully imported Python module: " + moduleName);
    } catch (const py::error_already_set& e) {
        AppendPythonError(std::string("Failed to import Python module '") + moduleName + "': " + e.what());
    }
}

void PythonManager::ValidatePackageInstallation(const std::string& packageName) {
    if (!pythonInitialized) {
        AppendPythonWarning(std::string("Python interpreter not initialized; cannot validate package: ") + packageName);
        return;
    }

    std::string importName = GetImportNameForPackage(packageName);
    try {
        py::module_::import(importName.c_str());
        AppendPythonMessage("Python package '" + packageName + "' is already installed (import: " + importName + ").");
    } catch (const py::error_already_set&) {
        AppendPythonMessage("Python package '" + packageName + "' not found (tried import: " + importName + "). Installing...");
        InstallPackage(packageName);
    }
}

void PythonManager::EnsurePythonPackagesInstalled(const std::vector<std::string>& packages) {
    if (!pythonInitialized) {
        AppendPythonWarning("Python interpreter not initialized; skipping package installation checks");
        return;
    }
    for (const auto& package : packages) ValidatePackageInstallation(package);
}

void PythonManager::GetPackagesInstalledStatus(const std::vector<std::string>& packages) {
    if (!pythonInitialized) {
        AppendPythonWarning("Python interpreter not initialized; cannot determine package status");
        return;
    }

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
        std::filesystem::path modulePath(path);
        // If provided path is not absolute, try resolving relative to current and parent directory
        if (!modulePath.is_absolute()) {
            std::filesystem::path candidate = std::filesystem::absolute(std::filesystem::current_path() / modulePath);
            if (std::filesystem::exists(candidate)) {
                modulePath = candidate;
            } else {
                candidate = std::filesystem::absolute(std::filesystem::current_path().parent_path() / modulePath);
                if (std::filesystem::exists(candidate)) modulePath = candidate;
            }
        }

        if (!std::filesystem::exists(modulePath)) {
            AppendPythonError(std::string("Python module path not found: ") + modulePath.string());
            return;
        }

        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(modulePath.string());
        AppendPythonMessage(std::string("Added Python path: ") + modulePath.string());
    } catch (const std::exception& e) {
        AppendPythonError(std::string("Failed to add Python path: ") + e.what());
    }
}

py::object PythonManager::LoadModule(const std::string& moduleName) {
    if (!pythonInitialized) {
        AppendPythonError(std::string("Python interpreter not initialized; cannot load module: ") + moduleName);
        return py::none();
    }

    try {
        return py::module_::import(moduleName.c_str());
    } catch (const py::error_already_set& error) {
        AppendPythonError(std::string("Error loading module '" + moduleName + "': " + error.what()));
        return py::none();
    }
}

py::object PythonManager::CallFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args) {
    try {
        if (!py::hasattr(module, funcName.c_str())) {
            AppendPythonError(std::string("Function '") + funcName + "' not found in module");
            return py::none();
        }
        py::object func = module.attr(funcName.c_str());
        if (args.empty()) return func();

        // Build a tuple from the args and call the function with that tuple as its positional arguments
        py::tuple t(args.size());
        for (size_t i = 0; i < args.size(); ++i) t[i] = args[i];

        // Use the Python C-API to call the function with the tuple of args so the tuple is unpacked
        PyObject* resultPtr = PyObject_CallObject(func.ptr(), t.ptr());
        if (!resultPtr) {
            // If the call failed, capture and log the Python error
            if (PyErr_Occurred()) PyErr_Print();
            AppendPythonError(std::string("Error calling function '") + funcName + "' (call returned null)");
            return py::none();
        }
        return py::reinterpret_steal<py::object>(resultPtr);
    } catch (const py::error_already_set& error) {
        AppendPythonError(std::string("Error calling function '") + funcName + "': " + error.what());
        return py::none();
    }
}

#else
void PythonManager::Initialize() {
    AppendPythonMessage("Python embedding is disabled. Initialize() is a no-op.");
}
void PythonManager::Finalize() {
    AppendPythonMessage("Python embedding is disabled. Finalize() is a no-op.");
}
void PythonManager::InstallPackage(const std::string& packageName) {
    AppendPythonMessage("Python embedding is disabled. InstallPackage() is a no-op for package: " + packageName);
}
void PythonManager::ImportModule(const std::string& moduleName) {
    AppendPythonMessage("Python embedding is disabled. ImportModule() is a no-op for module: " + moduleName);
}
void PythonManager::ValidatePackageInstallation(const std::string& packageName) {
    AppendPythonMessage("Python embedding is disabled. ValidatePackageInstallation() is a no-op for package: " + packageName);
}
void PythonManager::EnsurePythonPackagesInstalled(const std::vector<std::string>& packages) {
    AppendPythonMessage("Python embedding is disabled. EnsurePythonPackagesInstalled() is a no-op.");
}
void PythonManager::GetPackagesInstalledStatus(const std::vector<std::string>& packages) {
    AppendPythonMessage("Python embedding is disabled. GetPackagesInstalledStatus() is a no-op.");
}
void PythonManager::AddModulePath(const std::string& path) {
    AppendPythonMessage("Python embedding is disabled. AddModulePath() is a no-op for path: " + path);
}
py::object PythonManager::LoadModule(const std::string& moduleName) {
    AppendPythonMessage("Python embedding is disabled. LoadModule() is a no-op for module: " + moduleName);
    return py::none();
}
py::object PythonManager::CallFunction(const py::object& module, const std::string& funcName, const std::vector<py::object>& args) {
    AppendPythonMessage("Python embedding is disabled. CallFunction() is a no-op for function: " + funcName);
    return py::none();
}
#endif