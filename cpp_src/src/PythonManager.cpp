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
            py::initialize_interpreter();
            pythonInitialized = true;

            py::module_ sys = py::module_::import("sys");
            
            // Check for local venv in project directory
            std::filesystem::path venvPath = std::filesystem::current_path() / "venv";
            
            if (std::filesystem::exists(venvPath)) {
                #ifdef _WIN32
                std::filesystem::path venvSitePackages = venvPath / "Lib" / "site-packages";
                #else
                std::filesystem::path libPath = venvPath / "lib";
                std::filesystem::path venvSitePackages;
                for (const auto& entry : std::filesystem::directory_iterator(libPath)) {
                    if (entry.is_directory() && 
                        entry.path().filename().string().find("python3.") == 0) {
                        venvSitePackages = entry.path() / "site-packages";
                        break;
                    }
                }
                #endif
                
                if (std::filesystem::exists(venvSitePackages)) {
                    // Add venv site-packages FIRST (highest priority)
                    sys.attr("path").attr("insert")(0, venvSitePackages.string());
                    AppendPythonMessage("✓ Using local venv: " + venvSitePackages.string());
                }
            } else { // Fall back to system site-packages
                AppendPythonWarning("No local venv found, using system Python");
                
                py::module_ site = py::module_::import("site");
                py::object site_packages = site.attr("getsitepackages")();
                
                if (py::isinstance<py::list>(site_packages)) {
                    for (auto path : site_packages) {
                        std::string pathStr = py::str(path);
                        sys.attr("path").attr("insert")(0, pathStr);
                        AppendPythonMessage("Added site-packages: " + pathStr);
                    }
                }
            }
            
            // Log diagnostic info
            try {
                py::module_ sys = py::module_::import("sys");
                AppendPythonMessage("sys.prefix: " + std::string(py::str(sys.attr("prefix"))));
                AppendPythonMessage("sys.executable: " + std::string(py::str(sys.attr("executable"))));
                
                py::list paths = sys.attr("path");
                AppendPythonMessage("Python sys.path entries:");
                for (size_t i = 0; i < py::len(paths); ++i) {
                    AppendPythonMessage("  [" + std::to_string(i) + "] " + std::string(py::str(paths[i])));
                }
            } catch (...) {
                AppendPythonWarning("Could not query sys info");
            }

            // Add custom module paths
            try {
                std::filesystem::path scriptPath = std::filesystem::absolute(
                    std::filesystem::current_path() / "py_src" / "src"
                );
                if (!std::filesystem::exists(scriptPath)) {
                    scriptPath = std::filesystem::absolute(
                        std::filesystem::current_path().parent_path() / "py_src" / "src"
                    );
                }
                if (std::filesystem::exists(scriptPath)) {
                    py::module_ sys = py::module_::import("sys");
                    sys.attr("path").attr("insert")(0, scriptPath.string());
                    AppendPythonMessage("Added custom Python path: " + scriptPath.string());
                } else {
                    AppendPythonError("Python script path not found: " + scriptPath.string());
                }
            } catch (const std::exception& e) {
                AppendPythonError("Failed to add custom Python path: " + std::string(e.what()));
            }

            AppendPythonMessage("Checking for required Python packages...");
            for (const auto& package : requiredPythonPackages) {
                std::string importName = GetImportNameForPackage(package);
                try {
                    py::module_::import(importName.c_str());
                    AppendPythonMessage("✓ " + package + " found");
                } catch (const py::error_already_set&) {
                    AppendPythonError("✗ " + package + " NOT FOUND. Please install manually.");
                }
            }

        } catch (const py::error_already_set& e) {
            AppendPythonError("Failed to initialize Python interpreter: " + std::string(e.what()));
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

    std::string pythonExe;
    try {
        pythonExe = sys.attr("executable").cast<std::string>();
    } catch(...) {
        pythonExe = "python";
    }

    if (pythonExe.find("python") == std::string::npos) {
        // Try to find Python in the same directory as sys.executable
        std::filesystem::path exePath(pythonExe);
        std::filesystem::path exeDir = exePath.parent_path();
        
        #ifdef _WIN32
        std::filesystem::path venvPython = exeDir / "Scripts" / "python.exe";
        if (!std::filesystem::exists(venvPython)) {
            venvPython = exeDir / "python.exe";
        }
        #else
        std::filesystem::path venvPython = exeDir / "bin" / "python";
        if (!std::filesystem::exists(venvPython)) {
            venvPython = exeDir / "python3";
        }
        #endif
        
        if (std::filesystem::exists(venvPython)) {
            pythonExe = venvPython.string();
            AppendPythonMessage("Found venv Python: " + pythonExe);
        } else {
            AppendPythonWarning("Could not locate Python executable, using 'python'");
            pythonExe = "python";
        }
    }
    
    AppendPythonMessage("Installing " + packageName + " using: " + pythonExe);

    auto commandArgs = py::make_tuple(pythonExe, "-m", "pip", "install", packageName);

    try {
        subprocess.attr("check_call")(commandArgs);
        AppendPythonMessage("Successfully installed: " + packageName);
        
        try {
            // Invalidate import caches
            py::module_::import("importlib").attr("invalidate_caches")();
            
            // Reload site module using importlib.reload()
            py::module_ importlib = py::module_::import("importlib");
            py::module_ site = py::module_::import("site");
            importlib.attr("reload")(site);
        } catch(...) {}
        
        // Verify import
        try {
            std::string importName = GetImportNameForPackage(packageName);
            py::module_::import(importName.c_str());
            AppendPythonMessage("✓ Verified import: " + importName);
        } catch (const py::error_already_set& e) {
            AppendPythonError("✗ Package installed but import failed: " + std::string(e.what()));
        }
        
    } catch (const py::error_already_set& error) {
        AppendPythonError("Failed to install " + packageName + ": " + std::string(error.what()));
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
    
    bool isAnaconda = false;
    try {
        py::module_ sys = py::module_::import("sys");
        std::string prefix = py::str(sys.attr("prefix"));
        std::string executable = py::str(sys.attr("executable"));
        
        if (prefix.find("anaconda") != std::string::npos || 
            prefix.find("miniconda") != std::string::npos ||
            executable.find("anaconda") != std::string::npos ||
            executable.find("miniconda") != std::string::npos) {
            isAnaconda = true;
            AppendPythonMessage("Detected Anaconda/Miniconda environment");
        }
    } catch(...) {}
    
    for (const auto& package : packages) {
        std::string importName = GetImportNameForPackage(package);
        
        try {
            py::module_::import(importName.c_str());
            AppendPythonMessage("✓ Package '" + package + "' is already available");
        } catch (const py::error_already_set&) {
            if (isAnaconda) {
                // Don't try to install with pip in Anaconda - user should use conda
                AppendPythonWarning("Package '" + package + "' not found. Please install with: conda install " + package);
            } else {
                AppendPythonMessage("Package '" + package + "' not found. Installing...");
                InstallPackage(package);
            }
        }
    }
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