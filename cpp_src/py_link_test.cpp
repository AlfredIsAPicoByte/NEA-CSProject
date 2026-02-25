#include <pybind11/embed.h>
#include <iostream>
#include <filesystem>
#include <vector>

namespace py = pybind11;
namespace fs = std::filesystem;

int main()
{
    try {
        py::scoped_interpreter guard{};

        py::module_ sys = py::module_::import("sys");

        // Try several reasonable locations for the project's py_src so import("tester") works.
        std::vector<fs::path> candidates;
        fs::path cwd = fs::current_path();
        candidates.push_back(cwd / "py_src");
        candidates.push_back(cwd / "py_src" / "src");
        candidates.push_back(cwd.parent_path() / "py_src");
        candidates.push_back(cwd.parent_path() / "py_src" / "src");

        for (auto &p : candidates) {
            if (fs::exists(p) && fs::is_directory(p)) {
                sys.attr("path").attr("insert")(0, p.string());
            }
        }

        // Import and run the python tester
        py::module_ tester = py::module_::import("tester");
        bool ok = tester.attr("run_unit_tests")().cast<bool>();

        std::cout << "[cpp-tester] python tests returned: " << (ok ? "OK" : "FAIL") << std::endl;
        return ok ? 0 : 1;
    }
    catch (const std::exception &e) {
        std::cerr << "[cpp-tester][EXCEPTION] " << e.what() << std::endl;
        return 2;
    }
}