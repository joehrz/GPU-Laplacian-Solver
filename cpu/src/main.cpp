// cpu/src/main.cpp

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS   // silence getenv warning
#endif

#include "simulation_config.h"
#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "solver_base.h"
#include "laplace_analytical_solution.h"
#include "solver_basic.h"
#include "solver_red_black.h"
#include "solution_export.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <cmath>        // std::abs, std::sqrt
#include <limits.h>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <libgen.h>
#include <unistd.h>
#endif

using namespace std;
namespace fs = std::filesystem;

/*── forward declaration (default arg kept here) ───────────────*/
double computeL2Error(const std::vector<double>& numeric,
                      const std::vector<double>& exact,
                      int w, int h,
                      bool skipZeros = true);

/*──────────────────────── helper: command-exists ─────────────*/
#ifdef _WIN32
bool CommandExists(const std::string& cmd) {
    return system(("where " + cmd + " >nul 2>&1").c_str()) == 0;
}
#else
bool CommandExists(const std::string& cmd) {
    return system(("which " + cmd + " >/dev/null 2>&1").c_str()) == 0;
}
#endif

/*──────────────────────── helper: find Python ────────────────*/
std::string getPythonCommand()
{
#ifdef _WIN32
    std::vector<std::string> cand = {
        R"(C:\Python312\python.exe)",
        R"(C:\Python311\python.exe)",
        R"(C:\Python310\python.exe)",
        R"(C:\Python39\python.exe)"
    };
    if (const char* app = std::getenv("LOCALAPPDATA")) {
        std::string base = std::string(app) + R"(\Programs\Python\)";
        cand.push_back(base + R"(Python312\python.exe)");
        cand.push_back(base + R"(Python311\python.exe)");
        cand.push_back(base + R"(Python310\python.exe)");
        cand.push_back(base + R"(Python39\python.exe)");
    }
    cand.push_back("python.exe");
    cand.push_back("python3.exe");
#else
    std::vector<std::string> cand = {"python3", "python"};
#endif
    for (const auto& p : cand) {
#ifdef _WIN32
        bool ok = (p.find(':') != std::string::npos) ? fs::exists(p)
                                                     : CommandExists(p);
#else
        bool ok = CommandExists(p);
#endif
        if (ok) { std::cerr << "[Debug] Using Python interpreter: " << p << '\n'; return p; }
    }
    throw std::runtime_error("Python interpreter not found.");
}

/*──────────────────────── path helpers ───────────────────────*/
#ifdef _WIN32
std::string getExecutablePath() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) throw std::runtime_error("GetModuleFileNameA failed.");
    return std::string(buf, n);
}
#else
std::string getExecutablePath() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, PATH_MAX);
    if (n == -1) throw std::runtime_error("readlink failed.");
    return std::string(buf, n);
}
#endif

std::string getProjectDir() {
    fs::path exeDir = fs::path(getExecutablePath()).parent_path();
    return exeDir.parent_path().parent_path().parent_path().string();  // …/build/cpu/ ⇒ root
}

/* quote helper */
static std::string q(const std::string& s) {
    return (s.find_first_of(" \t") == std::string::npos) ? s : '"' + s + '"';
}

/*──────────────────────── globals ─────────────────────────────*/
std::string project_dir;
std::string script_path_str;
std::string config_file_path_str;
FullConfig  config;

/*──────────────────────── main ───────────────────────────────*/
int main(int argc, char* argv[])
{
    /*──── 1.  load config & paths ───────────────────────────*/
    try {
        project_dir = getProjectDir();
        std::cout << "Project Directory: " << project_dir << '\n';

        config_file_path_str =
            //(fs::path(project_dir) / "boundary_conditions" / "boundary_conditions.json").string();
            (fs::path(project_dir) / "common" / "simulation_config.json").string();

        if (argc > 1) {                  // allow overriding config path
            std::string first = argv[1];
            if (first != "all" && first != "basic_cpu" && first != "red_black_cpu")
                config_file_path_str = first;
        }

        config = loadConfiguration(config_file_path_str);
        fs::create_directories(fs::path(project_dir) / "solutions");

        script_path_str =
            (fs::path(project_dir) / "scripts" / "plot_solution.py").string();
    }
    catch (const std::exception& e) {
        std::cerr << "Init error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    const int W = config.sim_params.width;
    const int H = config.sim_params.height;

    /*──── 2.  problem setup ─────────────────────────────────*/
    UniversalFourierSolution analytical(config.bc.left,  config.bc.right,
                                        config.bc.top,   config.bc.bottom,
                                        25);
    std::vector<double> U_exact = analytical.compute(W, H);
    std::vector<double> U(W * H, 0.0);
    initializeGrid(U.data(), W, H, config.bc);

    SolverStandardSOR solverSOR(U.data(), W, H, "BasicSOR");
    SolverRedBlack    solverRB (U.data(), W, H, "RedBlackSOR");

    /*──── 3.  parse solver type ─────────────────────────────*/
    std::string solverType = "all";
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "all" || arg1 == "basic_cpu" || arg1 == "red_black_cpu")
            solverType = arg1;
        else if (argc > 2)
            solverType = argv[2];
    }

    /*──── 4.  plotting lambda ───────────────────────────────*/
    const std::string python_cmd = getPythonCommand();
    auto plot_solution = [&](const std::string& tag, const std::string& csv) {
        std::string cmd = q(python_cmd) + ' ' +
                          q(script_path_str) + ' ' +
                          tag + ' ' +
                          q(csv);
        std::cerr << "[Debug] " << cmd << '\n';
        system(cmd.c_str());
    };

    /*──── 5.  run chosen solvers ────────────────────────────*/
    try {
        if (solverType == "basic_cpu" || solverType == "all") {
            initializeGrid(U.data(), W, H, config.bc);
            auto t0 = std::chrono::high_resolution_clock::now();
            solverSOR.solve(config.sim_params);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::cout << "Basic SOR time: "
                      << std::chrono::duration<double>(t1 - t0).count() << " s\n";
            std::cout << "L2 error: "
                      << computeL2Error(U, U_exact, W, H) << '\n';

            fs::path csv = fs::path(project_dir) / "solutions" / "solution_basic_cpu.csv";
            exportHostDataToCSV(U.data(), W, H, csv.string(), solverSOR.getName());
            plot_solution("basic_cpu", csv.string());
        }

        if (solverType == "red_black_cpu" || solverType == "all") {
            initializeGrid(U.data(), W, H, config.bc);
            auto t0 = std::chrono::high_resolution_clock::now();
            solverRB.solve(config.sim_params);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::cout << "Red-Black SOR time: "
                      << std::chrono::duration<double>(t1 - t0).count() << " s\n";
            std::cout << "L2 error: "
                      << computeL2Error(U, U_exact, W, H) << '\n';

            fs::path csv = fs::path(project_dir) / "solutions" / "solution_red_black_cpu.csv";
            exportHostDataToCSV(U.data(), W, H, csv.string(), solverRB.getName());
            plot_solution("red_black_cpu", csv.string());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Solver error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return 0;
}

/*──────────────────────── computeL2Error impl ────────────────*/
double computeL2Error(const std::vector<double>& numeric,
                      const std::vector<double>& exact,
                      int w, int h,
                      bool skipZeros)
{
    double sum2 = 0.0; int cnt = 0;
    for (int j = 1; j < h - 1; ++j)
        for (int i = 1; i < w - 1; ++i) {
            double e = exact[j * w + i];
            if (skipZeros && std::abs(e) < 1e-15) continue;
            double d = numeric[j * w + i] - e;
            sum2 += d * d; ++cnt;
        }
    return cnt ? std::sqrt(sum2 / cnt) : 0.0;
}













