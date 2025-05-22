// cuda/src/main.cpp
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS   // silence getenv warning
#endif

#include "simulation_config.h"
#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "solution_export.h"
#include "solver_base.h"

#include "solver_basic.h"
#include "solver_shared.h"

#include "utilities.h"
#include "laplace_analytical_solution.h"

// ── Standard C++ headers ───────────────────────────────────────
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits.h>
#include <string>
#include <vector>

#ifdef __unix__
#include <libgen.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

/*───────────────── simple wall-clock timer ───────────────────*/
struct WallTimer {
    std::chrono::steady_clock::time_point t0;
    WallTimer()             : t0(std::chrono::steady_clock::now()) {}
    double seconds() const  {
        using d = std::chrono::duration<double>;
        return std::chrono::duration_cast<d>(
                   std::chrono::steady_clock::now() - t0).count();
    }
};

/*──────────────────────── helpers ─────────────────────────────*/
#ifdef _WIN32
bool CommandExists(const std::string& cmd) {
    std::string q = "where " + cmd + " >nul 2>&1";
    return system(q.c_str()) == 0;
}
#else
bool CommandExists(const std::string& cmd) {
    std::string q = "which " + cmd + " >/dev/null 2>&1";
    return system(q.c_str()) == 0;
}
#endif

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
        if (ok) { std::cerr << "[Debug] Using Python interpreter: " << p << '\n';
                  return p; }
    }
    throw std::runtime_error("Python not found.");
}

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
    return exeDir.parent_path().parent_path().parent_path().string(); // …/build/cuda/ ⇒ root
}

/* quote helper */
static std::string q(const std::string& s) {
    return (s.find_first_of(" \t") == std::string::npos) ? s : '"' + s + '"';
}

/*──────────────────────── main ───────────────────────────────*/
int main(int argc, char* argv[])
{
    std::string solverType = "all";
    std::string project_dir, script_path_str, config_file_path_str;
    FullConfig  config;

    /* -------- init paths & config --------------------------- */
    try {
        project_dir = getProjectDir();
        std::cout << "Project Directory: " << project_dir << '\n';

        config_file_path_str =
            (fs::path(project_dir) / "common" / "simulation_config.json").string();

        if (argc > 1) {
            std::string first = argv[1];
            if (first == "all" || first == "basic_cuda" || first == "shared" || first == "thrust")
                solverType = first;
            else {
                config_file_path_str = first;
                if (argc > 2) solverType = argv[2];
            }
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

    /* -------- grid setup ------------------------------------ */
    std::vector<double> U_host(W * H, 0.0);
    initializeGrid(U_host.data(), W, H, config.bc);

    double* d_U = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_U, W * H * sizeof(double)));

    /* -------- plotting lambda ------------------------------- */
    const std::string python_cmd = getPythonCommand();
    auto plot_solution = [&](const std::string& tag, const std::string& csv) {
        std::string cmd = q(python_cmd) + ' ' +
                          q(script_path_str) + ' ' +
                          tag + ' ' +
                          q(csv);
        std::cerr << "[Debug] Executing command: " << cmd << '\n';
        system(cmd.c_str());
    };

    /* -------- solver loop with timing ----------------------- */
    try {
        double totalSecs = 0.0;

        /* ==== Basic CUDA ==================================== */
        if (solverType == "basic_cuda" || solverType == "all") {
            WallTimer T;
            CUDA_CHECK_ERROR(cudaMemcpy(d_U, U_host.data(), W*H*sizeof(double),
                                        cudaMemcpyHostToDevice));

            SolverBasic solver(d_U, W, H, "BasicCUDASolver");
            solver.solve(config.sim_params);

            fs::path csv = fs::path(project_dir)/"solutions"/"solution_basic_cuda.csv";
            exportDeviceSolutionToCSV(d_U, W, H, csv.string(), solver.getName());
            plot_solution("basic_cuda", csv.string());

            double secs = T.seconds();
            totalSecs += secs;
            std::cout << "[Timing] Basic CUDA block took "
                      << secs << " s\n";
        }

        /* ==== Shared-memory CUDA ============================ */
        if (solverType == "shared" || solverType == "all") {
            WallTimer T;
        
            /* copy initial host grid into the *linear* buffer that main() owns */
            CUDA_CHECK_ERROR(
                cudaMemcpy(d_U, U_host.data(), W*H*sizeof(double),
                           cudaMemcpyHostToDevice));
        
            SolverShared solver(d_U, W, H, "SharedMemoryCUDASolver");
            solver.solve(config.sim_params);
        
        
            CUDA_CHECK_ERROR(
                cudaMemcpy2D(d_U,                         /* dst base + pitch */
                             W * sizeof(double),          /* dst pitch (bytes) */
                             solver.data(),               /* src pitched base  */
                             solver.pitchElems()*sizeof(double),
                             W * sizeof(double), H,
                             cudaMemcpyDeviceToDevice));
        
            fs::path csv = fs::path(project_dir) /
                           "solutions" / "solution_shared_cuda.csv";
            exportDeviceSolutionToCSV(d_U, W, H, csv.string(), solver.getName());
            plot_solution("shared", csv.string());
        
            const double secs = T.seconds();
            totalSecs += secs;
            std::cout << "[Timing] Shared-mem CUDA block took "
                      << secs << " s\n";
        }

        std::cout << "[Timing] === Total CUDA run time: "
                  << totalSecs << " s ===\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Solver error: " << e.what() << '\n';
        CUDA_CHECK_ERROR(cudaFree(d_U));
        return EXIT_FAILURE;
    }

    CUDA_CHECK_ERROR(cudaFree(d_U));
    return 0;
}


