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
#include <memory>

#ifdef __unix__
#include <libgen.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

/*───────────────── CUDA Event Timer ───────────────────*/
// A simple RAII wrapper for CUDA events to handle creation and destruction.
struct CudaEventTimer {
    cudaEvent_t start_event, stop_event;

    CudaEventTimer() {
        CUDA_CHECK_ERROR(cudaEventCreate(&start_event));
        CUDA_CHECK_ERROR(cudaEventCreate(&stop_event));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CUDA_CHECK_ERROR(cudaEventRecord(start_event));
    }

    // Returns time in milliseconds
    float stop() {
        float milliseconds = 0;
        CUDA_CHECK_ERROR(cudaEventRecord(stop_event));
        CUDA_CHECK_ERROR(cudaEventSynchronize(stop_event));
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
};


// /*───────────────── simple wall-clock timer ───────────────────*/
// struct WallTimer {
//     std::chrono::steady_clock::time_point t0;
//     WallTimer()             : t0(std::chrono::steady_clock::now()) {}
//     double seconds() const  {
//         using d = std::chrono::duration<double>;
//         return std::chrono::duration_cast<d>(
//                    std::chrono::steady_clock::now() - t0).count();
//     }
// };

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
    std::vector<float> U_host(W * H, 0.0);
    initializeGrid(U_host.data(), W, H, config.bc);

    // double* d_U = nullptr;
    // CUDA_CHECK_ERROR(cudaMalloc(&d_U, W * H * sizeof(double)));

    // Using a smart pointer for RAII on the CUDA buffer
    std::unique_ptr<float, decltype(&cudaFree)> d_U(nullptr, &cudaFree);
    {
        float* temp_ptr = nullptr;
        CUDA_CHECK_ERROR(cudaMalloc(&temp_ptr, W * H * sizeof(float)));
        d_U.reset(temp_ptr);
    }

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
        CudaEventTimer timer;
        double total_ms = 0.0;

        if (solverType == "basic_cuda" || solverType == "all") {
            CUDA_CHECK_ERROR(cudaMemcpy(d_U.get(), U_host.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
            SolverBasic solver(d_U.get(), W, H, "BasicCUDASolver");

            timer.start();
            solver.solve(config.sim_params);
            float elapsed_ms = timer.stop();
            total_ms += elapsed_ms;
            
            std::cout << "[Timing] Basic CUDA solver took " << elapsed_ms << " ms\n";

            fs::path csv = fs::path(project_dir) / "solutions" / "solution_basic_cuda.csv";
            exportDeviceSolutionToCSV(d_U.get(), W, H, csv.string(), solver.getName());
            plot_solution("basic_cuda", csv.string());
        }

        /* ==== Shared-memory CUDA ============================ */

        if (solverType == "shared" || solverType == "all") {
            // Re-initialize device memory for a fair comparison
            CUDA_CHECK_ERROR(cudaMemcpy(d_U.get(), U_host.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
            SolverShared solver(d_U.get(), W, H, "SharedMemoryCUDASolver");

            timer.start();
            solver.solve(config.sim_params);
            float elapsed_ms = timer.stop();
            total_ms += elapsed_ms;

            std::cout << "[Timing] Shared-mem CUDA solver took " << elapsed_ms << " ms\n";

            // The SolverShared class uses a pitched allocation internally. We need to copy
            // the result from its internal pitched buffer back to our linear buffer (d_U) for exporting.
            CUDA_CHECK_ERROR(
                cudaMemcpy2D(d_U.get(),                    // Dst pointer
                             W * sizeof(float),           // Dst pitch
                             solver.data(),                // Src pointer (pitched)
                             solver.pitchElems() * sizeof(float), // Src pitch
                             W * sizeof(float), H,        // Width in bytes, and height
                             cudaMemcpyDeviceToDevice));

            fs::path csv = fs::path(project_dir) / "solutions" / "solution_shared_cuda.csv";
            exportDeviceSolutionToCSV(d_U.get(), W, H, csv.string(), solver.getName());
            plot_solution("shared", csv.string());
        }

        std::cout << "[Timing] === Total CUDA run time: " << total_ms / 1000.0 << " s ===\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Solver error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return 0;
}

