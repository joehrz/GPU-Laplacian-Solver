// tests/test_solver_shared.cpp

#include "solver_shared.h"
#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "utilities.h"
#include "solution_export.h"
#include "simulation_config.h"         

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

/* ---------- helpers (same as in SolverBasic test) ------------ */
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
    return exeDir.parent_path().parent_path().string();
}

static std::vector<std::vector<float>>
read_csv(const std::string& file, int w, int h)
{
    std::vector<std::vector<float>> grid(h, std::vector<float>(w, 0.0));
    std::ifstream in(file);
    std::string line;
    for (int y = 0; y < h && std::getline(in, line); ++y) {
        size_t start = 0, end; int x = 0;
        while ((end = line.find(',', start)) != std::string::npos && x < w) {
            grid[y][x++] = static_cast<float>(std::stod(line.substr(start, end - start)));
            start = end + 1;
        }
        if (x < w && start < line.size()) grid[y][x] = static_cast<float>(std::stod(line.substr(start)));
    }
    return grid;
}

/* =======================  main  =============================== */
int main() {
    std::cout << "Running Test: SolverShared\n";
    const int width  = 50;
    const int height = 50;

    BoundaryConditions bc{0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> U_host(width * height, 0.0f);
    initializeGrid(U_host.data(), width, height, bc);

    float* d_U = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_U, width * height * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_U, U_host.data(),
                                width * height * sizeof(float),
                                cudaMemcpyHostToDevice));

    SolverShared solver(d_U, width, height, "SolverShared_Test");
    SimulationParameters params{width, height, 1e-5f, 20000, 1.9f};   // NEW
    solver.solve(params);                                           // NEW

    CUDA_CHECK_ERROR(cudaMemcpy(U_host.data(), d_U,
                                width * height * sizeof(float),
                                cudaMemcpyDeviceToHost));

    fs::path csvPath = fs::path(getProjectDir()) / "solutions" / "test_solution_shared.csv";
    exportDeviceSolutionToCSV(solver.getDevicePtr(), width, height,
                              csvPath.string(), solver.getName());

    auto grid = read_csv(csvPath.string(), width, height);
    for (int y = 0; y < height; ++y) {
        assert(std::abs(grid[y][0]         - bc.left ) < 1e-3);
        assert(std::abs(grid[y][width - 1] - bc.right) < 1e-3);
    }
    for (int x = 0; x < width; ++x) {
        assert(std::abs(grid[0][x]          - bc.top   ) < 1e-3);
        assert(std::abs(grid[height - 1][x] - bc.bottom) < 1e-3);
    }
    std::cout << "Test SolverShared Passed.\n";
    CUDA_CHECK_ERROR(cudaFree(d_U));
    return 0;
}