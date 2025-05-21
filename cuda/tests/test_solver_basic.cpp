// tests/test_solver_basic.cpp

#include "solver_basic.h"
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

/* ---------- helpers to locate the project root ---------------- */
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
    return exeDir.parent_path().parent_path().string();      // …/build/cuda/ → project
}

/* ---------- simple CSV reader --------------------------------- */
static std::vector<std::vector<double>>
read_csv(const std::string& file, int w, int h)
{
    std::vector<std::vector<double>> grid(h, std::vector<double>(w, 0.0));
    std::ifstream in(file);
    std::string line;
    int y = 0;
    while (std::getline(in, line) && y < h) {
        size_t start = 0, end = line.find(',');
        int x = 0;
        while (end != std::string::npos && x < w) {
            grid[y][x++] = std::stod(line.substr(start, end - start));
            start = end + 1;
            end   = line.find(',', start);
        }
        if (x < w && start < line.size()) grid[y][x] = std::stod(line.substr(start));
        ++y;
    }
    return grid;
}

/* =======================  main  =============================== */
int main() {
    std::cout << "Running Test: SolverBasic\n";
    const int width  = 50;
    const int height = 50;

    /* ---- boundary & host grid -------------------------------- */
    BoundaryConditions bc{0.0, 0.0, 0.0, 0.0};
    std::vector<double> U_host(width * height, 0.0);
    initializeGrid(U_host.data(), width, height, bc);

    /* ---- device buffer -------------------------------------- */
    double* d_U = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_U, width * height * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_U, U_host.data(),
                                width * height * sizeof(double),
                                cudaMemcpyHostToDevice));

    /* ---- solver --------------------------------------------- */
    SolverBasic solver(d_U, width, height, "SolverBasic_Test");
    SimulationParameters params{width, height, 1e-5, 20000, 1.9};   // NEW
    solver.solve(params);                                           // NEW

    CUDA_CHECK_ERROR(cudaMemcpy(U_host.data(), d_U,
                                width * height * sizeof(double),
                                cudaMemcpyDeviceToHost));

    /* ---- export & validation -------------------------------- */
    fs::path csvPath = fs::path(getProjectDir()) / "solutions" / "test_solution_basic.csv";
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
    std::cout << "Test SolverBasic Passed.\n";
    CUDA_CHECK_ERROR(cudaFree(d_U));
    return 0;
}