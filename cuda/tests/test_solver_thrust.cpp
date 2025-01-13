// tests/test_solver_thrust.cpp

#include "solver_thrust.h"
#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "utilities.h"
#include "solution_export.h"

#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem> // C++17
#include <vector>
#include <cstdlib> // For system()
#include <limits.h>
#include <string>


#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif



namespace fs = std::filesystem;

// Function to get the executable path
#ifdef _WIN32
std::string getExecutablePath() {
    char result[MAX_PATH];
    DWORD length = GetModuleFileNameA(nullptr, result, MAX_PATH);
    if (length == 0 || length == MAX_PATH) {
        throw std::runtime_error("Unable to determine executable path on Windows.");
    }
    return std::string(result, length);
}
#else
std::string getExecutablePath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count == -1) {
        throw std::runtime_error("Unable to determine executable path on Linux.");
    }
    return std::string(result, count);
}
#endif


// Function to get the project directory
std::string getProjectDir() {
    std::string exePath = getExecutablePath();
    fs::path exeDir = fs::path(exePath).parent_path(); // Directory containing the executable

    // Adjust the number of parent_path() calls based on your project structure
    // For example, if executable is in build_cpu/cpu/, and project root is GPU-Laplacian-Solver/
    fs::path projectDir = exeDir.parent_path().parent_path(); // Two levels up

    return projectDir.string();
}


// Function to read CSV file into a 2D vector
std::vector<std::vector<double>> read_csv(const std::string& filename, int width, int height) {
    std::vector<std::vector<double>> grid(height, std::vector<double>(width, 0.0));
    std::ifstream file(filename);
    std::string line;

    int j = 0;
    while (std::getline(file, line) && j < height) {
        size_t start = 0;
        size_t end = line.find(',');
        int i = 0;

        while (end != std::string::npos && i < width) {
            grid[j][i] = std::stod(line.substr(start, end - start));
            start = end + 1;
            end = line.find(',', start);
            i++;
        }

        if (i < width && start < line.size()) {
            grid[j][i] = std::stod(line.substr(start));
        }
        j++;
    }
    return grid;
}

int main() {
    std::cout << "Running Test: SolverBasic\n";
    std::string project_dir;

    project_dir = getProjectDir(); 

    // Define grid size
    const int width = 50;
    const int height = 50;

    // Define boundary conditions
    BoundaryConditions bc;
    bc.left = 0.0;
    bc.right = 0.0;
    bc.top = 0.0;
    bc.bottom = 0.0;

    // Create host array to store the grid
    std::vector<double> U_host(width * height, 0.0);

    // Initialize the grid with boundary conditions
    initializeGrid(U_host.data(), width, height, bc);


    // Allocate device memory
    double *d_U = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_U, width * height * sizeof(double)));

    // 4) Copy host array to device array
    CUDA_CHECK_ERROR(cudaMemcpy(d_U, U_host.data(),
                                width * height * sizeof(double),
                                cudaMemcpyHostToDevice));

    // Instantiate SolverThrust
    SolverThrust solver(d_U, width, height, "SolverThrust_Test");

    // Run the solver
    solver.solve();

    CUDA_CHECK_ERROR(cudaMemcpy(U_host.data(), d_U,
                                width * height * sizeof(double),
                                cudaMemcpyDeviceToHost));

    // Export the solution
    //std::string filename = "solutions/test_solution_basic.csv";
    fs::path solution_thrust_path = fs::path(project_dir) / "solutions" / "test_solution_thrust.csv";

    exportSolutionToCSV(
        solver.getDevicePtr(),
        width,
        height,
        solution_thrust_path.string(),
        solver.getName()
    );

    // Read the exported solution
    auto grid = read_csv(solution_thrust_path.string(), width, height);

    // Simple validation: Check boundary conditions
    for(int j = 0; j < height; ++j){
        assert(std::abs(grid[j][0] - bc.left) < 1e-3);
        assert(std::abs(grid[j][width-1] - bc.right) < 1e-3);
    }
    for(int i = 0; i < width; ++i){
        assert(std::abs(grid[0][i] - bc.top) < 1e-3);
        assert(std::abs(grid[height-1][i] - bc.bottom) < 1e-3);
    }

    // Further validations can include checking for expected values or convergence

    std::cout << "Test SolverThrust Passed.\n";

    // Free Unified Memory
    CUDA_CHECK_ERROR(cudaFree(d_U));

    return 0;
}
