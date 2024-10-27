// src/main.cpp

#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "solver_base.h"

#include "solver_basic.h"
#include "solver_shared.h"
#include "solver_thrust.h"

#include "utilities.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <filesystem> // C++17
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

namespace fs = std::filesystem;

// Function to get the directory of the executable
std::string getExecutableDir() {
    char buffer[1024];
    #ifdef _WIN32
        GetModuleFileNameA(NULL, buffer, sizeof(buffer));
        std::string path(buffer);
        return fs::path(path).parent_path().string();
    #else
        ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer));
        std::string path;
        if (count != -1) {
            path = std::string(buffer, count);
            return fs::path(path).parent_path().string();
        }
        return "";
    #endif
}

int main(int argc, char* argv[]){

    // Get executable directory
    std::string exe_dir = getExecutableDir();

    // Navigate up two levels to reach the 'cuda' directory
    fs::path path_exe(exe_dir);
    fs::path path_cuda = path_exe.parent_path().parent_path(); // Assuming exe_dir is '.../cuda/build/Release/'

    std::string cuda_dir = path_cuda.string();

    std::cout << "CUDA Directory: " << cuda_dir << "\n";

    // Construct paths relative to 'cuda_dir'
    std::string bc_file = cuda_dir + "/data/boundary_conditions.json"; // Adjusted Path
    if (argc > 1){
        bc_file = argv[1];
    }
    BoundaryConditions bc = loadBoundaryConditions(bc_file);

    // Default grid dimensions
    const int width = 100;
    const int height = 100;

    // Allocate Unified Memory for the grid
    double *U;
    CUDA_CHECK_ERROR(cudaMallocManaged(&U, width * height * sizeof(double)));

    // Initialize the grid with boundary conditions
    initializeGrid(U, width, height, bc);

    // Instantiate solver objects
    SolverBasic solverBasic(U, width, height, "BasicSolver");
    SolverShared solverShared(U, width, height, "SharedMemorySolver");
    SolverThrust solverThrust(U, width, height, "ThrustSolver");

    // Determine which solver to run based on command-line arguments
    // Usage: ./PDE_GPUSolver [boundary_conditions.json] [solver_type]
    // solver_type options: basic, shared, thrust, all

    std::string solverType = "all"; // Default to running all solvers
    if (argc > 2){
        solverType = argv[2];
    }  
    
        // Function to plot solutions
    auto plot_solution = [&](const std::string& solver_type, const std::string& filename){
        std::string script_path = cuda_dir + "/scripts/plot_solution.py"; // Adjusted Path
        std::string command = "python \"" + script_path + "\" \"" + solver_type + "\" \"" + filename + "\"";
        int ret = system(command.c_str());
        if (ret != 0){
            std::cerr << "Error: Plotting " << filename << " failed.\n";
        }
    };

    // Run the selected solver(s)
    if (solverType == "basic" || solverType == "all") {
        std::cout << "[Main] Running Basic Solver...\n";
        solverBasic.solve();
        std::string solution_basic = cuda_dir + "/data/solutions/solution_basic.csv"; // Adjusted Path
        solverBasic.exportSolution(solution_basic);
        plot_solution("basic", solution_basic);
        if (solverType == "all") {
            initializeGrid(U, width, height, bc); // Re-initialize for next solver
        }
    }

    if (solverType == "shared" || solverType == "all") {
        std::cout << "[Main] Running Shared Memory Solver...\n";
        solverShared.solve();
        std::string solution_shared = cuda_dir + "/data/solutions/solution_shared.csv"; // Adjusted Path
        solverShared.exportSolution(solution_shared);
        plot_solution("shared", solution_shared);
        if (solverType == "all") {
            initializeGrid(U, width, height, bc); // Re-initialize for next solver
        }
    }

    if (solverType == "thrust" || solverType == "all") {
        std::cout << "[Main] Running Thrust Optimized Solver...\n";
        solverThrust.solve();
        std::string solution_thrust = cuda_dir + "/data/solutions/solution_thrust.csv"; // Adjusted Path
        solverThrust.exportSolution(solution_thrust);
        plot_solution("thrust", solution_thrust);
    }

    // Free Unified Memory
    CUDA_CHECK_ERROR(cudaFree(U));

    return 0;
}

