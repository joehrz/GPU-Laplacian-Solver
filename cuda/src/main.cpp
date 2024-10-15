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

int main(int argc, char* argv[]){

    // Load boundary conditions from a JSON file
    std::string bc_file = "data/boundary_conditions.json";
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

    // Determine which solver to run base on command-line arguments
    // Usage: ./PDE_GPUSolver [boundary_conditions.json] [solver_type]
    // solver_type options: basic, shared, thrust, all

    std::string solverType = "all"; // Default to running all solvers
    if (argc > 2){
        solverType = argv[2];

    }  
    
    // Functions to plot solutions
    auto plot_solution = [&](const std::string& filename){
        std::string command = "python scripts/plot_solution.py " + filename;
        int ret = system(command.c_str());
        if (ret != 0){
            std::cerr << "Error: Plotting " << filename << " failed.\n";
        }
    };

    // Run the selected solver(s)
    if (solverType == "basic" || solverType == "all") {
        std::cout << "[Main] Running Basic Solver...\n";
        solverBasic.solve();
        solverBasic.exportSolution("data/solutions/solution_basic.csv");
        plot_solution("data/solutions/solution_basic.csv");
        if (solverType == "all") {
            initializeGrid(U, width, height, bc); // Re-initialize for next solver
        }
    }

    if (solverType == "shared" || solverType == "all") {
        std::cout << "[Main] Running Shared Memory Solver...\n";
        solverShared.solve();
        solverShared.exportSolution("data/solutions/solution_shared.csv");
        plot_solution("data/solutions/solution_shared.csv");
        if (solverType == "all") {
            initializeGrid(U, width, height, bc); // Re-initialize for next solver
        }
    }

    if (solverType == "thrust" || solverType == "all") {
        std::cout << "[Main] Running Thrust Optimized Solver...\n";
        solverThrust.solve();
        solverThrust.exportSolution("data/solutions/solution_thrust.csv");
        plot_solution("data/solutions/solution_thrust.csv");
    }

    // Free Unified Memory
    CUDA_CHECK_ERROR(cudaFree(U));

    return 0;
}