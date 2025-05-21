// cpu/src/solver_basic.cpp

#include "solver_basic.h" // Contains declaration for SolverStandardSOR
#include <algorithm>      // For std::max
#include <cmath>          // For std::abs
#include <iostream>       // For std::cout


// Constructor 
SolverStandardSOR::SolverStandardSOR(double* grid, int w, int h, const std::string& name)
    : Solver(grid, w, h, name) {}

// Destructor
SolverStandardSOR::~SolverStandardSOR() {}

// Standard SOR method implementation
void SolverStandardSOR::solve(const SimulationParameters& sim_params) {
    for (int iter = 0; iter < sim_params.max_iterations; ++iter) { // USE sim_params.max_iterations
        double maxError = 0.0;

        for (int j = 1; j < height - 1; ++j) {
            for (int i = 1; i < width - 1; ++i) {
                int idx = i + j * width;
                double oldVal = U[idx];
                double newVal = 0.25 * (U[idx + 1] + U[idx - 1] + U[idx + width] + U[idx - width]);
                // USE sim_params.omega:
                U[idx] = oldVal + sim_params.omega * (newVal - oldVal);
                maxError = std::max(maxError, std::abs(newVal - oldVal));
            }
        }

        // USE sim_params.tolerance:
        if (maxError < sim_params.tolerance) {
            std::cout << "[" << solverName << "] Standard SOR converged after " << iter + 1 << " iterations.\n";
            return;
        }
    }
    std::cout << "[" << solverName << "] Standard SOR reached the maximum iteration limit (" << sim_params.max_iterations << ").\n";
}


