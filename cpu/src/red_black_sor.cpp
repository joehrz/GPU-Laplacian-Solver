// cpu/src/red_black_sor.cpp

#include "solver_red_black.h"
#include "laplace_analytical_solution.h"        // Include concrete analytical solution
#include <algorithm>
#include <cmath>
#include <iostream>


// Constructor
SolverRedBlack::SolverRedBlack(double* grid, int w, int h, const std::string& name)
    : Solver(grid, w, h, name) {}

// Destructor
SolverRedBlack::~SolverRedBlack() {}

// Implementation of the solve method using Red-Black SOR
void SolverRedBlack::solve(const SimulationParameters& sim_params) {
    for (int iter = 0; iter < sim_params.max_iterations; ++iter) { // USE sim_params
        double maxError = 0.0;

        // Red update
        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1 + (i % 2); j < width - 1; j += 2) { // Red nodes
                int idx = i * width + j;
                double oldVal = U[idx];
                double newVal = 0.25 * (U[idx + 1] + U[idx - 1] + U[idx + width] + U[idx - width]);
                // USE sim_params:
                U[idx] = oldVal + sim_params.omega * (newVal - oldVal);
                maxError = std::max(maxError, std::abs(newVal - oldVal)); // Or std::abs(U[idx] - oldVal)
            }
        }

        // Black update
        for (int i = 1; i < height - 1; ++i) {
            for (int j = 2 - (i % 2); j < width - 1; j += 2) { // Black nodes
                int idx = i * width + j;
                double oldVal = U[idx];
                double newVal = 0.25 * (U[idx + 1] + U[idx - 1] + U[idx + width] + U[idx - width]);
                // USE sim_params 
                U[idx] = oldVal + sim_params.omega * (newVal - oldVal);
                maxError = std::max(maxError, std::abs(newVal - oldVal)); // Or std::abs(U[idx] - oldVal)
            }
        }

        // USE sim_params 
        if (maxError < sim_params.tolerance) {
            std::cout << "[" << solverName << "] Red-Black SOR converged after " << iter + 1 << " iterations.\n";
            return;
        }
    }
    // USE sim_params:
    std::cout << "[" << solverName << "] Red-Black SOR reached the maximum iteration limit (" << sim_params.max_iterations << ").\n";
}


