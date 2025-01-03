// cpu/src/red_black_sor.cpp

#include "solver_red_black.h"
#include "config.h"
#include "laplace_analytical_solution.h"        // Include concrete analytical solution
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>


// Constructor
SolverRedBlack::SolverRedBlack(double* grid, int w, int h, const std::string& name)
    : Solver(grid, w, h, name) {}

// Destructor
SolverRedBlack::~SolverRedBlack() {}

// Implementation of the solve method using Red-Black SOR
void SolverRedBlack::solve() {
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double maxError = 0.0;

        // Red update
        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1 + (i % 2); j < width - 1; j += 2) { // Red nodes
                int idx = i * width + j;
                double oldVal = U[idx];
                double newVal = 0.25 * (U[idx + 1] + U[idx - 1] + U[idx + width] + U[idx - width]);
                U[idx] = oldVal + OMEGA * (newVal - oldVal);
                maxError = std::max(maxError, std::abs(newVal - oldVal));
            }
        }

        // Black update
        for (int i = 1; i < height - 1; ++i) {
            for (int j = 2 - (i % 2); j < width - 1; j += 2) { // Black nodes
                int idx = i * width + j;
                double oldVal = U[idx];
                double newVal = 0.25 * (U[idx + 1] + U[idx - 1] + U[idx + width] + U[idx - width]);
                U[idx] = oldVal + OMEGA * (newVal - oldVal);
                maxError = std::max(maxError, std::abs(newVal - oldVal));
            }
        }

        if (maxError < TOL) {
            std::cout << "[" << solverName << "] Red-Black SOR converged after " << iter + 1 << " iterations.\n";
            return;
        }
    }
    std::cout << "[" << solverName << "] Red-Black SOR reached the maximum iteration limit.\n";
}



// Implementation of the exportSolution method
void SolverRedBlack::exportSolution(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[" << solverName << "] Error: Cannot open file " << filename << " for writing.\n";
        return;
    }

    file << std::fixed << std::setprecision(6);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int idx = i + j * width;
            file << U[idx];
            if (i < width - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "[" << solverName << "] Solution exported to " << filename << ".\n";
}
