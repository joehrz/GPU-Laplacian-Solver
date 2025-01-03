// cpu/src/solver_basic.cpp

#include "solver_basic.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>


// Constructor
SolverStandardSOR::SolverStandardSOR(double* grid, int w, int h, const std::string& name)
    : Solver(grid, w, h, name) {}

// Destructor
SolverStandardSOR::~SolverStandardSOR() {}

// Standard SOR method implementation
void SolverStandardSOR::solve() {
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double maxError = 0.0;

        for (int j = 1; j < height - 1; ++j) {
            for (int i = 1; i < width - 1; ++i) {
                int idx = i + j * width;
                double oldVal = U[idx];
                double newVal = 0.25 * (U[idx + 1] + U[idx - 1] + U[idx + width] + U[idx - width]);
                U[idx] = oldVal + OMEGA * (newVal - oldVal);
                maxError = std::max(maxError, std::abs(newVal - oldVal));
            }
        }

        if (maxError < TOL) {
            std::cout << "[" << solverName << "] Standard SOR converged after " << iter + 1 << " iterations.\n";
            return;
        }
    }
    std::cout << "[" << solverName << "] Standard SOR reached the maximum iteration limit.\n";
}

// Implementation of the exportSolution method
void SolverStandardSOR::exportSolution(const std::string& filename) {
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
