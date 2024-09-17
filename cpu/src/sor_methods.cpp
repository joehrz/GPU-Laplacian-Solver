#include "sor_methods.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>

// Define the global constants
const int M = 100;
const int N = 100;
const double OMEGA = 1.85;
const int MAX_ITER = 10000;
const double TOL = 1e-6;

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// Analytical solution for comparison
double analyticalSolution(int x, int y, int m_local, int n_local) {
    double phi = 0.0;
    for (int n = 1; n <= 99; n += 2) {  // Sum over odd n only
        double term = (4 * 100 / (n * M_PI)) * sin(n * M_PI * x / m_local) * sinh(n * M_PI * y / m_local) / sinh(n * M_PI * n_local / m_local);
        phi += term;
    }
    return phi;
}
// Validate numerical solution against analytical solution
void validateSolution(const std::vector<std::vector<double>>& grid) {
    double maxError = 0.0;
    double totalError = 0.0;
    int count = 0;

    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double exact = analyticalSolution(i, j, M, N);
            if (exact != 0) {
                double error = fabs(grid[i][j] - exact);
                maxError = std::max(maxError, error);
                totalError += error;
                count++;
            }
        }
    }

    if (count > 0) {
        double avgError = totalError / count;
        std::cout << "Validation Results:\n";
        std::cout << "  Maximum error: " << maxError << std::endl;
        std::cout << "  Average error: " << avgError << std::endl;
    } else {
        std::cout << "No valid points to calculate average error." << std::endl;
    }
}

// Initialize grid with boundary conditions
void initializeGrid(std::vector<std::vector<double>>& grid) {
    for (int j = 0; j < N; j++) {
        grid[0][j] = 0;     // Bottom boundary
        grid[M-1][j] = 0;   // Top boundary
    }
    for (int i = 0; i < M; i++) {
        grid[i][0] = 0;     // Left boundary
        grid[i][N-1] = 100; // Right boundary
    }
}

// Measure the execution time of the SOR methods
double timeSOR(void (*sorMethod)(std::vector<std::vector<double>>&), std::vector<std::vector<double>>& grid) {
    auto start = std::chrono::high_resolution_clock::now();
    sorMethod(grid);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

// Export solution to a file for plotting
void exportSolution(const std::vector<std::vector<double>>& grid, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            file << i << " " << j << " " << grid[i][j] << std::endl;
        }
        file << std::endl; // Blank line to separate rows
    }
    file.close();
}

