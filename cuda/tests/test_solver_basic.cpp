// tests/test_solver_basic.cpp

#include "solver_basic.h"
#include "boundary_conditions.h"
#include "grid_initialization.h"
#include "utilities.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>


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

    // Define grid size
    const int width = 50;
    const int height = 50;

    // Define boundary conditions
    BoundaryConditions bc;
    bc.left = 0.0;
    bc.right = 0.0;
    bc.top = 100.0;
    bc.bottom = 0.0;

    // Allocate Unified Memory for the grid
    double* U;
    CUDA_CHECK_ERROR(cudaMallocManaged(&U, width * height * sizeof(double)));

    // Initialize the grid with boundary conditions
    initializeGrid(U, width, height, bc);

    // Instantiate SolverBasic
    SolverBasic solver(U, width, height, "SolverBasic_Test");

    // Run the solver
    solver.solve();

    // Export the solution
    std::string filename = "data/solutions/test_solution_basic.csv";
    solver.exportSolution(filename);

    // Read the exported solution
    auto grid = read_csv(filename, width, height);

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

    std::cout << "Test SolverBasic Passed.\n";

    // Free Unified Memory
    CUDA_CHECK_ERROR(cudaFree(U));

    return 0;
}