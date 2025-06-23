// common/src/grid_initialization.cpp

#include "grid_initialization.h" 
#include "boundary_conditions.h" 
#include <iostream>
#include <vector>

void initializeGrid(float *U, int width, int height, const BoundaryConditions& bc) {
    // Initialize all interior points to zero
    for (int j = 1; j < height - 1; ++j) {
        for (int i = 1; i < width - 1; ++i) {
            U[i + j * width] = 0.0f;
        }
    }

    // Set boundary conditions
    for (int j = 0; j < height; ++j) {
        U[0 + j * width] = static_cast<float>(bc.left);              // Left boundary
        U[(width - 1) + j * width] = static_cast<float>(bc.right);  // Right boundary
    }

    for (int i = 0; i < width; ++i) {
        U[i + 0 * width] = static_cast<float>(bc.top);               // Top boundary
        U[i + (height - 1) * width] = static_cast<float>(bc.bottom); // Bottom boundary
    }

    std::cout << "Grid initialized with boundary conditions." << std::endl;
}