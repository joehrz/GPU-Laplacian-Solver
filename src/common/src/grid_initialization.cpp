// src/common/src/grid_initialization.cpp

#include "grid_initialization.h"
#include <iostream>

void initializeGrid(float* U, int W, int H, const BoundaryConditions& bc)
{
    // Initialize interior points to zero
    for (int j = 1; j < H - 1; ++j)
        for (int i = 1; i < W - 1; ++i)
            U[i + j * W] = 0.0f;
    
    // Set boundary conditions with convention: j = 0 → top, j = H-1 → bottom
    for (int i = 0; i < W; ++i) {
        U[i]                   = bc.top;                 // j = 0 (top row)
        U[i + (H - 1) * W]     = bc.bottom;              // j = H-1 (bottom row)
    }
    
    // Left and right boundaries (excluding corners)
    for (int j = 1; j < H - 1; ++j) {
        U[j * W]               = bc.left;                // i = 0 (left column)
        U[(W - 1) + j * W]     = bc.right;               // i = W-1 (right column)
    }
}