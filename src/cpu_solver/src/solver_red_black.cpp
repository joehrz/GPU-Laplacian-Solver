// src/cpu_solver/src/solver_red_black.cpp

#include "solver_red_black.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

SolverRedBlack::SolverRedBlack(float* grid, int w, int h,
                               const std::string& n)
    : Solver(grid, w, h, n)
{}

SolverRedBlack::~SolverRedBlack() = default;

// ─────────────────────────────────────────────────────────────────────────────
void SolverRedBlack::solve(const SimulationParameters& prm)
{
    const int   itMax = prm.max_iterations;
    const float tol   = prm.tolerance;
    const float omega = prm.omega;
    
    // DEBUG: Check grid dimensions
    std::cout << "[DEBUG RB] Solver dimensions: width=" << width << ", height=" << height << "\n";
    
    // DEBUG: Verify boundaries before solving
    // std::cout << "[DEBUG RB] Before solving - Bottom row values:\n";
    // for (int i = 0; i < width; ++i) {
    //     std::cout << grid_ptr[i + (height-1) * width] << " ";
    // }
    std::cout << "\n";
    
    for (int iter = 0; iter < itMax; ++iter)
    {
        float maxErr = 0.0f;
        
        // Two colour passes: 0 = red, 1 = black
        for (int colour = 0; colour < 2; ++colour)
        {
            // Interior rows only
            for (int j = 1; j < height - 1; ++j)
            {
                // DEBUG: Verify the starting index calculation
                int start_i = 1 + ((j + colour) & 1);
                if (start_i < 1 || start_i >= width - 1) {
                    std::cerr << "[ERROR RB] Invalid start_i: " << start_i 
                              << " for j=" << j << ", colour=" << colour << "\n";
                }
                
                // Start index depends on row-parity + colour
                for (int i = start_i; i < width - 1; i += 2)
                {
                    const int idx = i + j * width;
                    
                    // DEBUG: Verify we're not touching boundaries
                    if (j == 0 || j == height-1 || i == 0 || i == width-1) {
                        std::cerr << "[ERROR RB] Attempting to update boundary at (" 
                                  << i << "," << j << ")\n";
                        continue;
                    }
                    
                    // DEBUG: Check array bounds
                    if (idx < 0 || idx >= width * height) {
                        std::cerr << "[ERROR RB] Index out of bounds: " << idx 
                                  << " for (" << i << "," << j << ")\n";
                        continue;
                    }
                    
                    const float old = grid_ptr[idx];
                    const float sigma = (grid_ptr[idx - 1]     + grid_ptr[idx + 1] +
                                         grid_ptr[idx - width] + grid_ptr[idx + width]) * 0.25f;
                    const float diff  = sigma - old;
                    grid_ptr[idx]     = old + omega * diff;
                    maxErr = std::max(maxErr, std::fabs(diff));
                }
            }
        }
        
        // DEBUG: Check boundaries after first iteration
        // if (iter == 0) {
        //     std::cout << "[DEBUG RB] After iteration 1 - Bottom row values:\n";
        //     for (int i = 0; i < width; ++i) {
        //         std::cout << grid_ptr[i + (height-1) * width] << " ";
        //     }
        //     std::cout << "\n";
        // }
        
        if (maxErr < tol)
        {
            std::cout << '[' << name << "] Red-Black SOR converged in "
                      << iter + 1 << " iterations (residual=" << maxErr << ")\n";
            return;
        }
    }
    std::cout << '[' << name << "] Red-Black SOR hit the max-iteration limit ("
              << itMax << ")\n";
}