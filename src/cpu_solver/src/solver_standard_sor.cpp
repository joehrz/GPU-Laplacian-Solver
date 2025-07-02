// src/cpu_solver/src/solver_standard_sor.cpp

#include "solver_standard_sor.hpp"
#include <algorithm>   // std::max
#include <cmath>
#include <iostream>

// Constructor simply forwards to the base class
SolverStandardSOR::SolverStandardSOR(float* grid, int w, int h,
                                     const std::string& n)
    : Solver(grid, w, h, n)
{}

// Defaulted destructor
SolverStandardSOR::~SolverStandardSOR() = default;

// ─────────────────────────────────────────────────────────────────────────────
void SolverStandardSOR::solve(const SimulationParameters& prm)
{
    const int   itMax = prm.max_iterations;
    const float tol   = prm.tolerance;
    const float omega = prm.omega;
    
    // DEBUG: Check grid dimensions
    std::cout << "[DEBUG] Solver dimensions: width=" << width << ", height=" << height << "\n";
    
    // DEBUG: Verify boundaries before solving
    // std::cout << "[DEBUG] Before solving - Bottom row values:\n";
    // for (int i = 0; i < width; ++i) {
    //     std::cout << grid_ptr[i + (height-1) * width] << " ";
    // }
    // std::cout << "\n";
    
    for (int iter = 0; iter < itMax; ++iter)
    {
        float maxErr = 0.0f;
        
        // DEBUG: Add bounds checking
        for (int j = 1; j < height - 1; ++j) {
            for (int i = 1; i < width - 1; ++i)
            {
                const int idx = i + j * width;
                
                // DEBUG: Verify we're not touching boundaries
                if (j == 0 || j == height-1 || i == 0 || i == width-1) {
                    std::cerr << "[ERROR] Attempting to update boundary at (" 
                              << i << "," << j << ")\n";
                    continue;
                }
                
                // DEBUG: Check array bounds
                if (idx < 0 || idx >= width * height) {
                    std::cerr << "[ERROR] Index out of bounds: " << idx 
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
        
        // DEBUG: Check boundaries after first iteration
        // if (iter == 0) {
        //     std::cout << "[DEBUG] After iteration 1 - Bottom row values:\n";
        //     for (int i = 0; i < width; ++i) {
        //         std::cout << grid_ptr[i + (height-1) * width] << " ";
        //     }
        //     std::cout << "\n";
        // }
        
        if (maxErr < tol)
        {
            std::cout << '[' << name << "] Standard SOR converged in "
                      << iter + 1 << " iterations (residual=" << maxErr << ")\n";
            return;
        }
    }
    std::cout << '[' << name << "] Standard SOR hit the max-iteration limit ("
              << itMax << ")\n";
}

// Verification helper function (optional - can be moved to a separate utility file)
void verifyGridIntegrity(const float* grid, int w, int h, const std::string& checkpoint) {
    std::cout << "\n[VERIFY] Grid integrity at: " << checkpoint << "\n";
    
    // Check for NaN or Inf values
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < w * h; ++i) {
        if (std::isnan(grid[i])) nan_count++;
        if (std::isinf(grid[i])) inf_count++;
    }
    
    if (nan_count > 0) std::cout << "  WARNING: Found " << nan_count << " NaN values!\n";
    if (inf_count > 0) std::cout << "  WARNING: Found " << inf_count << " Inf values!\n";
    
    // Check boundaries
    std::cout << "  First value (0,0): " << grid[0] << "\n";
    std::cout << "  Last value (" << (w-1) << "," << (h-1) << "): " 
              << grid[(h-1)*w + (w-1)] << "\n";
}