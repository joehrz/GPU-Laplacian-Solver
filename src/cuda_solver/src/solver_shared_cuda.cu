// src/cuda_solver/src/solver_shared_cuda.cu

#include "solver_shared_cuda.hpp"
#include "sor_kernels.cuh" // Our unified, templated kernel
#include "utilities.h"     // For CUDA_CHECK_ERROR
#include <thrust/device_vector.h>
#include "simulation_config.h"
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <iostream>

// Define the size of the tile for shared memory. Must be a power of 2
#define TILE_SIZE 32

// Constructor: Allocates *pitched* GPU memory for optimal alignment
SolverSharedMemCUDA::SolverSharedMemCUDA(float* host_grid, int w, int h, const std::string& name)
    : Solver(host_grid, w, h, name), d_grid_pitched(nullptr), pitchB(0) {
    
    // Use cudaMallocPitch for optimal memory alignment for 2D grids.
    CUDA_CHECK_ERROR(cudaMallocPitch(&d_grid_pitched, &pitchB, w * sizeof(float), h));

    // Use cudaMemcpy2D to correctly copy to pitched memory.
    CUDA_CHECK_ERROR(cudaMemcpy2D(d_grid_pitched, pitchB,
                                  host_grid, w * sizeof(float),
                                  w * sizeof(float), h,
                                  cudaMemcpyHostToDevice));
}

// Destructor
SolverSharedMemCUDA::~SolverSharedMemCUDA() {
    if (d_grid_pitched){
        cudaFree(d_grid_pitched);
    }
}

// The host-side solve method that orchestrates the optimized GPU computation.
void SolverSharedMemCUDA::solve(const SimulationParameters& prm) {
    // Kernel launch configuration. Block dimensions must match the tile size.
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
                    (height + TILE_SIZE - 1) / TILE_SIZE);

    // Calculate the required size of dynamic shared memory for our tile (including a 1-element halo).
    const size_t smBytes = (TILE_SIZE + 2) * (TILE_SIZE + 2) * sizeof(float);

    thrust::device_vector<float> d_residuals(grid.x * grid.y, 0.0f);
    
    // Create a pitched view of our memory for the kernel.
    Pitch2D<float> view{ d_grid_pitched, pitchB };

    for (int iter = 0; iter < prm.max_iterations; ++iter) {
        thrust::fill(d_residuals.begin(), d_residuals.end(), 0.0f);

        // --- Kernel Launches ---
        // Launch the kernel with shared memory and the TILE_SIZE template argument.
        sor_color_kernel<TILE_SIZE><<<grid, block, smBytes>>>(view, width, height, prm.omega, 0, thrust::raw_pointer_cast(d_residuals.data()));
        sor_color_kernel<TILE_SIZE><<<grid, block, smBytes>>>(view, width, height, prm.omega, 1, thrust::raw_pointer_cast(d_residuals.data()));
        
        CUDA_CHECK_ERROR(cudaGetLastError());

        // --- Convergence Check ---
        float total_residual = thrust::reduce(d_residuals.begin(), d_residuals.end(), 0.0f) / (width * height);

        if (iter > 0 && iter % 100 == 0) {
            std::cout << '[' << name << "] iter " << iter << "  residual = " << total_residual << '\n';
        }
        if (total_residual < prm.tolerance) {
            std::cout << '[' << name << "] converged in " << iter + 1 << " iterations.\n";
            return;
        }
    }
    std::cout << '[' << name << "] max iterations reached.\n";
}