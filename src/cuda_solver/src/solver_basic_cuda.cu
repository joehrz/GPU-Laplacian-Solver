// src/cuda_solver/src/solver_basic_cuda.cu

#include "solver_basic_cuda.hpp"
#include "sor_kernels.cuh" 
#include "utilities.h"     
#include "pitch2d.h"
#include "simulation_config.h" 
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <iostream>

// Constructor: Allocates GPU memory and copies the initial grid from the host
SolverBasicCUDA::SolverBasicCUDA(float* host_grid, int w, int h, const std::string& name)
    : Solver(host_grid, w, h, name), d_grid(nullptr) {
    
    const size_t gridSize = static_cast<size_t>(w) * h * sizeof(float);
    CUDA_CHECK_ERROR(cudaMalloc(&d_grid, gridSize));
    CUDA_CHECK_ERROR(cudaMemcpy(d_grid, host_grid, gridSize, cudaMemcpyHostToDevice));
}

// Destructor: Frees the GPU memory
SolverBasicCUDA::~SolverBasicCUDA(){
    if (d_grid){
        cudaFree(d_grid);
    }
}

// The host-side solve method that orchestrates the GPU computation
void SolverBasicCUDA::solve(const SimulationParameters& prm){
    // Set up the kernel launch configuration
    const dim3 block(32, 8); // 256 threads per block
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    // Allocate a device vector (managed by Thrust) to store the residula from each block
    thrust::device_vector<float> d_residuals(grid.x * grid.y, 0.0f);

    // Create a non-pitched view of our linear memory for the kernel
    Pitch2D<float> view{ d_grid, static_cast<size_t>(width) * sizeof(float) };

    for (int iter = 0; iter < prm.max_iterations; ++iter){
        // Reset residuals to zero before each iteration
        thrust::fill(d_residuals.begin(), d_residuals.end(), 0.0f);

        // --- Kernel Launches ---
        // Launch the kernel for the red pass. Note the <0> template argument
        sor_color_kernel<0><<<grid, block>>>(view, width, height, prm.omega, 0, thrust::raw_pointer_cast(d_residuals.data()));

        // Launch the kernel for the black pass
        sor_color_kernel<0><<<grid, block>>>(view, width, height, prm.omega, 1, thrust::raw_pointer_cast(d_residuals.data()));

        CUDA_CHECK_ERROR(cudaGetLastError()); // Check for any errors during kernel launch

        // --- Convergence Check ---
        // Use Thrust to perform a parallel reduction (sum) on the block residuals.
        float total_residual = thrust::reduce(d_residuals.begin(), d_residuals.end(), 0.0f) / (width * height);
        
        if (iter > 0 && iter % 100 == 0) {
            std::cout << '[' << name << "] iter " << iter << "  residual = " << total_residual << '\n';
        }

        if (total_residual < prm.tolerance) {
             std::cout << '[' << name << "] converged in " << iter + 1 << " iterations.\n";
             return; // Exit early if converged
        }
    }
    std::cout << '[' << name << "] max iterations reached.\n";
}
