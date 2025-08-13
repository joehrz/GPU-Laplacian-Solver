// src/cuda_solver/src/solver_mixed_bc_cuda.cu

#include "solver_mixed_bc_cuda.hpp"
#include "neumann_kernels.cuh"
#include "utilities.h"
#include "simulation_config.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <iostream>

SolverMixedBCCUDA::SolverMixedBCCUDA(float* grid_ptr_, int w, int h, const std::string& name_)
    : Solver(grid_ptr_, w, h, name_), d_grid(nullptr) {
    
    // Assume unit square domain for now
    grid_spacing = 1.0f / (w - 1);
    
    const size_t gridSize = static_cast<size_t>(w) * h * sizeof(float);
    CUDA_CHECK_ERROR(cudaMalloc(&d_grid, gridSize));
    CUDA_CHECK_ERROR(cudaMemcpy(d_grid, grid_ptr_, gridSize, cudaMemcpyHostToDevice));
}

SolverMixedBCCUDA::~SolverMixedBCCUDA() {
    if (d_grid) {
        cudaFree(d_grid);
    }
}

void SolverMixedBCCUDA::setBoundaryConditions(const BoundaryConditions& bc) {
    boundary_conditions = bc;
}

SolverStatus SolverMixedBCCUDA::solve(const SimulationParameters& prm) {
    dim3 block(32, 8);
    dim3 grid((width_ + block.x - 1) / block.x,
              (height_ + block.y - 1) / block.y);
    
    thrust::device_vector<float> d_residuals(grid.x * grid.y, 0.0f);
    
    // For applying Neumann BC
    dim3 bc_block(256);
    dim3 bc_grid_x((width_ + bc_block.x - 1) / bc_block.x);
    dim3 bc_grid_y((height_ + bc_block.x - 1) / bc_block.x);
    
    for (int iter = 0; iter < prm.max_iterations; ++iter) {
        thrust::fill(d_residuals.begin(), d_residuals.end(), 0.0f);
        
        // Apply Neumann boundary conditions before each iteration
        apply_neumann_bc_kernel<<<bc_grid_x, bc_block>>>(
            d_grid, width_, height_, grid_spacing,
            boundary_conditions.top, boundary_conditions.bottom,
            boundary_conditions.left, boundary_conditions.right);
        
        // Red pass
        size_t shared_size = block.x * block.y * sizeof(float);
        sor_mixed_bc_kernel<0><<<grid, block, shared_size>>>(
            d_grid, width_, height_, prm.omega, 0,
            boundary_conditions.top, boundary_conditions.bottom,
            boundary_conditions.left, boundary_conditions.right,
            thrust::raw_pointer_cast(d_residuals.data()));
        
        // Apply Neumann BC again
        apply_neumann_bc_kernel<<<bc_grid_x, bc_block>>>(
            d_grid, width_, height_, grid_spacing,
            boundary_conditions.top, boundary_conditions.bottom,
            boundary_conditions.left, boundary_conditions.right);
        
        // Black pass
        sor_mixed_bc_kernel<0><<<grid, block, shared_size>>>(
            d_grid, width_, height_, prm.omega, 1,
            boundary_conditions.top, boundary_conditions.bottom,
            boundary_conditions.left, boundary_conditions.right,
            thrust::raw_pointer_cast(d_residuals.data()));
        
        CUDA_CHECK_ERROR(cudaGetLastError());
        
        // Convergence check
        float total_residual = thrust::reduce(d_residuals.begin(), d_residuals.end(), 0.0f) / (width_ * height_);
        
        if (iter > 0 && iter % 100 == 0) {
            std::cout << '[' << name_ << "] iter " << iter << "  residual = " << total_residual << '\n';
        }
        
        if (total_residual < prm.tolerance) {
            std::cout << '[' << name_ << "] converged in " << iter + 1 << " iterations.\n";
            CUDA_CHECK_ERROR(cudaMemcpy(grid_ptr_, d_grid, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost));
            SolverStatus status;
            status.iterations = iter + 1;
            status.residual = total_residual;
            status.converged = true;
            status.message = "MixedBCCUDA converged";
            updateStatus(status);
            return status;
        }
    }
    
    std::cout << '[' << name_ << "] max iterations reached.\n";
    CUDA_CHECK_ERROR(cudaMemcpy(grid_ptr_, d_grid, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost));
    SolverStatus status;
    status.iterations = prm.max_iterations;
    status.residual = 0.0f;
    status.converged = false;
    status.message = "MixedBCCUDA reached maximum iterations";
    updateStatus(status);
    return status;
}