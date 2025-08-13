// src/cuda_solver/include/neumann_kernels.cuh

#pragma once
#include <cuda_runtime.h>
#include "boundary_conditions.h"

// Kernel to apply Neumann boundary conditions
// For Neumann BC: du/dn = g, where n is outward normal
// Using ghost point method: u_ghost = u_interior - h*g (for 1st order)
// or u_boundary = u_interior - h*g/2 (for 2nd order at boundary)
__global__ void apply_neumann_bc_kernel(float* grid, int W, int H, float h,
                                       const BoundaryEdge top, const BoundaryEdge bottom,
                                       const BoundaryEdge left, const BoundaryEdge right) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Top boundary (j = 0)
    if (idx < W && top.type == BCType::NEUMANN) {
        // Second-order approximation: u_0 = u_1 - h * g
        grid[idx] = grid[idx + W] - h * top.value;
    }
    
    // Bottom boundary (j = H-1)
    if (idx < W && bottom.type == BCType::NEUMANN) {
        int j = H - 1;
        grid[idx + j * W] = grid[idx + (j - 1) * W] + h * bottom.value;
    }
    
    // Left boundary (i = 0)
    if (idx < H && left.type == BCType::NEUMANN) {
        // Skip corners
        if (idx > 0 && idx < H - 1) {
            grid[idx * W] = grid[1 + idx * W] - h * left.value;
        }
    }
    
    // Right boundary (i = W-1)
    if (idx < H && right.type == BCType::NEUMANN) {
        // Skip corners
        if (idx > 0 && idx < H - 1) {
            int i = W - 1;
            grid[i + idx * W] = grid[(i - 1) + idx * W] + h * right.value;
        }
    }
}

// Modified SOR kernel that handles mixed boundary conditions
template<int TILE>
__global__ void sor_mixed_bc_kernel(float* grid, int W, int H, float omega, int colour,
                                   const BoundaryEdge top, const BoundaryEdge bottom,
                                   const BoundaryEdge left, const BoundaryEdge right,
                                   float* residualBlock) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    extern __shared__ float partial_residuals[];
    const int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
    partial_residuals[thread_idx] = 0.0f;
    
    if (i < W && j < H) {
        // Check if this is an interior point or Neumann boundary
        bool is_interior = (i > 0 && i < W - 1 && j > 0 && j < H - 1);
        bool is_neumann_boundary = false;
        
        // Check if on Neumann boundary
        if (j == 0 && top.type == BCType::NEUMANN) is_neumann_boundary = true;
        if (j == H - 1 && bottom.type == BCType::NEUMANN) is_neumann_boundary = true;
        if (i == 0 && left.type == BCType::NEUMANN) is_neumann_boundary = true;
        if (i == W - 1 && right.type == BCType::NEUMANN) is_neumann_boundary = true;
        
        if ((is_interior || is_neumann_boundary) && (((i + j) & 1) == colour)) {
            float center = grid[j * W + i];
            
            // Get neighbor values, using one-sided differences at boundaries
            float vleft   = (i > 0) ? grid[j * W + (i - 1)] : grid[j * W + (i + 1)];
            float vright  = (i < W - 1) ? grid[j * W + (i + 1)] : grid[j * W + (i - 1)];
            float vtop    = (j > 0) ? grid[(j - 1) * W + i] : grid[(j + 1) * W + i];
            float vbottom = (j < H - 1) ? grid[(j + 1) * W + i] : grid[(j - 1) * W + i];
            
            float sigma = (vleft + vright + vtop + vbottom) * 0.25f;
            float diff = sigma - center;
            
            grid[j * W + i] = center + omega * diff;
            partial_residuals[thread_idx] = fabsf(diff);
        }
    }
    
    __syncthreads();
    
    // Parallel reduction for residual
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            partial_residuals[thread_idx] += partial_residuals[thread_idx + s];
        }
        __syncthreads();
    }
    
    if (thread_idx == 0 && residualBlock) {
        residualBlock[blockIdx.x + blockIdx.y * gridDim.x] = partial_residuals[0];
    }
}