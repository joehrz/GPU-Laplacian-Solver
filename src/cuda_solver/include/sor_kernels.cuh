// src/cuda_solver/include/sor_kernels.cuh

#pragma once
#include "pitch2d.h"
#include <cuda_runtime.h>

// This single, templated kernel handles both the basic (global memory) and
// the optimized (shared memory) versions of the solver. The `TILE` template
// parameter controls which code path is compiled.
template<int TILE>
__global__
void sor_color_kernel(Pitch2D<float> grid,
                      int W, int H,
                      float omega,
                      int colour,
                      float* residualBlock)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // A shared memory array to store the partial residuals from each thread.
    // This is visible to all threads within the same block.
    __shared__ float partial_residuals[TILE > 0 ? TILE*TILE : 256];

    // Each thread calculates its unique 1D index within the block.
    const int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
    
    // Initialize this thread's contribution to the residual to zero.
    partial_residuals[thread_idx] = 0.0f;

    // ====================================================================
    //  Shared-memory path (compiled only when TILE > 0)
    // ====================================================================
    if constexpr (TILE > 0) {
        __shared__ float sm_tile[(TILE + 2) * (TILE + 2)];

        // 1. Load data from global to shared memory
        if (i < W && j < H) {
            const int li = threadIdx.x + 1;
            const int lj = threadIdx.y + 1;
            sm_tile[li + lj * (TILE + 2)] = grid.row(j)[i];
            if (threadIdx.x == 0 && i > 0)           sm_tile[0 + lj * (TILE + 2)] = grid.row(j)[i - 1];
            if (threadIdx.x == TILE - 1 && i < W - 1) sm_tile[TILE + 1 + lj * (TILE + 2)] = grid.row(j)[i + 1];
            if (threadIdx.y == 0 && j > 0)           sm_tile[li] = grid.row(j - 1)[i];
            if (threadIdx.y == TILE - 1 && j < H - 1) sm_tile[li + (TILE + 1) * (TILE + 2)] = grid.row(j + 1)[i];
        }
        __syncthreads();

        // 2. Perform SOR update from shared memory
        if (i > 0 && i < W - 1 && j > 0 && j < H - 1 && (((i + j) & 1) == colour)) {
            const int li = threadIdx.x + 1;
            const int lj = threadIdx.y + 1;
            float centre = sm_tile[li + lj * (TILE + 2)];
            float sigma  = (sm_tile[(li - 1) + lj * (TILE + 2)] +
                           sm_tile[(li + 1) + lj * (TILE + 2)] +
                           sm_tile[li + (lj - 1) * (TILE + 2)] +
                           sm_tile[li + (lj + 1) * (TILE + 2)]) * 0.25f;

            const float diff = sigma - centre;
            sm_tile[li + lj * (TILE + 2)] = centre + omega * diff;
            // Store this thread's contribution to the error
            partial_residuals[thread_idx] = fabsf(diff);
        }
        __syncthreads();

        // 3. Write updated tile back to global memory
        if (i < W && j < H) {
            grid.row(j)[i] = sm_tile[threadIdx.x + 1 + (threadIdx.y + 1) * (TILE + 2)];
        }
    }
    // ====================================================================
    //  Global-memory path (compiled only when TILE == 0)
    // ====================================================================
    else {
        if (i > 0 && i < W - 1 && j > 0 && j < H - 1 && (((i + j) & 1) == colour)) {
            float centre = grid.row(j)[i];
            float sigma  = (grid.row(j)[i - 1] + grid.row(j)[i + 1] +
                           grid.row(j - 1)[i] + grid.row(j + 1)[i]) * 0.25f;
            const float diff = sigma - centre;
            grid.row(j)[i] = centre + omega * diff;
            // Store this thread's contribution to the error
            partial_residuals[thread_idx] = fabsf(diff);
        }
    }

    __syncthreads(); // Wait for all SOR updates and residual calculations to complete.

    // ====================================================================
    //  Final Per-Block Reduction
    // ====================================================================
    // A standard parallel reduction algorithm using shared memory.
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            partial_residuals[thread_idx] += partial_residuals[thread_idx + s];
        }
        __syncthreads();
    }

    // After the loop, the first thread (index 0) holds the total sum for the entire block.
    // It writes this final value to the output array in global memory.
    if (thread_idx == 0) {
        if (residualBlock) {
            residualBlock[blockIdx.x + blockIdx.y * gridDim.x] = partial_residuals[0];
        }
    }
}