// src/solver_shared.cu

#include "solver_shared.h"
#include "utilities.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
// Define the block size for CUDA kernels
#define BLOCK_SIZE 32

// Constructor
SolverShared::SolverShared(double* U, int width, int height, const std::string& name)
    : Solver(U, width, height, name) {}

// Destructor
SolverShared::~SolverShared() {}

// -----------------------------------------------------------------------------
// CUDA Kernel for Red-Black SOR Update with Shared Memory
// -----------------------------------------------------------------------------
__global__ void sor_red_black_shared_kernel(double* U,
                                            int width,
                                            int height,
                                            double omega,
                                            int color,
                                            double* residuals) 
{
    // Allocate shared memory for a tile
    __shared__ double tile[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2)];

    // Calculate global indices
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Calculate shared memory indices (offset by 1 for halo)
    int local_i = threadIdx.x + 1;
    int local_j = threadIdx.y + 1;

    // Load data into shared memory with halo
    if (i < width && j < height) {
        tile[local_i + local_j * (BLOCK_SIZE + 2)] = U[i + j * width];
    }

    // Load halo regions
    if (threadIdx.x == 0 && i > 0) {
        tile[0 + local_j * (BLOCK_SIZE + 2)] = U[(i - 1) + j * width];
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && i < width - 1) {
        tile[(BLOCK_SIZE + 1) + local_j * (BLOCK_SIZE + 2)] = U[(i + 1) + j * width];
    }
    if (threadIdx.y == 0 && j > 0) {
        tile[local_i + 0 * (BLOCK_SIZE + 2)] = U[i + (j - 1) * width];
    }
    if (threadIdx.y == BLOCK_SIZE - 1 && j < height - 1) {
        tile[local_i + (BLOCK_SIZE + 1) * (BLOCK_SIZE + 2)] = U[i + (j + 1) * width];
    }

    // Synchronize to ensure all data is loaded
    __syncthreads();

    // Perform computation only if within valid range
    if ((i < width - 1) && (j < height - 1) && (i > 0) && (j > 0)) {
        // Determine checkerboard color
        if ((i + j) % 2 == color) {
            // Compute sigma using shared memory
            double sigma = ( tile[local_i - 1 + local_j * (BLOCK_SIZE + 2)]
                           + tile[local_i + 1 + local_j * (BLOCK_SIZE + 2)]
                           + tile[local_i + (local_j - 1) * (BLOCK_SIZE + 2)]
                           + tile[local_i + (local_j + 1) * (BLOCK_SIZE + 2)] ) / 4.0;

            double oldVal = tile[local_i + local_j * (BLOCK_SIZE + 2)];
            double residual = fabs(sigma - oldVal);

            // Update global U
            U[i + j * width] += omega * (sigma - oldVal);

            // Accumulate residual using atomic operation
            atomicAdd(residuals, residual);
        }
    }
    // No need to __syncthreads() at the end
}

// -----------------------------------------------------------------------------
// Implementation of the solve method
// -----------------------------------------------------------------------------
void SolverShared::solve(){
    const int MAX_ITER = 20000;
    const double TOL   = 1e-5;
    const double omega = 1.92; // Relaxation factor

    // Define CUDA grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize( (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (height + BLOCK_SIZE - 1) / BLOCK_SIZE );

    // Instead of cudaMallocManaged, do plain cudaMalloc for residual
    double* d_residual = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_residual, sizeof(double)));

    double h_residual = 0.0; // host variable for residual

    int iter = 0;
    double residual = 0.0;

    for (iter = 0; iter < MAX_ITER; ++iter) {
        // Reset device residual to 0.0
        h_residual = 0.0;
        CUDA_CHECK_ERROR(cudaMemcpy(d_residual, &h_residual, sizeof(double),
                                    cudaMemcpyHostToDevice));

        // Update Red nodes (color = 0)
        sor_red_black_shared_kernel<<<gridSize, blockSize>>>(U, width, height,
                                                             omega, 0, d_residual);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Update Black nodes (color = 1)
        sor_red_black_shared_kernel<<<gridSize, blockSize>>>(U, width, height,
                                                             omega, 1, d_residual);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        // Copy the accumulated residual sum back to host
        CUDA_CHECK_ERROR(cudaMemcpy(&h_residual, d_residual, sizeof(double),
                                    cudaMemcpyDeviceToHost));

        // Compute residual (average)
        residual = h_residual / (width * height);

        // Print progress every 100 iterations
        if (iter % 100 == 0) {
            std::cout << "[" << solverName << "] Iteration " << iter
                      << " completed. Residual: " << residual << "\n";
        }

        // Check for convergence
        if (residual < TOL) {
            std::cout << "[" << solverName << "] Converged in " << iter + 1
                      << " iterations. Residual: " << residual << "\n";
            break;
        }
    }

    if (iter == MAX_ITER) {
        std::cout << "[" << solverName << "] Reached maximum iterations (" 
                  << MAX_ITER << ") without convergence. Final Residual: "
                  << residual << "\n";
    }

    // Free device residual memory
    CUDA_CHECK_ERROR(cudaFree(d_residual));
}

// --------------------------------------------------------------------------
// Implementation of the exportSolution method
// --------------------------------------------------------------------------
// void SolverShared::exportSolution(const std::string& filename) {
//      // Make sure GPU kernels have finished
//     CUDA_CHECK_ERROR(cudaDeviceSynchronize());

//     // 1) Allocate a temporary host buffer
//     std::vector<double> hostData(width * height);

//     // 2) Copy the device array 'U' into this host buffer
//     CUDA_CHECK_ERROR(
//         cudaMemcpy(hostData.data(), U, width * height * sizeof(double),
//                    cudaMemcpyDeviceToHost)
//     );

//     // 3) Now iterate over 'hostData' when writing to file
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "[" << solverName << "] Error: Cannot open file "
//                   << filename << " for writing.\n";
//         return;
//     }

//     file << std::fixed << std::setprecision(6);
//     for (int j = 0; j < height; ++j) {
//         for (int i = 0; i < width; ++i) {
//             int idx = i + j * width;
//             file << hostData[idx];
//             if (i < width - 1)
//                 file << ",";
//         }
//         file << "\n";
//     }

//     file.close();
//     std::cout << "[" << solverName << "] Solution exported to "
//               << filename << ".\n";
// }

