// src/solver_basic.cu

#include "solver_basic.h"
#include "utilities.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

// Constructor
SolverBasic::SolverBasic(double* grid, int w, int h, const std::string& name)
    : Solver(grid, w, h, name) {}

// Destructor
SolverBasic::~SolverBasic() {}

// CUDA Kernel for Red-Black SOR Update (Basic)
__global__ void sor_red_black_kernel(double* U, int width, int height, double omega, int color, double* residuals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure within bounds
    if (i >= width || j >= height) return;

    // Determine the checkerboard color
    if ((i + j) % 2 != color)
        return; // Skip if not the current color

    // Ensure grid boundaries
    if (i <= 0 || i >= width - 1 || j <= 0 || j >= height - 1)
        return;

    int idx = i + j * width;

    // Compute the new value using the SOR formula
    double sigma = (U[idx - 1] + U[idx + 1] + U[idx - width] + U[idx + width]) / 4.0;
    double residual = fabs(sigma - U[idx]);
    U[idx] += omega * (sigma - U[idx]);

    // Accumulate residual using atomic operation to prevent data races
    atomicAdd(residuals, residual);
}

// Implementation of the solve method
void SolverBasic::solve() {
    const int MAX_ITER = 10000;
    const double TOL = 1e-6;
    const double omega = 1.85; // Relaxation factor

    // Define CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Allocate memory for residual on device
    double* d_residual;
    CUDA_CHECK_ERROR(cudaMallocManaged(&d_residual, sizeof(double)));

    int iter = 0;
    double residual = 0.0;

    for (iter = 0; iter < MAX_ITER; ++iter) {
        // Reset residual
        *d_residual = 0.0;

        // Update Red nodes (color = 0)
        sor_red_black_kernel<<<gridSize, blockSize>>>(U, width, height, omega, 0, d_residual);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Update Black nodes (color = 1)
        sor_red_black_kernel<<<gridSize, blockSize>>>(U, width, height, omega, 1, d_residual);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        // Compute residual (average residual)
        residual = *d_residual / (width * height);

        // Print progress every 100 iterations
        if (iter % 100 == 0) {
            std::cout << "[" << solverName << "] Iteration " << iter << " completed. Residual: " << residual << "\n";
        }

        // Check for convergence
        if (residual < TOL) {
            std::cout << "[" << solverName << "] Converged in " << iter + 1 << " iterations. Residual: " << residual << "\n";
            break;
        }
    }

    if (iter == MAX_ITER) {
        std::cout << "[" << solverName << "] Reached maximum iterations (" << MAX_ITER << ") without convergence. Final Residual: " << residual << "\n";
    }

    // Free residual memory
    CUDA_CHECK_ERROR(cudaFree(d_residual));
}

// Implementation of the exportSolution method
void SolverBasic::exportSolution(const std::string& filename) {
    // Ensure all device operations are complete
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[" << solverName << "] Error: Cannot open file " << filename << " for writing.\n";
        return;
    }

    file << std::fixed << std::setprecision(6);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int idx = i + j * width;
            file << U[idx];
            if (i < width - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "[" << solverName << "] Solution exported to " << filename << ".\n";
}