// src/solver_thrust.cu

#include "solver_thrust.h"
#include "utilities.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>      // <- Add this
#include <thrust/execution_policy.h>      // <- And this

// Define the block size for CUDA kernels
#define BLOCK_SIZE 16

// Constructor
SolverThrust::SolverThrust(double* U, int width, int height, const std::string& name)
    : Solver(U, width, height, name) {}

// Destructor
SolverThrust::~SolverThrust() {}

struct SquareDiff {
    __device__
    double operator()(const thrust::tuple<double, double>& t) const {
        double d = thrust::get<0>(t) - thrust::get<1>(t);
        return d * d;
    }
};

// CUDA Kernel for Red-Black SOR Update (Thrust Optimized)
__global__ void sor_red_black_thrust_kernel(double* U, int width, int height, double omega, int color) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Determine the checkerboard color
    if ((i + j) % 2 != color)
        return; // Skip if not the current color

    // Ensure grid boundaries
    if (i <= 0 || i >= width - 1 || j <= 0 || j >= height - 1)
        return;

    int idx = i + j * width;

    // Compute the new value using the SOR formula
    double sigma = (U[idx - 1] + U[idx + 1] + U[idx - width] + U[idx + width]) / 4.0;
    U[idx] += omega * (sigma - U[idx]);
}

// Implementation of the solve method
void SolverThrust::solve() {
    const int MAX_ITER = 10000;
    const double TOL = 1e-6;
    const double omega = 1.85; // Relaxation factor

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    int iter = 0;
    double residual = 0.0;

    // Allocate U_prev
    double* U_prev;
    // Allocate with cudaMallocManaged for consistency
    CUDA_CHECK_ERROR(cudaMallocManaged(&U_prev, width * height * sizeof(double)));

    for (iter = 0; iter < MAX_ITER; ++iter) {
        // Copy U to U_prev
        CUDA_CHECK_ERROR(cudaMemcpy(U_prev, U, width * height * sizeof(double), cudaMemcpyDeviceToDevice));

        // Update Red nodes
        sor_red_black_thrust_kernel<<<gridSize, blockSize>>>(U, width, height, omega, 0);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Update Black nodes
        sor_red_black_thrust_kernel<<<gridSize, blockSize>>>(U, width, height, omega, 1);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Synchronize
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        // Compute residual using transform_reduce with a functor
        thrust::device_ptr<double> U_ptr(U);
        thrust::device_ptr<double> U_prev_ptr(U_prev);

        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(U_ptr, U_prev_ptr));
        auto zip_end = zip_begin + width * height;

        double sum = 0.0;
        try {
            sum = thrust::transform_reduce(
                thrust::device,
                zip_begin,
                zip_end,
                SquareDiff(),
                0.0,
                thrust::plus<double>()
            );
        } catch (thrust::system_error &e) {
            std::cerr << "Error during Thrust operation: " << e.what() << "\n";
            CUDA_CHECK_ERROR(cudaFree(U_prev));
            return;
        }

        residual = sqrt(sum);

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
        std::cout << "[" << solverName << "] Reached maximum iterations without convergence. Final Residual: " << residual << "\n";
    }

    // Free U_prev
    CUDA_CHECK_ERROR(cudaFree(U_prev));
}


// Implementation of the exportSolution method
void SolverThrust::exportSolution(const std::string& filename) {
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
