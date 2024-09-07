#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h> // Include for CUDART_PI
#include <fstream>

// Constants
const int M = 100;
const int N = 100;
const int MAX_ITER = 10000;
const double TOL = 1e-6;

__constant__ double OMEGA = 1.85;


#define CUDA_CHECK_ERROR(call)                                  \
    {                                                           \
        const cudaError_t error = call;                         \
        if (error != cudaSuccess) {                             \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__; \
            std::cerr << ", code: " << error                    \
                      << ", reason: " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                            \
        }                                                       \
    }

// Analytical solution for comparison (unchanged)
double analyticalSolution(int x, int y, int M, int N) {
    double phi = 0.0;
    double M_double = static_cast<double>(M);
    double N_double = static_cast<double>(N);

    for (int n = 1; n <= 99; n += 2) {
        double numerator = 4.0 * 100.0 / (n * CUDART_PI);
        double sine_term = sin(n * CUDART_PI * x / M_double);
        double sinh_term = sinh(n * CUDART_PI * y / M_double);
        double denominator = sinh(n * CUDART_PI * N_double / M_double);

        double term = numerator * sine_term * sinh_term / denominator;
        phi += term;
    }
    return phi;
}

void printGrid(const double *grid, int width, int height, const char *name) {
    std::cout << name << " Grid:" << std::endl;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int index = i + j * width;
            std::cout << grid[index] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Kernel to update red nodes
__global__ void updateRedNodes(double *U, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * width;

    if (i >= 1 && i < width - 1 && j >= 1 && j < height - 1 && ((i + j) % 2 == 0)) {
        double old_val = U[index];
        double temp = (U[index - 1] + U[index + 1] + U[index - width] + U[index + width]) / 4.0;
        double res = temp - old_val;

        U[index] = old_val + OMEGA * res;  // Apply SOR relaxation
    }
}

// Kernel to update black nodes
__global__ void updateBlackNodes(double *U, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * width;

    if (i >= 1 && i < width - 1 && j >= 1 && j < height - 1 && ((i + j) % 2 == 1)) {
        double old_val = U[index];
        double temp = (U[index - 1] + U[index + 1] + U[index - width] + U[index + width]) / 4.0;
        double res = temp - old_val;

        U[index] = old_val + OMEGA * res;  // Apply SOR relaxation
    }
}

// Kernel to compute residual (max difference between U and U_old)
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void computeResidual(double *U, double *U_old, double *d_maxResidual, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * width;

    if (i >= 1 && i < width - 1 && j >= 1 && j < height - 1) {
        double residual = fabs(U[index] - U_old[index]);
        atomicMaxDouble(d_maxResidual, residual);  // Use custom atomic max for double
    }
}

// Red-Black SOR method with residual check
void redBlackSOR(double *U, int width, int height) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    double *U_old;  // Array to hold old values for residual calculation
    cudaMallocManaged(&U_old, width * height * sizeof(double));

    double *d_maxResidual;  // To hold the max residual value on the device
    cudaMallocManaged(&d_maxResidual, sizeof(double));

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Copy U to U_old to track the previous iteration values
        cudaMemcpy(U_old, U, width * height * sizeof(double), cudaMemcpyDeviceToDevice);

        // Reset max residual to 0
        *d_maxResidual = 0.0;

        // Perform Red step
        updateRedNodes<<<gridSize, blockSize>>>(U, width, height);
        cudaDeviceSynchronize();

        // Perform Black step
        updateBlackNodes<<<gridSize, blockSize>>>(U, width, height);
        cudaDeviceSynchronize();

        // Calculate the residual and find the maximum value
        computeResidual<<<gridSize, blockSize>>>(U, U_old, d_maxResidual, width, height);
        cudaDeviceSynchronize();

        // Retrieve max residual and check convergence
        if (*d_maxResidual < TOL) {
            std::cout << "Converged after " << iter << " iterations.\n";
            break;  // Exit if converged
        }

        if (iter == MAX_ITER - 1) {
            std::cout << "Reached maximum iterations (" << MAX_ITER << ").\n";
        }
    }

    // Free memory
    cudaFree(U_old);
    cudaFree(d_maxResidual);
}

// Initialization function (simplified: no weights are used in the current version)
void initializeGrid(double *U, int width, int height) {
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int index = i + j * width;
            U[index] = 0.0;  // Initial guess for U
        }
    }

    // Set boundary conditions for U
    for (int j = 0; j < height; ++j) {
        U[j * width] = 0.0;             // Left boundary
        U[(j + 1) * width - 1] = 0.0;   // Right boundary
    }
    for (int i = 0; i < width; ++i) {
        U[i] = 0.0;                     // Top boundary
        U[(height - 1) * width + i] = 100.0; // Bottom boundary
    }
}

// Export U grid to a file for plotting (unchanged)
void exportSolution(const double *U, int width, int height, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            file << i << " " << j << " " << U[i + j * width] << "\n";
        }
        file << "\n"; // Blank line to separate rows
    }
    file.close();
}

int main() {
    int width = M;
    int height = N;

    double *U;
    cudaMallocManaged(&U, width * height * sizeof(double));

    // Initialize grid and boundary conditions
    initializeGrid(U, width, height);

    // Run the Red-Black SOR solver
    redBlackSOR(U, width, height);

    // Export solution
    exportSolution(U, width, height, "solution.txt");

    // Free memory
    cudaFree(U);

    return 0;
}
