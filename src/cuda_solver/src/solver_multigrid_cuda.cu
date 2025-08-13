#include "solver_multigrid_cuda.hpp"
#include "multigrid_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 16

__global__ void jacobi_smooth_kernel(float* u, const float* f, float* u_new, 
                                   int width, int height, float h2, float omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= 0 && i < width && j >= 0 && j < height) {
        int idx = j * width + i;
        
        // Copy boundary values unchanged
        if (i == 0 || i == width - 1 || j == 0 || j == height - 1) {
            u_new[idx] = u[idx];
        } else {
            float u_old = u[idx];
            float u_jacobi = 0.25f * (u[idx - 1] + u[idx + 1] + 
                                      u[idx - width] + u[idx + width] - h2 * f[idx]);
            u_new[idx] = (1.0f - omega) * u_old + omega * u_jacobi;
        }
    }
}

__global__ void compute_residual_kernel(const float* u, const float* f, float* residual,
                                      int width, int height, float h2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < width - 1 && j > 0 && j < height - 1) {
        int idx = j * width + i;
        float lap_u = (u[idx - 1] + u[idx + 1] + u[idx - width] + u[idx + width] - 4.0f * u[idx]) / h2;
        residual[idx] = f[idx] - lap_u;
    } else if (i < width && j < height) {
        residual[j * width + i] = 0.0f;
    }
}

__global__ void restrict_kernel(const float* fine_residual, float* coarse_f,
                              int fine_width, int fine_height,
                              int coarse_width, int coarse_height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < coarse_width && j < coarse_height) {
        int coarse_idx = j * coarse_width + i;
        
        if (i == 0 || i == coarse_width - 1 || j == 0 || j == coarse_height - 1) {
            coarse_f[coarse_idx] = 0.0f;
        } else {
            int fi = 2 * i;
            int fj = 2 * j;
            
            float sum = 0.0f;
            sum += 0.25f * fine_residual[fj * fine_width + fi];
            sum += 0.125f * (fine_residual[fj * fine_width + (fi-1)] + 
                            fine_residual[fj * fine_width + (fi+1)] +
                            fine_residual[(fj-1) * fine_width + fi] +
                            fine_residual[(fj+1) * fine_width + fi]);
            sum += 0.0625f * (fine_residual[(fj-1) * fine_width + (fi-1)] +
                             fine_residual[(fj-1) * fine_width + (fi+1)] +
                             fine_residual[(fj+1) * fine_width + (fi-1)] +
                             fine_residual[(fj+1) * fine_width + (fi+1)]);
            
            coarse_f[coarse_idx] = sum;
        }
    }
}

__global__ void prolongate_add_kernel(const float* coarse_error, float* fine_u,
                                    int coarse_width, int coarse_height,
                                    int fine_width, int fine_height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < fine_width - 1 && j > 0 && j < fine_height - 1) {
        int ci = i / 2;
        int cj = j / 2;
        
        float correction = 0.0f;
        
        if ((i % 2 == 0) && (j % 2 == 0)) {
            correction = coarse_error[cj * coarse_width + ci];
        } else if ((i % 2 == 1) && (j % 2 == 0)) {
            correction = 0.5f * (coarse_error[cj * coarse_width + ci] + 
                               coarse_error[cj * coarse_width + ci + 1]);
        } else if ((i % 2 == 0) && (j % 2 == 1)) {
            correction = 0.5f * (coarse_error[cj * coarse_width + ci] + 
                               coarse_error[(cj + 1) * coarse_width + ci]);
        } else {
            correction = 0.25f * (coarse_error[cj * coarse_width + ci] + 
                                coarse_error[cj * coarse_width + ci + 1] +
                                coarse_error[(cj + 1) * coarse_width + ci] +
                                coarse_error[(cj + 1) * coarse_width + ci + 1]);
        }
        
        fine_u[j * fine_width + i] += correction;
    }
}

__global__ void set_zero_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

__global__ void copy_boundary_kernel(const float* src, float* dst, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < width) {
        dst[idx] = src[idx];
        dst[(height - 1) * width + idx] = src[(height - 1) * width + idx];
    }
    
    if (idx < height) {
        dst[idx * width] = src[idx * width];
        dst[idx * width + width - 1] = src[idx * width + width - 1];
    }
}

__global__ void compute_norm_kernel(const float* data, float* partial_sums, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < size) ? data[idx] * data[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

SolverMultigridCUDA::SolverMultigridCUDA(float* host_grid, int width, int height, const std::string& name)
    : Solver(host_grid, width, height, name) {
    
    size_t size = width_ * height_ * sizeof(float);
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_u_new, size);
    cudaMemcpy(d_u, host_grid, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, host_grid, size, cudaMemcpyHostToDevice);
    
    createMultigridLevels();
}

SolverMultigridCUDA::~SolverMultigridCUDA() {
    destroyMultigridLevels();
    cudaFree(d_u);
    cudaFree(d_u_new);
}

void SolverMultigridCUDA::createMultigridLevels() {
    int w = width_;
    int h = height_;
    
    m_num_levels = 0;
    while (w > 4 && h > 4 && m_num_levels < 10) {
        m_num_levels++;
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
    
    std::cout << "[" << name_ << "] Creating " << m_num_levels << " multigrid levels\n";
    
    m_levels.resize(m_num_levels);
    
    w = width_;
    h = height_;
    
    for (int l = 0; l < m_num_levels; ++l) {
        Level& level = m_levels[l];
        level.width = w;
        level.height = h;
        
        size_t size = w * h * sizeof(float);
        cudaMalloc(&level.d_u, size);
        cudaMalloc(&level.d_f, size);
        cudaMalloc(&level.d_residual, size);
        cudaMalloc(&level.d_error, size);
        
        cudaMemset(level.d_u, 0, size);
        cudaMemset(level.d_f, 0, size);
        cudaMemset(level.d_residual, 0, size);
        cudaMemset(level.d_error, 0, size);
        
        if (l == 0) {
            cudaMemcpy(level.d_u, d_u, size, cudaMemcpyDeviceToDevice);
        }
        
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

void SolverMultigridCUDA::destroyMultigridLevels() {
    for (auto& level : m_levels) {
        cudaFree(level.d_u);
        cudaFree(level.d_f);
        cudaFree(level.d_residual);
        cudaFree(level.d_error);
    }
}

SolverStatus SolverMultigridCUDA::solve(const SimulationParameters& params) {
    float tolerance = params.tolerance;
    int max_vcycles = std::min(params.max_iterations / 10, 1000);
    
    // Initialize f=0 for Laplace equation
    cudaMemset(m_levels[0].d_f, 0, width_ * height_ * sizeof(float));
    
    for (int cycle = 0; cycle < max_vcycles; ++cycle) {
        vCycle(tolerance, params.max_iterations);
        
        if (cycle % 10 == 0) {
            computeResidual(m_levels[0]);
            float residual_norm = computeL2Norm(m_levels[0].d_residual, m_levels[0].width * m_levels[0].height);
            std::cout << "[" << name_ << "] V-cycle " << cycle << "  residual = " << residual_norm << "\n";
            
            if (residual_norm < tolerance) {
                std::cout << "[" << name_ << "] Converged after " << cycle << " V-cycles\n";
                cudaMemcpy(d_u, m_levels[0].d_u, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
                return SolverStatus{.iterations = cycle * 10, .residual = residual_norm, .converged = true};
            }
        }
    }
    
    cudaMemcpy(d_u, m_levels[0].d_u, width_ * height_ * sizeof(float), cudaMemcpyDeviceToDevice);
    return SolverStatus{.iterations = max_vcycles * 10, .residual = 0.0f, .converged = false};
}

void SolverMultigridCUDA::vCycle(float tolerance, int max_iter) {
    float h = 1.0f / (width_ - 1);
    
    for (int l = 0; l < m_num_levels - 1; ++l) {
        smooth(m_levels[l], 3, 0.6f);
        
        computeResidual(m_levels[l]);
        
        restrict(m_levels[l], m_levels[l + 1]);
        
        dim3 blockSize(256);
        dim3 gridSize((m_levels[l + 1].width * m_levels[l + 1].height + blockSize.x - 1) / blockSize.x);
        set_zero_kernel<<<gridSize, blockSize>>>(m_levels[l + 1].d_u, 
                                                m_levels[l + 1].width * m_levels[l + 1].height);
    }
    
    smooth(m_levels[m_num_levels - 1], 20, 0.6f);
    
    for (int l = m_num_levels - 2; l >= 0; --l) {
        prolongate(m_levels[l + 1], m_levels[l]);
        
        smooth(m_levels[l], 3, 0.6f);
    }
}

void SolverMultigridCUDA::smooth(Level& level, int iterations, float omega) {
    // Adjust h based on current level
    float h = 1.0f / (level.width - 1);
    float h2 = h * h;
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((level.width + blockSize.x - 1) / blockSize.x,
                  (level.height + blockSize.y - 1) / blockSize.y);
    
    // Allocate temporary buffer for this level if needed
    float* d_temp;
    cudaMalloc(&d_temp, level.width * level.height * sizeof(float));
    
    for (int iter = 0; iter < iterations; ++iter) {
        jacobi_smooth_kernel<<<gridSize, blockSize>>>(level.d_u, level.d_f, d_temp,
                                                     level.width, level.height, h2, omega);
        cudaMemcpy(level.d_u, d_temp, level.width * level.height * sizeof(float), 
                   cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_temp);
}

void SolverMultigridCUDA::computeResidual(Level& level) {
    float h = 1.0f / (level.width - 1);
    float h2 = h * h;
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((level.width + blockSize.x - 1) / blockSize.x,
                  (level.height + blockSize.y - 1) / blockSize.y);
    
    compute_residual_kernel<<<gridSize, blockSize>>>(level.d_u, level.d_f, level.d_residual,
                                                    level.width, level.height, h2);
}

void SolverMultigridCUDA::restrict(Level& fine, Level& coarse) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((coarse.width + blockSize.x - 1) / blockSize.x,
                  (coarse.height + blockSize.y - 1) / blockSize.y);
    
    restrict_kernel<<<gridSize, blockSize>>>(fine.d_residual, coarse.d_f,
                                            fine.width, fine.height,
                                            coarse.width, coarse.height);
}

void SolverMultigridCUDA::prolongate(Level& coarse, Level& fine) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((fine.width + blockSize.x - 1) / blockSize.x,
                  (fine.height + blockSize.y - 1) / blockSize.y);
    
    prolongate_add_kernel<<<gridSize, blockSize>>>(coarse.d_u, fine.d_u,
                                                  coarse.width, coarse.height,
                                                  fine.width, fine.height);
}

float SolverMultigridCUDA::computeL2Norm(float* d_data, int size) {
    int num_blocks = (size + 255) / 256;
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(float));
    
    compute_norm_kernel<<<num_blocks, 256, 256 * sizeof(float)>>>(d_data, d_partial_sums, size);
    
    float* h_partial_sums = new float[num_blocks];
    cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < num_blocks; ++i) {
        sum += h_partial_sums[i];
    }
    
    delete[] h_partial_sums;
    cudaFree(d_partial_sums);
    
    return sqrtf(sum / size);
}

float* SolverMultigridCUDA::deviceData() {
    return d_u;
}

const float* SolverMultigridCUDA::deviceData() const {
    return d_u;
}