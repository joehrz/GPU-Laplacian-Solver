#pragma once

#include <cuda_runtime.h>

__global__ void jacobi_smooth_kernel(float* u, const float* f, float* u_new, 
                                   int width, int height, float h2, float omega);

__global__ void compute_residual_kernel(const float* u, const float* f, float* residual,
                                      int width, int height, float h2);

__global__ void restrict_kernel(const float* fine_residual, float* coarse_f,
                              int fine_width, int fine_height,
                              int coarse_width, int coarse_height);

__global__ void prolongate_add_kernel(const float* coarse_error, float* fine_u,
                                    int coarse_width, int coarse_height,
                                    int fine_width, int fine_height);

__global__ void set_zero_kernel(float* data, int size);

__global__ void copy_boundary_kernel(const float* src, float* dst, int width, int height);

__global__ void compute_norm_kernel(const float* data, float* partial_sums, int size);