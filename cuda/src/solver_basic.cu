// src/solver_basic.cu

#include "solver_basic.h"
#include "pitch2d.h"
#include "sor_color_kernel.cuh"        // ← NEW (replaces sor_fused_kernel.cuh)
#include "utilities.h"

#include <cuda_runtime.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <iostream>

/* ============================================================= */
void SolverBasic::solve(const SimulationParameters& prm)
{
    /* 1. make a pitched copy of the input grid -------------------- */
    Pitch2D d_grid;
    CUDA_CHECK_ERROR(cudaMallocPitch(&d_grid.ptr, &d_grid.pitchB,
                                     width * sizeof(double), height));
    CUDA_CHECK_ERROR(cudaMemcpy2D(d_grid.ptr, d_grid.pitchB,
                                  U, width * sizeof(double),
                                  width * sizeof(double), height,
                                  cudaMemcpyDeviceToDevice));

    /* 2. residual scratch space (one float per block) ------------- */
    const dim3 block(32, 8);                       // 256 threads / block
    const dim3 grid((width  + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    thrust::device_vector<float> d_block(grid.x * grid.y, 0.0f);

    /* 3. iteration loop ------------------------------------------ */
    const double tol   = prm.tolerance;
    const int    itMax = prm.max_iterations;
    const double omega = prm.omega;

    float residual = std::numeric_limits<float>::infinity();
    int   iter     = 0;

    while (residual > tol && iter < itMax)
    {
        thrust::fill(d_block.begin(), d_block.end(), 0.0f);

        /* ---- red sweep ---- */
        sor_color_kernel<0><<<grid, block>>>(
            d_grid, width, height, omega, 0,      /* colour = red */
            thrust::raw_pointer_cast(d_block.data()));

        /* ---- black sweep ---- */
        sor_color_kernel<0><<<grid, block>>>(
            d_grid, width, height, omega, 1,      /* colour = black */
            thrust::raw_pointer_cast(d_block.data()));

        CUDA_CHECK_ERROR(cudaGetLastError());

        residual = thrust::reduce(d_block.begin(), d_block.end(), 0.0f) /
                   static_cast<float>(width * height);

        if ((iter & 255) == 0)
            std::cout << '[' << name << "] iter " << iter
                      << "  residual = " << residual << '\n';
        ++iter;
    }

    std::cout << '[' << name << "] "
              << ((residual <= tol) ? "converged" : "max iterations reached")
              << " in " << iter << " iterations  (residual = "
              << residual << ")\n";

    /* 4. copy result back & free --------------------------------- */
    CUDA_CHECK_ERROR(cudaMemcpy2D(U, width * sizeof(double),
                                  d_grid.ptr, d_grid.pitchB,
                                  width * sizeof(double), height,
                                  cudaMemcpyDeviceToDevice));
    cudaFree(d_grid.ptr);
}