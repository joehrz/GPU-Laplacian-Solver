// src/solver_shared.cu
#include "solver_shared.h"
#include "pitch2d.h"
#include "sor_color_kernel.cuh"          // NEW  (replaces sor_fused_kernel.cuh)
#include "utilities.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <iostream>

#define TILE 32        /* set 0 for no shared mem; 32 is a good default */

/* ------------------------------------------------------------ */
SolverShared::SolverShared(float* d_in, int w, int h, const std::string& n)
            : Solver(nullptr, w, h, n)
{
    /* pitched allocation for coalesced loads */
    size_t pitchB = 0;
    CUDA_CHECK_ERROR(cudaMallocPitch(&U, &pitchB, w * sizeof(float), h));
    pitchElems_ = static_cast<int>(pitchB / sizeof(float));

    if (d_in) {
        CUDA_CHECK_ERROR(
            cudaMemcpy2D(U,       pitchB,
                         d_in,    w * sizeof(float),
                         w * sizeof(float), h,
                         cudaMemcpyDeviceToDevice));
    }
}

SolverShared::~SolverShared() { cudaFree(U); }

/* ------------------------------------------------------------ */
void SolverShared::solve(const SimulationParameters& p)
{
    const dim3 block(TILE, TILE);
    const dim3 grid((width  + TILE - 1) / TILE,
                    (height + TILE - 1) / TILE);

    thrust::device_vector<float> d_block(grid.x * grid.y, 0.0f);

    Pitch2D<float> view{ U, size_t(pitchElems_ * sizeof(float)) };
    const size_t smBytes = (TILE + 2) * (TILE + 2) * sizeof(float);

    for (int iter = 0; iter < p.max_iterations; ++iter)
    {
        thrust::fill(d_block.begin(), d_block.end(), 0.0f);

        /* ---- red sweep ---- */
        sor_color_kernel<TILE><<<grid, block, smBytes>>>(
            view, width, height, p.omega, 0,
            thrust::raw_pointer_cast(d_block.data()));

        /* ---- black sweep ---- */
        sor_color_kernel<TILE><<<grid, block, smBytes>>>(
            view, width, height, p.omega, 1,
            thrust::raw_pointer_cast(d_block.data()));

        CUDA_CHECK_ERROR(cudaGetLastError());

        float residual =
            thrust::reduce(d_block.begin(), d_block.end(), 0.0) /
            (width * height);

        if (iter % 100 == 0)
            std::cout << '[' << name << "] iter " << iter
                      << "  residual = " << residual << '\n';

        if (residual < p.tolerance) {
            std::cout << '[' << name << "] converged in "
                      << iter + 1 << " iterations  (residual = "
                      << residual << ")\n";
            return;
        }
    }
    std::cout << '[' << name << "] max iterations reached.\n";
}