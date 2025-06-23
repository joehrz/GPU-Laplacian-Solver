#pragma once
#include "pitch2d.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* =============================================================
   One-colour SOR sweep (0 = red, 1 = black).
   Call the kernel twice per iteration:

       sor_color_kernel<TILE><<<grid,block,smBytes>>>(…, 0, …);
       sor_color_kernel<TILE><<<grid,block,smBytes>>>(…, 1, …);

   TILE may be 0 for the purely global-memory variant, or 32 (etc.)
   for a shared-memory tile of (TILE×TILE) interior points.
   ============================================================ */

template<int TILE>
__global__
void sor_color_kernel(Pitch2D grid,
                      int     W, int H,
                      double  omega,
                      int     colour,
                      float*  residualBlock)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    /* ---------------- Shared-memory path (TILE > 0) ---------------- */
    if constexpr (TILE > 0)
    {
        /* declaration moved here ↓ */
        __shared__ double sm_tile[(TILE + 2) * (TILE + 2)];

        // 1. Load data (centre + halo) into shared memory
        if (i < W && j < H) {
            const int li = threadIdx.x + 1;
            const int lj = threadIdx.y + 1;

            sm_tile[li + lj * (TILE + 2)] = grid.row(j)[i];

            if (threadIdx.x == 0          && i > 0)       sm_tile[                  lj * (TILE + 2)] = grid.row(j)[i - 1];
            if (threadIdx.x == TILE - 1   && i < W - 1)   sm_tile[TILE + 1 + lj * (TILE + 2)]        = grid.row(j)[i + 1];
            if (threadIdx.y == 0          && j > 0)       sm_tile[li]                                = grid.row(j - 1)[i];
            if (threadIdx.y == TILE - 1   && j < H - 1)   sm_tile[li + (TILE + 1) * (TILE + 2)]      = grid.row(j + 1)[i];
        }
        __syncthreads();

        // 2. SOR update (interior points only)
        float accum = 0.0f;
        if (i > 0 && i < W - 1 && j > 0 && j < H - 1 && (((i + j) & 1) == colour))
        {
            const int li = threadIdx.x + 1;
            const int lj = threadIdx.y + 1;

            double centre = sm_tile[li + lj * (TILE + 2)];
            double sigma  = (sm_tile[(li - 1) + lj * (TILE + 2)] +
                             sm_tile[(li + 1) + lj * (TILE + 2)] +
                             sm_tile[li + (lj - 1) * (TILE + 2)] +
                             sm_tile[li + (lj + 1) * (TILE + 2)]) * 0.25;

            const double diff = sigma - centre;
            sm_tile[li + lj * (TILE + 2)] = centre + omega * diff;
            accum = fabsf(static_cast<float>(diff));
        }
        __syncthreads();

        // 3. Write tile back to global memory
        if (i < W && j < H) {
            const int li = threadIdx.x + 1;
            const int lj = threadIdx.y + 1;
            grid.row(j)[i] = sm_tile[li + lj * (TILE + 2)];
        }

        // 4. Per-block residual reduction (unchanged)
        const int lane   = threadIdx.x & 31;
        const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
        for (int ofs = 16; ofs; ofs >>= 1) accum += __shfl_down_sync(0xFFFFFFFF, accum, ofs);
        if (lane == 0) {
            __shared__ float warpSum[1024 / 32];
            warpSum[warpId] = accum;
            __syncthreads();
            if (warpId == 0) {
                float s = 0.0f;
                const int nWarp = (blockDim.x * blockDim.y) / 32;
                for (int w = 0; w < nWarp; ++w) s += warpSum[w];
                if (residualBlock)
                    residualBlock[blockIdx.x + blockIdx.y * gridDim.x] = s;
            }
        }
    }
    /* ------------- Global-memory path (TILE == 0) ------------------ */
    else
    {
        float accum = 0.0f;
        if (i > 0 && i < W - 1 && j > 0 && j < H - 1 && (((i + j) & 1) == colour))
        {
            double centre = grid.row(j)[i];
            double sigma  = (grid.row(j)[i - 1] + grid.row(j)[i + 1] +
                             grid.row(j - 1)[i] + grid.row(j + 1)[i]) * 0.25;

            const double diff = sigma - centre;
            grid.row(j)[i] = centre + omega * diff;
            accum = fabsf(static_cast<float>(diff));
        }

        // residual reduction (unchanged)
        const int lane   = threadIdx.x & 31;
        const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
        for (int ofs = 16; ofs; ofs >>= 1) accum += __shfl_down_sync(0xFFFFFFFF, accum, ofs);
        if (lane == 0) {
            __shared__ float warpSum[1024 / 32];
            warpSum[warpId] = accum;
            __syncthreads();
            if (warpId == 0) {
                float s = 0.0f;
                const int nWarp = (blockDim.x * blockDim.y) / 32;
                for (int w = 0; w < nWarp; ++w) s += warpSum[w];
                if (residualBlock)
                    residualBlock[blockIdx.x + blockIdx.y * gridDim.x] = s;
            }
        }
    }
}