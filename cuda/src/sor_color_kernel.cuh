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
                      int     colour,            // 0 = red, 1 = black
                      float*  residualBlock)      // per-block L1 residual
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    /* -------- shared-memory tile --------------------------- */
    constexpr int SHARED_SZ = (TILE ? (TILE + 2) * (TILE + 2) : 1);
    __shared__ double sm[SHARED_SZ];

    double centre = 0.0;
    if (i < W && j < H) {
        centre = grid.row(j)[i];

        if constexpr (TILE) {
            const int li = threadIdx.x + 1;
            const int lj = threadIdx.y + 1;
            sm[li + lj * (TILE + 2)] = centre;

            /* halo loads */
            if (threadIdx.x == 0      && i > 0   )
                sm[              lj * (TILE + 2)]     = grid.row(j)[i - 1];
            if (threadIdx.x == TILE-1 && i < W-1 )
                sm[TILE + 1 + lj * (TILE + 2) ]       = grid.row(j)[i + 1];
            if (threadIdx.y == 0      && j > 0   )
                sm[li                            ]     = grid.row(j - 1)[i];
            if (threadIdx.y == TILE-1 && j < H-1 )
                sm[li + (TILE + 1) * (TILE + 2)]       = grid.row(j + 1)[i];
        }
    }
    if constexpr (TILE) __syncthreads();

    /* helper to fetch neighbours (uses shared mem when possible) */
    auto LOAD = [&](int offx, int offy) -> double {
        if constexpr (TILE) {
            const int li = threadIdx.x + 1 + offx;
            const int lj = threadIdx.y + 1 + offy;
            if (li >= 1 && li <= TILE && lj >= 1 && lj <= TILE)
                return sm[li + lj * (TILE + 2)];
        }
        return grid.row(j + offy)[i + offx];
    };

    /* -------- SOR update ----------------------------------- */
    float accum = 0.0f;
    if (i > 0 && i < W - 1 && j > 0 && j < H - 1 &&
        (((i + j) & 1) == colour))
    {
        const double sigma =
            ( LOAD(-1,0) + LOAD(1,0) + LOAD(0,-1) + LOAD(0,1) ) * 0.25;

        const double diff = sigma - centre;
        centre += omega * diff;
        accum   = fabsf(static_cast<float>(diff));

        if constexpr (TILE)
            sm[(threadIdx.x + 1) + (threadIdx.y + 1) * (TILE + 2)] = centre;
    }
    if constexpr (TILE) __syncthreads();
    if (i < W && j < H) grid.row(j)[i] = centre;

    /* -------- per-block L1 residual reduce ------------------ */
    const int lane   = threadIdx.x & 31;
    const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;

    for (int ofs = 16; ofs; ofs >>= 1)
        accum += __shfl_down_sync(0xFFFFFFFF, accum, ofs);

    __shared__ float warpSum[1024 / 32];            // up to 32×32 threads
    if (lane == 0) warpSum[warpId] = accum;
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float s = 0.0f;
        const int nWarp = blockDim.x * blockDim.y / 32;
        #pragma unroll
        for (int w = 0; w < nWarp; ++w) s += warpSum[w];

        residualBlock[blockIdx.x + blockIdx.y * gridDim.x] = s;
    }
}