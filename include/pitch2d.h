#pragma once
#include <cuda_runtime.h>

/* =============================================================
   Thin helper around a cudaMallocPitch–allocated 2-D buffer.
   Kernels can fetch a row with

       grid.row(j)[i]

   `pitchElems()` returns the pitch in elements (not bytes).
   ============================================================ */
struct Pitch2D
{
    double* ptr    = nullptr;
    size_t  pitchB = 0;            /* byte pitch returned by cudaMallocPitch */

    __host__ __device__
    inline double* row(int j) const
    {
        return reinterpret_cast<double*>(
               reinterpret_cast<char*>(ptr) + j * pitchB);
    }

    __host__ __device__
    inline int pitchElems() const   /* pitch / sizeof(double) */
    {
        return static_cast<int>(pitchB / sizeof(double));
    }
};
