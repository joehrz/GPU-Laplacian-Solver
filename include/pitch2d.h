#pragma once
#include <cuda_runtime.h>

/* =============================================================
   Thin helper around a cudaMallocPitch–allocated 2-D buffer.
   Kernels can fetch a row with

       grid.row(j)[i]

   `pitchElems()` returns the pitch in elements (not bytes).
   ============================================================ */
template<typename T>
struct Pitch2D
{
    T* ptr    = nullptr;
    size_t pitchB = 0;

    __host__ __device__
    inline T* row(int j) const {
        return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + j * pitchB);
    }
    __host__ __device__
    inline int pitchElems() const {
        return static_cast<int>(pitchB / sizeof(T));
    }
};
