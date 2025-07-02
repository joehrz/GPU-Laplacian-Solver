// src/cuda_solver/include/pitch2d.h

#pragma once
#include <cuda_runtime.h>

/* =============================================================
   Thin helper around a cudaMallocPitch–allocated 2-D buffer.
   Kernels can fetch a row with

       grid.row(j)[i]

   `pitchElems()` returns the pitch in elements (not bytes).
   ============================================================ */

template<typename T>
struct Pitch2D {
    T* ptr      = nullptr;
    size_t pitchB = 0; // Pitch in bytes

    // Gets a pointer to a specific row
    __host__ __device__
    inline T* row(int j) const {
        return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + j * pitchB);
    }

    // Calculates the pitch in terms of the number of elements, not bytes.
    __host__ __device__
    inline int pitchElems() const {
        return static_cast<int>(pitchB / sizeof(T));
    }
};