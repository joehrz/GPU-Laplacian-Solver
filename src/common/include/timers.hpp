// src/common/include/timers.hpp
#pragma once
#include <chrono>

#ifdef SOLVER_ENABLE_CUDA
    #include <cuda_runtime.h>
    // A helper macro for checking CUDA errors.
    #ifndef CUDA_CHECK_ERROR
    #define CUDA_CHECK_ERROR(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
    #endif
#endif

// --- WallTimer Declaration ---
class WallTimer {
    std::chrono::steady_clock::time_point t0;
public:
    WallTimer(); 
    void   start();
    double stop();    
};


// --- CudaEventTimer Declaration (if CUDA is enabled) ---
#ifdef SOLVER_ENABLE_CUDA
class CudaEventTimer {
    cudaEvent_t start_, stop_;
public:
    CudaEventTimer();
    ~CudaEventTimer();
    void   start();
    double stop(); // Returns seconds
};
#endif

// --- DefaultTimer Selection ---
#ifdef SOLVER_ENABLE_CUDA
using DefaultTimer = CudaEventTimer;
#else
using DefaultTimer = WallTimer;
#endif