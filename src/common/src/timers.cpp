// src/common/src/timers.cpp

#include "timers.hpp"
#include <iostream>

// --- WallTimer Implementation ---
WallTimer::WallTimer() : t0(std::chrono::steady_clock::now()) {}

void WallTimer::start() {
    t0 = std::chrono::steady_clock::now();
}

double WallTimer::stop() {
    using d = std::chrono::duration<double>;
    return std::chrono::duration_cast<d>(std::chrono::steady_clock::now() - t0).count();
}

// --- CudaEventTimer Implementation (only if CUDA is enabled) ---
#ifdef SOLVER_ENABLE_CUDA

#ifndef CUDA_CHECK_ERROR
#define CUDA_CHECK_ERROR(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

CudaEventTimer::CudaEventTimer(){
    CUDA_CHECK_ERROR(cudaEventCreate(&start_));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop_));
}

CudaEventTimer::~CudaEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void CudaEventTimer::start(){
    CUDA_CHECK_ERROR(cudaEventRecord(start_));
}

// Returns time in seconds
double CudaEventTimer::stop() {
    float ms = 0.f;
    CUDA_CHECK_ERROR(cudaEventRecord(stop_));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop_));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&ms, start_, stop_));
    return static_cast<double>(ms) / 1000.0;
}

#endif // SOLVER_ENABLE_CUDA