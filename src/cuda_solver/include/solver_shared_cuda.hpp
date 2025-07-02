// src/cuda_solver/include/solver_shared_cuda.hpp

#pragma once
#include "solver_base.h"

// This class manages the optimized GPU solver that uses shared memory
class SolverSharedMemCUDA : public Solver {
    float* d_grid_pitched;   // pointer to pitched memory on the GPU
    size_t pitchB;           // pitch (bytes per row, incl. padding)

public:
    SolverSharedMemCUDA(float* host_grid, int w, int h, const std::string& name);
    ~SolverSharedMemCUDA() override;

    void solve(const SimulationParameters& params) override;

    // ── “view” accessors ────────────────────────────────────────────────────
    bool        isOnDevice() const override { return true; }
    float*      deviceData()       override { return d_grid_pitched; }
    size_t      pitchBytes() const { return pitchB; }     
};