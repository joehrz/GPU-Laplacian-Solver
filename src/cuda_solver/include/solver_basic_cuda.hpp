// src/cuda_solver/include/solver_basic_cuda.hpp

#pragma once
#include "solver_base.h"

// This class manages the baseline GPU solver that uses global memory only.
class SolverBasicCUDA : public Solver {
    float* d_grid; // Pointer to the grid data on the GPU device


public:
    SolverBasicCUDA(float* host_grid, int w, int h, const std::string& name);
    ~SolverBasicCUDA() override;

    void solve(const SimulationParameters& params) override;

    // Override virtual functions to report that data is on the device
    bool   isOnDevice()   const override { return true; }
    float* deviceData()         override { return d_grid; }
};
