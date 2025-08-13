// src/cuda_solver/include/solver_mixed_bc_cuda.hpp

#pragma once
#include "solver_base.h"
#include "boundary_conditions.h"
#include <cuda_runtime.h>

// CUDA solver supporting mixed Dirichlet/Neumann boundary conditions
class SolverMixedBCCUDA : public Solver {
public:
    SolverMixedBCCUDA(float* host_grid, int w, int h, const std::string& name = "MixedBCCUDA");
    ~SolverMixedBCCUDA();
    
    SolverStatus solve(const SimulationParameters& prm) override;
    void setBoundaryConditions(const BoundaryConditions& bc);
    
    // Override virtual functions to report that data is on the device
    bool isOnDevice() const override { return true; }
    float* deviceData() override { return d_grid; }
    const float* deviceData() const override { return d_grid; }
    
private:
    float* d_grid;
    BoundaryConditions boundary_conditions;
    float grid_spacing;
};