// src/cpu_solver/include/solver_standard_sor.hpp

#pragma once
#include "solver_base.h"
#include "simulation_config.h"

class SolverStandardSOR : public Solver {
public:
    SolverStandardSOR(float* grid,int w,int h,const std::string& name);
    ~SolverStandardSOR() override;

    void solve(const SimulationParameters&) override;

    bool   isOnDevice()   const override { return false; }
    float* deviceData()         override { return nullptr; }
};