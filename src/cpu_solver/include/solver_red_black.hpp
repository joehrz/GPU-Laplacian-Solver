// src/cpu_solver/include/solver_red_black.hpp

#pragma once
#include "solver_base.h"
#include "simulation_config.h"

class SolverRedBlack : public Solver {
public:
    SolverRedBlack(float* grid,int w,int h,const std::string& name);
    ~SolverRedBlack() override;

    void solve(const SimulationParameters&) override;

    bool   isOnDevice()   const override { return false; }
    float* deviceData()         override { return nullptr; }
};