// cpu/include/solver_basic.h

#ifndef SOLVER_BASIC_H
#define SOLVER_BASIC_H

#include "solver_base.h"
#include "simulation_config.h"

// Derived class for the basic CPU solver
class SolverStandardSOR : public Solver {
    public:
        SolverStandardSOR(double* grid, int w, int h, const std::string& name);
        ~SolverStandardSOR();

        void solve(const SimulationParameters& sim_params) override;
    };

#endif // SOLVER_BASIC_H