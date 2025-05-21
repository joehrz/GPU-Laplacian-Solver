// cpu/include/solver_red_black.h

#ifndef SOLVER_RED_BLACK_H
#define SOLVER_RED_BLACK_H


#include "solver_base.h"
#include "simulation_config.h"

class SolverRedBlack : public Solver {
public:
    SolverRedBlack(double* grid, int w, int h, const std::string& name);
    ~SolverRedBlack();
    // MODIFIED SIGNATURE:
    void solve(const SimulationParameters& sim_params) override;
};

#endif // SOLVER_RED_BLACK_H