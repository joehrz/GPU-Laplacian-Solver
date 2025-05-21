// include/solver_basic.h

#ifndef SOLVER_BASIC_H
#define SOLVER_BASIC_H

#include "solver_base.h"
#include "simulation_config.h"

// Derived class for the basic GPU solver
class SolverBasic : public Solver {
public:
    // Constructor
    SolverBasic(double* grid, int w, int h, const std::string& name);

    // Destructor
    ~SolverBasic();

    // Override the solve method
    void solve(const SimulationParameters& sim_params) override;

    // Override the exportSolution method
    //void exportSolution(const std::string& filename) override;
};

#endif // SOLVER_BASIC_H
