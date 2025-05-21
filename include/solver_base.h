// include/solver_base.h

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include <string>
#include "simulation_config.h"

// Abstract base class for solvers
class Solver {
protected:
    double* U;
    int width;
    int height;
    std::string solverName;

public:
    Solver(double* grid, int w, int h, const std::string& name)
      : U(grid), width(w), height(h), solverName(name) {}

    virtual ~Solver() {}

    // Remains pure virtual for solving
    virtual void solve(const SimulationParameters& sim_params) = 0;

    // no exportSolution(...) here anymore

    // If you want to retrieve the device pointer from outside:
    double* getDevicePtr() const { return U; }

    // Retrieve solver name
    std::string getName() const { return solverName; }
};

#endif // SOLVER_BASE_H
