// cpu/include/solver_basic.h

#ifndef SOLVER_BASIC_H
#define SOLVER_BASIC_H

#include "solver_base.h"

// Derived class for the basic CPU solver
class SolverStandardSOR : public Solver {
public:
    // Constructor
    SolverStandardSOR(double* grid, int w, int h, const std::string& name);

    // Destructor
    ~SolverStandardSOR();

    // Override the solve method
    void solve() override;

    // Override the exportSolution method
    void exportSolution(const std::string& filename) override;
};

#endif // SOLVER_BASIC_H