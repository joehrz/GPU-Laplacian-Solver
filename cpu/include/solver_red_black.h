// cpu/include/solver_red_black.h

#ifndef SOLVER_RED_BLACK_H
#define SOLVER_RED_BLACK_H


#include "solver_base.h"

// Derived class for the Red-Black SOR CPU solver
class SolverRedBlack : public Solver {
public:
    // Constructor
    SolverRedBlack(double* grid, int w, int h, const std::string& name);

    // Destructor
    ~SolverRedBlack();

    // Override the solve method
    void solve() override;

    // Override the exportSolution method
    void exportSolution(const std::string& filename) override;
};

#endif // SOLVER_RED_BLACK_H