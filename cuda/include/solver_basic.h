// include/solver_basic.h

#ifndef SOLVER_BASIC_H
#define SOLVER_BASIC_H

#include "solver_base.h"

/* =============================================================
   GPU baseline SOR (global-memory only)
   ============================================================= */
class SolverBasic : public Solver
{
public:
    SolverBasic(double* d_grid, int w, int h, const std::string& n)
        : Solver(d_grid, w, h, n) {}

    ~SolverBasic() override = default;

    void solve(const SimulationParameters& p) override;
};

#endif // SOLVER_BASIC_H
