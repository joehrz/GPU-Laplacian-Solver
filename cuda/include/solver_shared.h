// include/solver_shared.h

#ifndef SOLVER_SHARED_H
#define SOLVER_SHARED_H

#include "solver_base.h"
#include "simulation_config.h"

/* =============================================================
   GPU SOR with a shared–memory tile (TILE = 32)
   ============================================================= */
class SolverShared : public Solver
{
public:
    SolverShared(double* in_dGrid, int w, int h, const std::string& n);
    ~SolverShared() override;

    void solve(const SimulationParameters& p) override;

    /* --- small accessors so main() can copy data back -------- */
    double* data()       const { return U; }          // overrides base
    int     pitchElems() const { return pitchElems_; }

private:
    int pitchElems_{0};          /* logical pitch in elements */
};
#endif /* SOLVER_SHARED_H */