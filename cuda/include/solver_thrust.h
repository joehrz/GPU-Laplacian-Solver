// include/solver_thrust.h

#ifndef SOLVER_THRUST_H
#define SOLVER_THRUST_H

#include "solver_base.h"

// Thrust-Optimized CUDA-based SOR Red-Black Solver
class SolverThrust : public Solver{
public:
    // Constructor
    SolverThrust(double *U, int width, int height, const std::string& name);

    // Destructor
    virtual ~SolverThrust();

    // Implementation of the solving algorithm using the Thrust library
    void solve() override;

    // Implementation of the solution export
    void exportSolution(const std::string& filename) override;
};

#endif // SOLVER_THRUST_H
