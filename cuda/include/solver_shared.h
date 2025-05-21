// include/solver_shared.h

#ifndef SOLVER_SHARED_H
#define SOLVER_SHARED_H

#include "solver_base.h"
#include "simulation_config.h"

// Shared Memory Optimized CUDA-based SOR Red-Black Solver

class SolverShared : public Solver{
public:
    // Constructor
    SolverShared(double *U, int width, int height, const std::string& name);

    // Destructor 
    virtual ~SolverShared();


    // Implementation of the solving algorithm using CUDA shared memory
    void solve(const SimulationParameters& sim_params) override;

    // Implementation of the solution export
    //void exportSolution(const std::string& filename) override;

};

#endif // SOLVER_SHARED_H