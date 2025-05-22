// cpu/src/solver_basic.cpp

// -------------------------------------------------------------
//   Standard SOR (Gauss–Seidel with over-relaxation)
// -------------------------------------------------------------

#include "solver_basic.h"          // declares SolverStandardSOR
#include <algorithm>               // std::max
#include <cmath>                   // std::abs
#include <iostream>                // std::cout

/*--------- ctor / dtor matching the declarations --------------*/
SolverStandardSOR::SolverStandardSOR(double* grid,
                                     int     w,
                                     int     h,
                                     const std::string& n)
    : Solver(grid, w, h, n)        // delegate to base-class ctor
{}

SolverStandardSOR::~SolverStandardSOR() = default;

/*--------- main iteration loop --------------------------------*/
void SolverStandardSOR::solve(const SimulationParameters& prm)
{
    const int    itMax = prm.max_iterations;
    const double tol   = prm.tolerance;
    const double omega = prm.omega;

    for (int iter = 0; iter < itMax; ++iter)
    {
        double maxErr = 0.0;

        for (int j = 1; j < height - 1; ++j)
            for (int i = 1; i < width - 1; ++i)
            {
                const int idx   = i + j * width;
                const double old = U[idx];

                const double sigma =
                      ( U[idx - 1]     + U[idx + 1]
                      + U[idx - width] + U[idx + width] ) * 0.25;

                const double diff = sigma - old;
                U[idx]  += omega * diff;                 // SOR update
                maxErr   = std::max(maxErr, std::abs(diff));
            }

        if (maxErr < tol) {
            std::cout << '[' << name << "] Standard SOR converged in "
                      << iter + 1 << " iters (res=" << maxErr << ")\n";
            return;
        }
    }
    std::cout << '[' << name << "] Standard SOR hit the max-iteration limit ("
              << itMax << ")\n";
}

