// cpu/src/red_black_sor.cpp

// -------------------------------------------------------------
//   Red–Black SOR (checker-board update so rows are independent)
// -------------------------------------------------------------

#include "solver_red_black.h"
#include <algorithm>
#include <cmath>
#include <iostream>

/*--------- ctor / dtor ----------------------------------------*/
SolverRedBlack::SolverRedBlack(float* grid,
                               int     w,
                               int     h,
                               const std::string& n)
    : Solver(grid, w, h, n)
{}

SolverRedBlack::~SolverRedBlack() = default;

/*--------- main iteration loop --------------------------------*/
void SolverRedBlack::solve(const SimulationParameters& prm)
{
    const int    itMax = prm.max_iterations;
    const float tol   = prm.tolerance;
    const float omega = prm.omega;

    for (int iter = 0; iter < itMax; ++iter)
    {
        float maxErr = 0.0;

        /* two-colour sweep: colour = 0 (red) then 1 (black) */
        for (int colour = 0; colour < 2; ++colour)
        {
            for (int j = 1; j < height - 1; ++j)
                for (int i = 1 + ((j + colour) & 1); i < width - 1; i += 2)
                {
                    const int idx   = i + j * width;
                    const float old = U[idx];

                    const float sigma =
                          ( U[idx - 1]     + U[idx + 1]
                          + U[idx - width] + U[idx + width] ) * 0.25f;

                    const float diff = sigma - old;
                    U[idx]  += omega * diff;
                    maxErr   = std::max(maxErr, std::abs(diff));
                }
        }

        if (maxErr < tol) {
            std::cout << '[' << name << "] Red-Black SOR converged in "
                      << iter + 1 << " iters (res=" << maxErr << ")\n";
            return;
        }
    }
    std::cout << '[' << name << "] Red-Black SOR hit the max-iteration limit ("
              << itMax << ")\n";
}