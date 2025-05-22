// include/solver_base.h

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include <string>
#include "simulation_config.h"

/* =============================================================
   Abstract base – common to CPU and GPU implementations
   ============================================================= */
class Solver
{
protected:
    double*     U       = nullptr;    // raw pointer (host or device)
    int         width   = 0;
    int         height  = 0;
    std::string name;

public:
    Solver(double* grid, int w, int h, std::string n)
        : U(grid), width(w), height(h), name(std::move(n)) {}

    virtual ~Solver() noexcept = default;

    /* must be provided by every concrete solver ----------------- */
    virtual void solve(const SimulationParameters& p) = 0;

    /* helpers ---------------------------------------------------- */
    double*     data () const { return U;     }
    double*     getDevicePtr()const { return U; }
    std::string getName() const { return name; }
    const char* c_str () const { return name.c_str(); }
    int Nx() const { return width;  }
    int Ny() const { return height; }
};

#endif // SOLVER_BASE_H
