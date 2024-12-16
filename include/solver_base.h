// include/solver_base.h

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include <string>

// Abstract base class for solvers
class Solver {
protected:
    double* U;               // Pointer to the grid
    int width;               // Grid width
    int height;              // Grid height
    std::string solverName;  // Name identifier for the solver

public:
    // Constructor
    Solver(double* grid, int w, int h, const std::string& name)
        : U(grid), width(w), height(h), solverName(name) {}

    // Virtual destructor
    virtual ~Solver() {}

    // Pure virtual function to perform solving
    virtual void solve() = 0;

    // Pure virtual function to export solution to a CSV file
    virtual void exportSolution(const std::string& filename) = 0;

    // Getter for solver name
    std::string getName() const { return solverName; }
};

#endif // SOLVER_BASE_H
