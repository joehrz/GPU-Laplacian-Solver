// src/cpu_solver/src/solver_base.cpp

#include "solver_base.h"

// Provide the implementation for the constructor
Solver::Solver(float* grid, int w, int h, const std::string& n)
    : name(n), grid_ptr(grid), width(w), height(h) {}

// Provide the implementation for the default isOnDevice function
bool Solver::isOnDevice() const {
    return false;
}

// Provide the implementation for the default deviceData function
float* Solver::deviceData() {
    return nullptr;
}

// Provide the implementation for the getName function
const std::string& Solver::getName() const {
    return name;
}
