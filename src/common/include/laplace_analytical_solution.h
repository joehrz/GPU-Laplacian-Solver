// src/common/include/laplace_analytical_solution.h

#pragma once
#include "boundary_conditions.h" // include this to use the struct
#include <vector>

class UniversalFourierSolution{
    BoundaryConditions bc; // Store the boundary conditions
    int n_terms;           // Number of terms for the Fourier series

public:
    // The constructor now takes the entire BoundaryConditions struct
    UniversalFourierSolution(const BoundaryConditions& conditions, int terms = 50);

    // The compute function calculates the full solution
    std::vector<float> compute(int W, int H) const;
};
