// src/common/src/laplace_analytical_solution.cpp

#include "laplace_analytical_solution.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The constructor takes the BoundaryConditions struct
UniversalFourierSolution::UniversalFourierSolution(const BoundaryConditions& conditions, int terms)
    : bc(conditions), n_terms(terms) {}

// The compute method implements the full superposition principle
std::vector<float> UniversalFourierSolution::compute(int W, int H) const {
    std::vector<float> grid(W * H, 0.0f);
    double Lx = 1.0; // Assume a unit domain width
    double Ly = 1.0; // Assume a unit domain height

    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            double x = static_cast<double>(i) / (W - 1);
            double y = static_cast<double>(j) / (H - 1);
            double val = 0.0;

            // Sum the contributions from each boundary
            for (int n = 1; n <= n_terms; ++n) {
                // Contribution from Bottom Boundary (T_B)
                if (std::abs(bc.bottom) > 1e-9) {
                    double lambda_n = n * M_PI / Lx;
                    double C_n = (2.0 * bc.bottom / (n * M_PI)) * (1.0 - std::cos(n * M_PI));
                    val += C_n * (std::sinh(lambda_n * y) / std::sinh(lambda_n * Ly)) * std::sin(lambda_n * x);
                }

                // Contribution from Top Boundary (T_T)
                if (std::abs(bc.top) > 1e-9) {
                    double lambda_n = n * M_PI / Lx;
                    double C_n = (2.0 * bc.top / (n * M_PI)) * (1.0 - std::cos(n * M_PI));
                    val += C_n * (std::sinh(lambda_n * (Ly - y)) / std::sinh(lambda_n * Ly)) * std::sin(lambda_n * x);
                }

                // Contribution from Left Boundary (T_L)
                if (std::abs(bc.left) > 1e-9) {
                    double mu_n = n * M_PI / Ly;
                    double D_n = (2.0 * bc.left / (n * M_PI)) * (1.0 - std::cos(n * M_PI));
                    val += D_n * (std::sinh(mu_n * (Lx - x)) / std::sinh(mu_n * Lx)) * std::sin(mu_n * y);
                }

                // Contribution from Right Boundary (T_R)
                if (std::abs(bc.right) > 1e-9) {
                    double mu_n = n * M_PI / Ly;
                    double D_n = (2.0 * bc.right / (n * M_PI)) * (1.0 - std::cos(n * M_PI));
                    val += D_n * (std::sinh(mu_n * x) / std::sinh(mu_n * Lx)) * std::sin(mu_n * y);
                }
            }
            grid[j * W + i] = static_cast<float>(val);
        }
    }
    return grid;
}