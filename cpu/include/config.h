// cpu/include/config.h

#ifndef CONFIG_H
#define CONFIG_H

// Maximum number of iterations for SOR methods
constexpr int MAX_ITER = 10000;

// Relaxation factor for SOR methods
constexpr double OMEGA = 1.5;

// Tolerance for convergence
constexpr double TOL = 1e-6;

#endif // CONFIG_H