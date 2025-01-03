// include/fourier_analytical_solution.h

#ifndef FOURIER_ANALYTICAL_SOLUTION_H
#define FOURIER_ANALYTICAL_SOLUTION_H

#include "analytical_solution.h"
#include <cmath>


// Helper: compute partial sum of the "heated-left" solution u_left
//   u_left(x,y) = sum_{m=0..n_max} [ 4/((2m+1) pi) ] * sin((2m+1) pi y) * exp(-(2m+1) pi x)
inline double u_left(double x, double y, int n_max) {
    static const double PI = 3.14159265358979323846;
    double sum = 0.0;
    // We'll only do odd n => (2m+1)
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;  // odd
        double term = (4.0 / (n*PI))
                    * std::sin(n*PI*y)
                    * std::exp(-n*PI*x);
        sum += term;
    }
    return sum;
}

// Similarly for the right boundary
inline double u_right(double x, double y, int n_max) {
    static const double PI = 3.14159265358979323846;
    double sum = 0.0;
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;
        double term = (4.0 / (n*PI))
                    * std::sin(n*PI*y)
                    // e^{-n pi (1 - x)} = e^{-n pi} * e^{n pi x}
                    * std::exp(-n*PI*(1.0 - x));
        sum += term;
    }
    return sum;
}

// For the bottom boundary
inline double u_bottom(double x, double y, int n_max) {
    static const double PI = 3.14159265358979323846;
    double sum = 0.0;
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;
        double term = (4.0 / (n*PI))
                    * std::sin(n*PI*x)
                    * std::exp(-n*PI*y);
        sum += term;
    }
    return sum;
}

// For the top boundary
inline double u_top(double x, double y, int n_max) {
    static const double PI = 3.14159265358979323846;
    double sum = 0.0;
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;
        double term = (4.0 / (n*PI))
                    * std::sin(n*PI*x)
                    * std::exp(-n*PI*(1.0 - y));
        sum += term;
    }
    return sum;
}


// ----------------------------------------------------
// 2) The build function 
// ----------------------------------------------------
inline std::vector<double> buildLaplaceSolution2D(
    int width, int height,
    double A, double B, double C, double D,
    int n_max
) {
    std::vector<double> U(width*height, 0.0);

    for (int j=0; j<height; ++j){
        double y = double(j) / (height - 1);
        for (int i=0; i<width; ++i){
            double x = double(i) / (width - 1);

            double val = A * u_left(x, y, n_max)
                       + B * u_right(x, y, n_max)
                       + C * u_top(x, y, n_max)
                       + D * u_bottom(x, y, n_max);

            U[j*width + i] = val;
        }
    }
    return U;
}

// ------------------------------------------------------------
// 3) The actual class that uses these partial solutions
// ------------------------------------------------------------
class UniversalFourierSolution : public AnalyticalSolution {
public:
    UniversalFourierSolution(double A, double B, double C, double D, int n_max=25)
        : A_(A), B_(B), C_(C), D_(D), n_max_(n_max)
    { }

    // Implementation of the pure virtual function from AnalyticalSolution
    std::vector<double> compute(int width, int height) const override {
        return buildLaplaceSolution2D(width, height, A_, B_, C_, D_, n_max_);
    }

private:
    double A_;
    double B_;
    double C_;
    double D_;
    int    n_max_;
};










#endif // FOURIER_ANALYTICAL_SOLUTION_H
