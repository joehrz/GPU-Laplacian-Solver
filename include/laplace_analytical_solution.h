// include/laplace_analytical_solution.h
#ifndef FOURIER_ANALYTICAL_SOLUTION_H
#define FOURIER_ANALYTICAL_SOLUTION_H

#include "analytical_solution.h"
#include <cmath>

// All calculations now use float
inline float u_left(float x, float y, int n_max) {
    constexpr float PI = 3.1415926535f;
    float sum = 0.0f;
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;
        float term = (4.0f / (n*PI)) * sinf(n*PI*y) * expf(-n*PI*x);
        sum += term;
    }
    return sum;
}

// Similarly for the right boundary
inline float u_right(float x, float y, int n_max) {
    constexpr float PI = 3.14159265358979323846f;
    float sum = 0.0f;
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;
        float term = (4.0f / (n*PI))
                    * std::sin(n*PI*y)
                    // e^{-n pi (1 - x)} = e^{-n pi} * e^{n pi x}
                    * std::exp(-n*PI*(1.0f - x));
        sum += term;
    }
    return sum;
}

// For the bottom boundary
inline float u_bottom(float x, float y, int n_max) {
    constexpr float PI = 3.14159265358979323846f;
    float sum = 0.0f;
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;
        float term = (4.0f / (n*PI))
                    * std::sin(n*PI*x)
                    * std::exp(-n*PI*y);
        sum += term;
    }
    return sum;
}

// For the top boundary
inline float u_top(float x, float y, int n_max) {
    constexpr float PI = 3.14159265358979323846f;
    float sum = 0.0f;
    for(int m=0; m<=n_max; ++m) {
        int n = 2*m + 1;
        float term = (4.0f / (n*PI))
                    * std::sin(n*PI*x)
                    * std::exp(-n*PI*(1.0f - y));
        sum += term;
    }
    return sum;
}


// ----------------------------------------------------
// 2) The build function 
// ----------------------------------------------------
inline std::vector<float> buildLaplaceSolution2D(
    int width, int height,
    float A, float B, float C, float D,
    int n_max
) {
    std::vector<float> U(width*height, 0.0);

    for (int j=0; j<height; ++j){
        float y = float(j) / (height - 1);
        for (int i=0; i<width; ++i){
            float x = float(i) / (width - 1);

            float val = A * u_left(x, y, n_max)
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
    UniversalFourierSolution(float A, float B, float C, float D, int n_max=25)
        : A_(A), B_(B), C_(C), D_(D), n_max_(n_max)
    { }

    // Implementation of the pure virtual function from AnalyticalSolution
    std::vector<float> compute(int width, int height) const override {
        return buildLaplaceSolution2D(width, height, A_, B_, C_, D_, n_max_);
    }

private:
    float A_;
    float B_;
    float C_;
    float D_;
    int    n_max_;
};




#endif // FOURIER_ANALYTICAL_SOLUTION_H
