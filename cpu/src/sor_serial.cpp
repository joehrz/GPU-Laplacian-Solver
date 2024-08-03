#include <cstdio>
#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h> // Include this header for pclose

extern "C" FILE *popen(const char *command, const char *mode);
extern "C" void pclose(FILE *pipe);

using namespace std;

void sorMethod(vector<vector<double>> &phi, const vector<vector<double>> &a,
                const vector<vector<double>> &b, const vector<vector<double>> &c,
                const vector<vector<double>> &d, const vector<vector<double>> &e,
                const vector<vector<double>> &f, double omega, double tol, int maxIter) {

    int M = phi.size();
    int N = phi[0].size();

    vector<vector<double>> r(M, vector<double>(N, 0.0));

    for (int n = 0; n < maxIter; n++) {
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < N - 1; ++j) {
                r[i][j] = f[i][j] - (a[i][j] * phi[i + 1][j] + b[i][j] * phi[i - 1][j] +
                                     c[i][j] * phi[i][j + 1] + d[i][j] * phi[i][j - 1] + e[i][j] * phi[i][j]);
                phi[i][j] += omega * r[i][j] / e[i][j];
            }
        }

        double norm_r = 0.0;
        double norm_f = 0.0;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; ++j) {
                norm_r += r[i][j] * r[i][j];
                norm_f += f[i][j] * f[i][j];
            }
        }

        if (sqrt(norm_r) / sqrt(norm_f) < tol) {
            cout << "Converged after " << n + 1 << " iterations." << endl;
            return;
        }

        // Print residual norm at each step for debugging
        if (n % 1000 == 0) {
            cout << "Iteration " << n << ": Residual norm = " << sqrt(norm_r) << endl;
        }
    }
    cout << "Reached maximum iterations." << endl;
}

double analyticalSolution(int x, int y, int M, int N) {
    double phi = 0.0;
    for (int n = 1; n <= 99; n += 2) {  // Sum over odd n only
        double term = (4 * 100 / (n * M_PI)) * sin(n * M_PI * x / M) * sinh(n * M_PI * y / M) / sinh(n * M_PI * N / M);
        phi += term;
    }
    return phi;
}

void printErrorAnalysis(const vector<vector<double>> &numerical, const vector<vector<double>> &analytical, int M, int N) {
    double maxError = 0.0;
    double totalError = 0.0;
    int count = 0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double error = abs(numerical[i][j] - analytical[i][j]);
            maxError = max(maxError, error);
            totalError += error;
            count++;
        }
    }

    double avgError = totalError / count;
    cout << "Maximum error: " << maxError << endl;
    cout << "Average error: " << avgError << endl;
}

void printSolutions(const vector<vector<double>> &numerical, const vector<vector<double>> &analytical, int M, int N) {
    cout << "x y Numerical Analytical Error" << endl;
    for (int i = 0; i < M; i += M / 10) { // Print 10 sample rows
        for (int j = 0; j < N; j += N / 10) { // Print 10 sample columns
            double numericalVal = numerical[i][j];
            double analyticalVal = analytical[i][j];
            double error = abs(numericalVal - analyticalVal);
            cout << i << " " << j << " " << numericalVal << " " << analyticalVal << " " << error << endl;
        }
    }
}

void printBoundaryValues(const vector<vector<double>> &phi, int M, int N, const char *label) {
    cout << label << " boundary values:" << endl;
    // Print bottom boundary
    cout << "Bottom boundary (y=0): ";
    for (int j = 0; j < N; ++j) {
        cout << phi[0][j] << " ";
    }
    cout << endl;

    // Print top boundary
    cout << "Top boundary (y=N-1): ";
    for (int j = 0; j < N; ++j) {
        cout << phi[M-1][j] << " ";
    }
    cout << endl;

    // Print left boundary
    cout << "Left boundary (x=0): ";
    for (int i = 0; i < M; ++i) {
        cout << phi[i][0] << " ";
    }
    cout << endl;

    // Print right boundary
    cout << "Right boundary (x=M-1): ";
    for (int i = 0; i < M; ++i) {
        cout << phi[i][N-1] << " ";
    }
    cout << endl;
}

void plotSolution(const vector<vector<double>> &solution, int M, int N, const char *title) {
    FILE *gnuplotPipe = popen("gnuplot -persist", "w");
    if (!gnuplotPipe) {
        cerr << "Gnuplot not found." << endl;
        return;
    }

    fprintf(gnuplotPipe, "set title '%s'\n", title);
    fprintf(gnuplotPipe, "set pm3d map\n");
    fprintf(gnuplotPipe, "set palette defined (0 'blue', 1 'white', 2 'red')\n");
    fprintf(gnuplotPipe, "splot '-' matrix with image\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(gnuplotPipe, "%lf ", solution[i][j]);
        }
        fprintf(gnuplotPipe, "\n"); // End of row
    }
    fprintf(gnuplotPipe, "e\n"); // End of data block
    fflush(gnuplotPipe);
    pclose(gnuplotPipe);
}

int main() {
    int M = 100, N = 100;
    double omega = 1.5; // Adjusted relaxation factor
    double tol = 1e-6;
    int maxIter = 50000; // Increased max iterations

    vector<vector<double>> phi(M, vector<double>(N, 0)), f(M, vector<double>(N, 0));
    vector<vector<double>> a(M, vector<double>(N, 1)), b(M, vector<double>(N, 1)),
                           c(M, vector<double>(N, 1)), d(M, vector<double>(N, 1)),
                           e(M, vector<double>(N, -4));
    vector<vector<double>> analytical(M, vector<double>(N, 0));

    // Setting boundary conditions for the numerical solution
    for (int j = 0; j < N; j++) {
        phi[0][j] = 0;     // Bottom boundary
        phi[M-1][j] = 0;   // Top boundary
    }
    for (int i = 0; i < M; i++) {
        phi[i][0] = 0;     // Left boundary
        phi[i][N-1] = 100; // Right boundary
    }

    // Print initial guess and boundary conditions for debugging
    cout << "Initial guess and boundary conditions:" << endl;
    for (int i = 0; i < M; i += M / 10) { // Print 10 sample rows
        for (int j = 0; j < N; j += N / 10) { // Print 10 sample columns
            cout << "phi[" << i << "][" << j << "] = " << phi[i][j] << endl;
        }
    }

    // Calculate the analytical solution
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            analytical[i][j] = analyticalSolution(i, j, M, N);
        }
    }

    // Solve using the SOR method
    sorMethod(phi, a, b, c, d, e, f, omega, tol, maxIter);

    // Print boundary values for numerical and analytical solutions
    printBoundaryValues(phi, M, N, "Numerical");
    printBoundaryValues(analytical, M, N, "Analytical");

    // Compare numerical and analytical solutions
    printErrorAnalysis(phi, analytical, M, N);
    printSolutions(phi, analytical, M, N);

    // Plot numerical solution
    plotSolution(phi, M, N, "Numerical Solution");

    // Plot analytical solution
    plotSolution(analytical, M, N, "Analytical Solution");

    return 0;
}


