#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>
#include <fstream>

extern "C" FILE *popen(const char *command, const char *mode);
extern "C" void pclose(FILE *pipe);

using namespace std;

// Constants
const int M = 100;
const int N = 100;
const int MAX_ITER = 10000;
const double OMEGA = 1.85;
const double TOL = 1e-6;

// Analytical solution for comparison
double analyticalSolution(int x, int y, int M, int N) {
    double phi = 0.0;
    for (int n = 1; n <= 99; n += 2) {  // Sum over odd n only
        double term = (4 * 100 / (n * M_PI)) * sin(n * M_PI * x / M) * sinh(n * M_PI * y / M) / sinh(n * M_PI * N / M);
        phi += term;
    }
    return phi;
}

// Validate numerical solution against analytical solution
void validateSolution(const vector<vector<double>>& grid) {
    double maxError = 0.0;
    double totalError = 0.0;
    int count = 0;

    for (int i = 1; i < M - 1; ++i) {  // Avoid boundary rows
        for (int j = 1; j < N - 1; ++j) {  // Avoid boundary columns
            double exact = analyticalSolution(i, j, M, N);
            if (exact != 0) {  // Avoid division by zero
                double error = abs(grid[i][j] - exact);  // Absolute error
                maxError = max(maxError, error);
                totalError += error;
                count++;
            }
        }
    }

    if (count > 0) {
        double avgError = totalError / count;
        cout << "Validation Results:\n";
        cout << "  Maximum error: " << maxError << endl;
        cout << "  Average error: " << avgError << endl;
    } else {
        cout << "No valid points to calculate average error." << endl;
    }
}

// Initialize grid with boundary conditions
void initializeGrid(vector<vector<double>>& grid) {
        // Setting boundary conditions for the numerical solution
    for (int j = 0; j < N; j++) {
        grid[0][j] = 0;     // Bottom boundary
        grid[M-1][j] = 0;   // Top boundary
    }
    for (int i = 0; i < M; i++) {
        grid[i][0] = 0;     // Left boundary
        grid[i][N-1] = 100; // Right boundary
    }

}

// Standard SOR method
void updateStandardSOR(vector<vector<double>>& grid) {
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double maxError = 0.0;

        for (int i = 1; i < M - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double oldVal = grid[i][j];
                double newVal = 0.25 * (grid[i + 1][j] + grid[i - 1][j] + grid[i][j + 1] + grid[i][j - 1]);
                grid[i][j] = oldVal + OMEGA * (newVal - oldVal);
                maxError = max(maxError, abs(newVal - oldVal));
            }
        }

        if (maxError < TOL) {
            cout << "Standard SOR converged after " << iter + 1 << " iterations.\n";
            return;
        }
    }
    cout << "Standard SOR reached the maximum iteration limit.\n";
}

// Red-Black SOR method
void updateRedBlackSOR(vector<vector<double>>& grid) {
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double maxError = 0.0;

        // Red update
        for (int i = 1; i < M - 1; ++i) {
            for (int j = 1 + (i % 2); j < N - 1; j += 2) { // Red nodes
                double oldVal = grid[i][j];
                double newVal = 0.25 * (grid[i + 1][j] + grid[i - 1][j] + grid[i][j + 1] + grid[i][j - 1]);
                grid[i][j] = oldVal + OMEGA * (newVal - oldVal);
                maxError = max(maxError, abs(newVal - oldVal));
            }
        }

        // Black update
        for (int i = 1; i < M - 1; ++i) {
            for (int j = 2 - (i % 2); j < N - 1; j += 2) { // Black nodes
                double oldVal = grid[i][j];
                double newVal = 0.25 * (grid[i + 1][j] + grid[i - 1][j] + grid[i][j + 1] + grid[i][j - 1]);
                grid[i][j] = oldVal + OMEGA * (newVal - oldVal);
                maxError = max(maxError, abs(newVal - oldVal));
            }
        }

        if (maxError < TOL) {
            cout << "Red-Black SOR converged after " << iter + 1 << " iterations.\n";
            return;
        }
    }
    cout << "Red-Black SOR reached the maximum iteration limit.\n";
}

// Measure the execution time of the SOR methods
double timeSOR(void (*sorMethod)(vector<vector<double>>&), vector<vector<double>>& grid) {
    auto start = chrono::high_resolution_clock::now();
    sorMethod(grid);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    return duration.count();
}

// Export solution to a file for plotting
void exportSolution(const vector<vector<double>>& grid, const string& filename) {
    ofstream file(filename);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            file << i << " " << j << " " << grid[i][j] << endl;
        }
        file << endl; // Blank line to separate rows
    }
    file.close();
}

// Plot the solutions using gnuplot
void plotSolution(const std::vector<std::vector<double>> &solution, int M, int N, const char *title) {
    FILE *gnuplotPipe = popen("gnuplot -persist", "w");
    if (!gnuplotPipe) {
        std::cerr << "Gnuplot not found." << std::endl;
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




