#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

const int M = 100;
const int N = 100;
const int MAX_ITER = 10000;
const double OMEGA = 1.85;
const double TOL = 1e-6;

double analyticalSolution(int x, int y) {
    double phi = 0.0;
    double L = N;  // Assume L is the width of the grid
    for (int n = 1; n <= 99; n += 2) {  // Sum over odd n only
        double term = (4 * 100 / (n * M_PI)) * sin(n * M_PI * x / L) * sinh(n * M_PI * (N - y) / L) / sinh(n * M_PI * N / L);
        phi += term;
    }
    return phi;
}

void printSolutions(const vector<vector<double>>& grid) {
    cout << "x y Numerical Analytical Error" << endl;
    for (int i = 0; i < M; i += M / 10) { // Print 10 sample rows
        for (int j = 0; j < N; j += N / 10) { // Print 10 sample columns
            double numerical = grid[i][j];
            double analytical = analyticalSolution(i, j);
            double error = abs(numerical - analytical);
            cout << i << " " << j << " " << numerical << " " << analytical << " " << error << endl;
        }
    }
}

void exportAnalyticalSolution() {
    ofstream outFile("analytical_output.dat");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i;  // x-coordinate in the grid
            double y = j;  // y-coordinate in the grid
            double value = analyticalSolution(x, y);
            outFile << x << " " << y << " " << value << endl;
        }
    }
    outFile.close();
}

void plotWithGnuplot() {
    FILE *gnuplotPipe = popen("gnuplot -persist", "w");
    if (!gnuplotPipe) {
        cerr << "Gnuplot not found." << endl;
        return;
    }
    fprintf(gnuplotPipe, "set xlabel 'X-axis'\n");
    fprintf(gnuplotPipe, "set ylabel 'Y-axis'\n");
    fprintf(gnuplotPipe, "set title 'Analytical Heat Distribution'\n");
    fprintf(gnuplotPipe, "set pm3d map\n");
    fprintf(gnuplotPipe, "set palette defined (0 'blue', 1 'white', 2 'red')\n");
    fprintf(gnuplotPipe, "splot 'analytical_output.dat' using 1:2:3 with image\n");
    fflush(gnuplotPipe);
    pclose(gnuplotPipe);
}

void validateSolution(const vector<vector<double>>& grid) {
    double maxError = 0.0;
    double totalError = 0.0;
    int count = 0;

    for (int i = 1; i < M - 1; ++i) {  // Avoid boundary rows as they are set by conditions
        for (int j = 1; j < N - 1; ++j) {  // Avoid boundary columns
            double exact = analyticalSolution(i, j);
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
        cout << "Maximum error compared to analytical solution: " << maxError << endl;
        cout << "Average error compared to analytical solution: " << avgError << endl;
    } else {
        cout << "No valid points to calculate average error." << endl;
    }
}

void initializeGrid(vector<vector<double>>& grid) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            grid[i][j] = 0.0;  // Set initial guesses
        }
    }
    // Set the top boundary to 100.0
    for (int j = 0; j < N; ++j) {
        grid[0][j] = 100.0;  // Top boundary
    }
}

void updateRedBlackSOR(vector<vector<double>>& grid) {
    double maxError;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        maxError = 0.0;

        // Red update
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1 + (i % 2); j < N - 1; j += 2) { // Start at 1 if i is odd, 2 if i is even (Red nodes)
                double oldVal = grid[i][j];
                double newVal = 0.25 * (grid[i+1][j] + grid[i-1][j] + grid[i][j+1] + grid[i][j-1]);
                grid[i][j] = oldVal + OMEGA * (newVal - oldVal);
                maxError = max(maxError, abs(newVal - oldVal));
            }
        }

        // Black update
        for (int i = 1; i < M - 1; i++) {
            for (int j = 2 - (i % 2); j < N - 1; j += 2) { // Start at 2 if i is odd, 1 if i is even (Black nodes)
                double oldVal = grid[i][j];
                double newVal = 0.25 * (grid[i+1][j] + grid[i-1][j] + grid[i][j+1] + grid[i][j-1]);
                grid[i][j] = oldVal + OMEGA * (newVal - oldVal);
                maxError = max(maxError, abs(newVal - oldVal));
            }
        }

        // Check for convergence
        if (maxError < TOL) {
            cout << "Convergence reached after " << iter << " iterations with max error: " << maxError << endl;
            break;
        }
    }
}

int main() {
    vector<vector<double>> grid(M, vector<double>(N));

    initializeGrid(grid);
    updateRedBlackSOR(grid);
    validateSolution(grid);
    printSolutions(grid);

    // Plotting with gnuplot
    FILE *gnuplotPipe = popen("gnuplot -persist", "w");
    if (!gnuplotPipe) {
        cerr << "Gnuplot not found." << endl;
        return -1;
    }

    fprintf(gnuplotPipe, "set xlabel 'X-axis label'\n");
    fprintf(gnuplotPipe, "set ylabel 'Y-axis label'\n");
    fprintf(gnuplotPipe, "set title 'Heat Distribution'\n");
    fprintf(gnuplotPipe, "set pm3d map\n");
    fprintf(gnuplotPipe, "set palette defined (0 'blue', 1 'white', 2 'red')\n");
    fprintf(gnuplotPipe, "splot '-' matrix with image\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(gnuplotPipe, "%lf ", grid[i][j]);
        }
        fprintf(gnuplotPipe, "\n"); // End of row
    }
    fprintf(gnuplotPipe, "e\n"); // End of data block
    fflush(gnuplotPipe);

    exportAnalyticalSolution();  // Save analytical solution
    plotWithGnuplot();    

    return 0;
}
