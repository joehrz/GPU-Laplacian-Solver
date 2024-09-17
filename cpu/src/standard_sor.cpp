#include "sor_methods.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

// Standard SOR method implementation
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
