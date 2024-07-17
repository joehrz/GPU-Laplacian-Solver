#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define M 100
#define N 100
#define maxIter 10000
#define OMEGA 1.85
#define TOL 1e-6

void initializeGrid(vector<vector<double>>& grid) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == 0 || j == 0 || i == M-1 || j == N-1) {
                grid[i][j] = 0.0;  // Boundary conditions
            } else {
                grid[i][j] = 0.5;  // Initial guess for interior points
            }
        }
    }
}


void updateRedBlack(vector<vector<double>>& grid) {
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double maxError = 0.0;

        for (int color = 0; color < 2; color++) { // 0 for Red, 1 for Black
            for (int i = 1; i < M - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    if ((i + j) % 2 == color) {
                        double oldPhi = grid[i][j];
                        double newPhi = 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
                        grid[i][j] = oldPhi + OMEGA * (newPhi - oldPhi);
                        maxError = max(maxError, abs(newPhi - oldPhi));
                    }
                }
            }
        }

        if (maxError < TOL) {
            cout << "Convergence reached after " << iter << " iterations." << endl;
            break;
        }
    }
}

int main() {
    vector<vector<double>> grid(M, vector<double>(N, 0.0));
    initializeGrid(grid);
    updateRedBlack(grid);

    return 0;
}