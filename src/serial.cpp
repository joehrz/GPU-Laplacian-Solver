#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>


using namespace std;

void sorMethod(vector<vector<double>> &phi, const vector<vector<double>> &a,
                const vector<vector<double>> &b, const vector<vector<double>> &c,
                const vector<vector<double>> &d, const vector<vector<double>> &e,
                const vector<vector<double>> &f, double omega, double tol, int maxIter){


    int M = phi.size();
    int N = phi[0].size();

    vector<vector<double>> r(M, vector<double>(N, 0.0));

    for (int n = 0; n < maxIter; n++){
        for (int i = 1; i < M-1; i++){
            for (int j = 1; j < N - 1; ++j){

                r[i][j] = f[i][j] - (a[i][j] * phi[i+1][j] + b[i][j] * phi[i-1][j] +
                                    c[i][j] * phi[i][j+1] + d[i][j] * phi[i][j-1] + e[i][j] * phi[i][j]);

                phi[i][j] += omega * r[i][j] / e[i][j];

            }


        }
    

        double norm_r = 0.0;
        double norm_f = 0.0;

        for (int i = 0; i < M; i++){
            for (int j = 0; j < N; ++j){
                norm_r += r[i][j] * r[i][j];
                norm_f += f[i][j] * f[i][j];
            }
        }

        if (sqrt(norm_r) / sqrt(norm_f) <  tol){
            cout << "Converged after " << n + 1 << " iterations." << endl;
            return;
        }
        cout << "Reached maximum iterations." << endl;
    }
}



int main() {
    int M = 100, N = 100;
    double omega = 1.9;
    double tol = 1e-6;
    int maxIter = 10000;

    vector<vector<double>> phi(M, vector<double>(N, 0)), f(M, vector<double>(N, 0));
    vector<vector<double>> a(M, vector<double>(N, 1)), b(M, vector<double>(N, 1)),
                           c(M, vector<double>(N, 1)), d(M, vector<double>(N, 1)),
                           e(M, vector<double>(N, -4));

    // Setting boundary conditions
    for (int j = 0; j < N; j++) {
        phi[0][j] = 0;     // Bottom boundary
        phi[M-1][j] = 0;   // Top boundary
    }
    for (int i = 0; i < M; i++) {
        phi[i][0] = 0;     // Left boundary
        phi[i][N-1] = 100; // Right boundary
    }

    sorMethod(phi, a, b, c, d, e, f, omega, tol, maxIter);

    // Plotting with gnuplot
    FILE *gnuplotPipe = popen("gnuplot -persist", "w");
    if (!gnuplotPipe) {
        cerr << "Gnuplot not found." << endl;
        return -1;
    }

    fprintf(gnuplotPipe, "set pm3d map\n");
    fprintf(gnuplotPipe, "set palette defined (0 'blue', 1 'white', 2 'red')\n");
    fprintf(gnuplotPipe, "splot '-' matrix with image\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(gnuplotPipe, "%lf ", phi[i][j]);
        }
        fprintf(gnuplotPipe, "\n"); // End of row
    }
    fprintf(gnuplotPipe, "e\n"); // End of data block
    fflush(gnuplotPipe);

    return 0;
}